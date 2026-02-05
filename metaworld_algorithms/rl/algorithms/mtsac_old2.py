"""Inspired by https://github.com/kevinzakka/robopianist-rl/blob/main/sac.py"""

import dataclasses
from functools import partial
from typing import Self, override

import flax.linen as nn
import gymnasium as gym
import jax
import jax.flatten_util as flatten_util
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
from flax import struct
from flax.core import FrozenDict
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from metaworld_algorithms.config.envs import EnvConfig
from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
)
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import AlgorithmConfig, OffPolicyTrainingConfig
from metaworld_algorithms.optim.pcgrad import PCGradState
from metaworld_algorithms.rl.buffers import MultiTaskReplayBuffer, InterTaskMultiTaskReplayBuffer
from metaworld_algorithms.rl.networks import (
    ContinuousActionPolicy,
    Ensemble,
    QValueFunction,
)
from metaworld_algorithms.types import (
    Action,
    LogDict,
    Observation,
    ReplayBufferSamples,
)

from .base import OffPolicyAlgorithm
from .utils import TrainState


class MultiTaskTemperature(nn.Module):
    num_tasks: int
    initial_temperature: float = 1.0

    def setup(self):
        self.log_alpha = self.param(
            "log_alpha",
            init_fn=lambda _: jnp.full(
                (self.num_tasks,), jnp.log(self.initial_temperature)
            ),
        )

    def __call__(
        self, task_ids: Float[Array, "... num_tasks"]
    ) -> Float[Array, "... 1"]:
        return jnp.exp(task_ids @ self.log_alpha.reshape(-1, 1))


class CriticTrainState(TrainState):
    target_params: FrozenDict | None = None


@jax.jit
def _sample_action(
    actor: TrainState, observation: Observation, key: PRNGKeyArray
) -> tuple[Float[Array, "... action_dim"], PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist = actor.apply_fn(actor.params, observation)
    action = dist.sample(seed=action_key)
    return action, key


@jax.jit
def _eval_action(
    actor: TrainState, observation: Observation
) -> Float[Array, "... action_dim"]:
    return actor.apply_fn(actor.params, observation).mode()


def extract_task_weights(
    alpha_params: FrozenDict, task_ids: Float[np.ndarray, "... num_tasks"]
) -> Float[Array, "... 1"]:
    log_alpha: jax.Array
    task_weights: jax.Array

    log_alpha = alpha_params["params"]["log_alpha"]  # pyright: ignore [reportAssignmentType]
    task_weights = jax.nn.softmax(-log_alpha)
    task_weights = task_ids @ task_weights.reshape(-1, 1)  # pyright: ignore [reportAssignmentType]
    task_weights *= log_alpha.shape[0]
    return task_weights


@dataclasses.dataclass(frozen=True)
class MTSACConfig(AlgorithmConfig):
    actor_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    critic_config: QValueFunctionConfig = QValueFunctionConfig()
    temperature_optimizer_config: OptimizerConfig = OptimizerConfig(max_grad_norm=None)
    initial_temperature: float = 1.0
    num_critics: int = 2
    tau: float = 0.005
    use_task_weights: bool = False
    max_q_value: float | None = 5000
    use_inter_task_sampling: bool = False
    use_intra_task_sampling: bool = False
    inter_sampling_beta: float = 1.0

    # (3) Success-based IL (soft blending; no hard threshold / no warmup)
    use_success_based_il: bool = False

    # success EMA update
    success_ema_tau: float = 0.01

    # IL weight (lambda) scheduler
    il_weight_mode: str = "sigmoid"   # "sigmoid" | "minmax"
    il_weight_temp: float = 0.1       # sigmoid용 temperature
    il_weight_power: float = 2.0      # minmax용 power

    # BC loss type + scaling
    il_loss_type: str = "mse"         # "mes" 권장 (NLL 등 확장 가능)
    il_coef: float = 1.0
    il_qfilter_top_p: float = 0.20     # SB3 TOP_P
    il_qfilter_min_good: int = 8       # SB3 good_cnt < 8 skip


class MTSAC(OffPolicyAlgorithm[MTSACConfig]):
    actor: TrainState
    critic: CriticTrainState
    alpha: TrainState
    key: PRNGKeyArray
    gamma: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    use_task_weights: bool = struct.field(pytree_node=False)
    split_actor_losses: bool = struct.field(pytree_node=False)
    split_critic_losses: bool = struct.field(pytree_node=False)
    num_critics: int = struct.field(pytree_node=False)
    max_q_value: float | None = struct.field(pytree_node=False)
    use_inter_task_sampling: bool = struct.field(pytree_node=False)
    use_intra_task_sampling: bool = struct.field(pytree_node=False)
    inter_sampling_beta: float = struct.field(pytree_node=False)
    # IL-RL Hyperparameters
    use_success_based_il: bool = struct.field(pytree_node=False)
    success_ema_tau: float = struct.field(pytree_node=False)
    il_weight_mode: str = struct.field(pytree_node=False)
    il_weight_temp: float = struct.field(pytree_node=False)
    il_weight_power: float = struct.field(pytree_node=False)
    il_loss_type: str = struct.field(pytree_node=False)
    il_coef: float = struct.field(pytree_node=False)
    il_qfilter_top_p: float = struct.field(pytree_node=False)
    il_qfilter_min_good: int = struct.field(pytree_node=False)
    success_ema: jax.Array

    @override
    @staticmethod
    def initialize(
        config: MTSACConfig, env_config: EnvConfig, seed: int = 1
    ) -> "MTSAC":
        assert isinstance(env_config.action_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )
        assert isinstance(env_config.observation_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )

        master_key = jax.random.PRNGKey(seed)
        algorithm_key, actor_init_key, critic_init_key, alpha_init_key = (
            jax.random.split(master_key, 4)
        )

        actor_net = ContinuousActionPolicy(
            int(np.prod(env_config.action_space.shape)), config=config.actor_config
        )
        dummy_obs = jnp.array(
            [env_config.observation_space.sample() for _ in range(config.num_tasks)]
        )
        actor = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(actor_init_key, dummy_obs),
            tx=config.actor_config.network_config.optimizer.spawn(),
        )

        critic_cls = partial(QValueFunction, config=config.critic_config)
        critic_net = Ensemble(critic_cls, num=config.num_critics)
        dummy_action = jnp.array(
            [env_config.action_space.sample() for _ in range(config.num_tasks)]
        )
        critic_init_params = critic_net.init(critic_init_key, dummy_obs, dummy_action)
        critic = CriticTrainState.create(
            apply_fn=critic_net.apply,
            params=critic_init_params,
            target_params=critic_init_params,
            tx=config.critic_config.network_config.optimizer.spawn(),
        )

        alpha_net = MultiTaskTemperature(config.num_tasks, config.initial_temperature)
        dummy_task_ids = jnp.array(
            [np.ones((config.num_tasks,)) for _ in range(config.num_tasks)]
        )
        alpha = TrainState.create(
            apply_fn=alpha_net.apply,
            params=alpha_net.init(alpha_init_key, dummy_task_ids),
            tx=config.temperature_optimizer_config.spawn(),
        )

        target_entropy = -np.prod(env_config.action_space.shape).item()

        return MTSAC(
            num_tasks=config.num_tasks,
            actor=actor,
            critic=critic,
            alpha=alpha,
            key=algorithm_key,
            gamma=config.gamma,
            tau=config.tau,
            target_entropy=target_entropy,
            use_task_weights=config.use_task_weights,
            num_critics=config.num_critics,
            split_actor_losses=config.actor_config.network_config.optimizer.requires_split_task_losses,
            split_critic_losses=config.critic_config.network_config.optimizer.requires_split_task_losses,
            max_q_value=config.max_q_value,
            use_inter_task_sampling = config.use_inter_task_sampling,
            use_intra_task_sampling = config.use_intra_task_sampling,
            inter_sampling_beta = config.inter_sampling_beta,
            # IL-RL Hyperparameters
            use_success_based_il = config.use_success_based_il,
            success_ema_tau = config.success_ema_tau,
            il_weight_mode = config.il_weight_mode,
            il_weight_temp = config.il_weight_temp,
            il_weight_power = config.il_weight_power,
            il_loss_type = config.il_loss_type,
            il_coef = config.il_coef,   
            success_ema = jnp.zeros(config.num_tasks),
            il_qfilter_top_p = config.il_qfilter_top_p,
            il_qfilter_min_good = config.il_qfilter_min_good,
        )

    @override
    def spawn_replay_buffer(
        self, env_config: EnvConfig, config: OffPolicyTrainingConfig, seed: int = 1
    ) -> MultiTaskReplayBuffer:
        if self.use_inter_task_sampling:
            return InterTaskMultiTaskReplayBuffer(
                total_capacity=config.buffer_size,
                num_tasks=self.num_tasks,
                env_obs_space=env_config.observation_space,
                env_action_space=env_config.action_space,
                seed=seed,
                max_steps=500,
                task_beta=self.inter_sampling_beta,
                use_intra_task_sampling=self.use_intra_task_sampling,
            )
        else:
            return MultiTaskReplayBuffer(
                total_capacity=config.buffer_size,
                num_tasks=self.num_tasks,
                env_obs_space=env_config.observation_space,
                env_action_space=env_config.action_space,
                seed=seed,
        )

    @override
    def get_num_params(self) -> dict[str, int]:
        return {
            "actor_num_params": sum(x.size for x in jax.tree.leaves(self.actor.params)),
            "critic_num_params": sum(
                x.size for x in jax.tree.leaves(self.critic.params)
            ),
        }

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        action, key = _sample_action(self.actor, observation, self.key)
        return self.replace(key=key), jax.device_get(action)

    @override
    def eval_action(self, observations: Observation) -> Action:
        return jax.device_get(_eval_action(self.actor, observations))

    def split_data_by_tasks(
        self,
        data: PyTree[Float[Array, "batch data_dim"]],
        task_ids: Float[npt.NDArray, "batch num_tasks"],
    ) -> PyTree[Float[Array, "num_tasks per_task_batch data_dim"]]:
        tasks = jnp.argmax(task_ids, axis=1)
        sorted_indices = jnp.argsort(tasks)

        def group_by_task_leaf(
            leaf: Float[Array, "batch data_dim"],
        ) -> Float[Array, "task task_batch data_dim"]:
            leaf_sorted = leaf[sorted_indices]
            return leaf_sorted.reshape(self.num_tasks, -1, leaf.shape[1])

        return jax.tree.map(group_by_task_leaf, data), sorted_indices

    def unsplit_data_by_tasks(
        self,
        split_data: PyTree[Float[Array, "num_tasks per_task_batch data_dim"]],
        sort_indices: jax.Array,
    ) -> PyTree[Float[Array, "batch data_dim"]]:
        def reconstruct_leaf(
            leaf: Float[Array, "num_tasks per_task_batch data_dim"],
        ) -> Float[Array, "batch data_dim"]:
            batch_size = leaf.shape[0] * leaf.shape[1]
            flat = leaf.reshape(batch_size, leaf.shape[-1])
            # Create inverse permutation
            inverse_indices = jnp.zeros_like(sort_indices)
            inverse_indices = inverse_indices.at[sort_indices].set(
                jnp.arange(batch_size)
            )
            return flat[inverse_indices]

        return jax.tree.map(reconstruct_leaf, split_data)

    def update_critic(
        self,
        data: ReplayBufferSamples,
        alpha_val: Float[Array, "*batch 1"],
        task_weights: Float[Array, "*batch 1"] | None = None,
    ) -> tuple[Self, LogDict]:
        key, critic_loss_key = jax.random.split(self.key)

        # Sample a'
        if self.split_critic_losses:
            next_actions, next_action_log_probs = jax.vmap(
                lambda x: self.actor.apply_fn(self.actor.params, x).sample_and_log_prob(
                    seed=critic_loss_key
                )
            )(data.observations)
            q_values = jax.vmap(self.critic.apply_fn, in_axes=(None, 0, 0))(
                self.critic.target_params, data.next_observations, next_actions
            )
        else:
            next_actions, next_action_log_probs = self.actor.apply_fn(
                self.actor.params, data.next_observations
            ).sample_and_log_prob(seed=critic_loss_key)
            q_values = self.critic.apply_fn(
                self.critic.target_params, data.next_observations, next_actions
            )

        def critic_loss(
            params: FrozenDict,
            _data: ReplayBufferSamples,
            _q_values: Float[Array, "#batch 1"],
            _alpha_val: Float[Array, "#batch 1"],
            _next_action_log_probs: Float[Array, " #batch"],
            _task_weights: Float[Array, "#batch 1"] | None = None,
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            # next_action_log_probs is (B,) shaped because of the sum(axis=1), while Q values are (B, 1)
            min_qf_next_target = jnp.min(
                _q_values, axis=0
            ) - _alpha_val * _next_action_log_probs.reshape(-1, 1)

            next_q_value = jax.lax.stop_gradient(
                _data.rewards + (1 - _data.dones) * self.gamma * min_qf_next_target
            )

            q_pred = self.critic.apply_fn(params, _data.observations, _data.actions)

            if self.max_q_value is not None:
                # HACK: Clipping Q values to approximate theoretical maximum for Metaworld
                next_q_value = jnp.clip(
                    next_q_value, -self.max_q_value, self.max_q_value
                )
                q_pred = jnp.clip(q_pred, -self.max_q_value, self.max_q_value)

            if _task_weights is not None:
                loss = (_task_weights * (q_pred - next_q_value) ** 2).mean()
            else:
                loss = ((q_pred - next_q_value) ** 2).mean()

            td = q_pred - next_q_value  # broadcast
            td_err = jnp.max(jnp.abs(td), axis=0).reshape(-1)  # (B,)
            
            return loss, (q_pred.mean(), td_err)

        if self.split_critic_losses:
            (critic_loss_value, (qf_values, td_errs)), critic_grads = jax.vmap(
                jax.value_and_grad(critic_loss, has_aux=True),
                in_axes=(None, 0, 0, 0, 0, 0),
                out_axes=0,
            )(
                self.critic.params,
                data,
                q_values,
                alpha_val,
                next_action_log_probs,
                task_weights,
            )
            td_errs = td_errs.reshape(-1)
            flat_grads, _ = flatten_util.ravel_pytree(
                jax.tree.map(lambda x: x.mean(axis=0), critic_grads)
            )
        else:
            (critic_loss_value, (qf_values, td_errs)), critic_grads = jax.value_and_grad(
                critic_loss, has_aux=True
            )(
                self.critic.params,
                data,
                q_values,
                alpha_val,
                next_action_log_probs,
                task_weights,
            )
            flat_grads, _ = flatten_util.ravel_pytree(critic_grads)

        key, optimizer_key = jax.random.split(key)
        critic = self.critic.apply_gradients(
            grads=critic_grads,
            optimizer_extra_args={
                "task_losses": critic_loss_value,
                "key": optimizer_key,
            },
        )
        critic = critic.replace(
            target_params=optax.incremental_update(
                critic.params,
                critic.target_params,  # pyright: ignore [reportArgumentType]
                self.tau,
            )
        )
        flat_params_crit, _ = flatten_util.ravel_pytree(critic.params)

        return self.replace(critic=critic, key=key), {
            "losses/qf_values": qf_values.mean(),
            "losses/qf_loss": critic_loss_value.mean(),
            "metrics/critic_grad_magnitude": jnp.linalg.norm(flat_grads),
            "metrics/critic_params_norm": jnp.linalg.norm(flat_params_crit),
        }, td_errs

    def compute_il_lambda_from_success_ema(self, success_ema, task_onehot):
        task_idx = jnp.argmax(task_onehot, axis=1)
        s = success_ema[task_idx].reshape(-1, 1)

        lam = jax.nn.sigmoid(s / self.il_weight_temp)
        return lam

    def update_actor(
        self,
        data: ReplayBufferSamples,
        alpha_val: Float[Array, "batch 1"],
        task_weights: Float[Array, "batch 1"] | None = None,
        success_ema: jax.Array | None = None,
    ) -> tuple[Self, Float[Array, " batch"], LogDict]:
        key, actor_loss_key = jax.random.split(self.key)

        def actor_loss(
            params: FrozenDict,
            _data: ReplayBufferSamples,
            _alpha_val: Float[Array, "batch 1"],
            _task_weights: Float[Array, "batch 1"] | None,
            _success_ema: jax.Array | None,
            _il_data: ReplayBufferSamples | None,
        ):
            # -------------------------
            # (1) SAC actor loss (RL batch)
            # -------------------------
            dist = self.actor.apply_fn(params, _data.observations)
            action_samples, log_probs = dist.sample_and_log_prob(seed=actor_loss_key)
            log_probs = log_probs.reshape(-1, 1)

            q_values = self.critic.apply_fn(
                self.critic.params, _data.observations, action_samples
            )
            min_qf_values = jnp.min(q_values, axis=0)

            rl_term = (_alpha_val * log_probs - min_qf_values)

            if _task_weights is not None:
                rl_loss = (_task_weights * rl_term).mean()
            else:
                rl_loss = rl_term.mean()

            # -------------------------
            # (2) BC loss from success buffer (optional)
            # -------------------------
            bc_loss = jnp.array(0.0, dtype=rl_loss.dtype)
            lam_mean = jnp.array(0.0, dtype=rl_loss.dtype)

            use_il = (
                self.use_success_based_il
                and (_success_ema is not None)
                and (_il_data is not None)
            )

            def _compute_bc(_params, _il_data_local, _success_ema_local):
                dist_il = self.actor.apply_fn(_params, _il_data_local.observations)

                if self.il_loss_type == "nll":
                    logp_demo = dist_il.log_prob(_il_data_local.actions)
                    if logp_demo.ndim == 1:
                        logp_demo_ = logp_demo.reshape(-1, 1)
                    else:
                        logp_demo_ = logp_demo
                    nll = -logp_demo_  # (B,1)

                    il_task_onehot = _il_data_local.observations[..., -self.num_tasks:]
                    lam = self.compute_il_lambda_from_success_ema(
                        _success_ema_local, il_task_onehot
                    )  # (B,1)
                    q_ens = self.critic.apply_fn(self.critic.params,
                                 _il_data_local.observations,
                                 _il_data_local.actions)     # (N,B,1)

                    q_min = jnp.min(q_ens, axis=0)  # (B,1)
                    q_flat = q_min.reshape(-1)      # (B,)

                    thr = jnp.quantile(q_flat, 1.0 - self.il_qfilter_top_p)  # scalar
                    good = (q_flat >= thr)                                   # (B,) bool
                    good_f = good.astype(nll.dtype).reshape(-1, 1)           # (B,1)

                    good_cnt = jnp.sum(good.astype(jnp.int32))
                        # SB3: good_cnt < 8 이면 IL skip
                    def _use_filtered():
                        bc = jnp.sum(lam * nll * good_f) / (jnp.sum(good_f) + 1e-8)
                        return bc

                    def _skip():
                        # IL을 완전히 안 하려면 0을 반환 (SB3와 가장 유사)
                        return jnp.array(0.0, dtype=nll.dtype)

                    bc_loss = jax.lax.cond(good_cnt >= self.il_qfilter_min_good,
                                        lambda _: _use_filtered(),
                                        lambda _: _skip(),
                                        operand=None)
                    # 로그용: lam_mean은 기존처럼
                    lam_mean = lam.mean()

                    # (선택) 디버그 로그용 통계도 같이 반환하고 싶으면 aux에 넣는 방식 추천
                    # return bc_loss, lam_mean, thr, q_flat.mean(), jnp.where(good, q_flat, 0.0).sum() / (good_cnt + 1e-8), good_cnt
                    return bc_loss, lam_mean
                elif self.il_loss_type == "mse":
                    # (1) deterministic pred action (SB3: actor(obs, deterministic=True))
                    pred = dist_il.mode()  # shape: (B, act_dim)

                    # (2) per-sample MSE (SB3: ((pred-act)^2).mean(dim=1))
                    per = jnp.mean((pred - _il_data_local.actions) ** 2, axis=-1, keepdims=True)  # (B,1)

                    # (3) lambda from success_ema (keep your current function or replace w/ minmax)
                    il_task_onehot = _il_data_local.observations[..., -self.num_tasks:]
                    lam = self.compute_il_lambda_from_success_ema(_success_ema_local, il_task_onehot)  # (B,1)

                    # (4) Q-filter top-p among successful (same as your code)
                    q_ens = self.critic.apply_fn(
                        self.critic.params,
                        _il_data_local.observations,
                        _il_data_local.actions,
                    )  # (N,B,1)
                    q_min = jnp.min(q_ens, axis=0)          # (B,1)
                    q_flat = q_min.reshape(-1)              # (B,)

                    thr = jnp.quantile(q_flat, 1.0 - self.il_qfilter_top_p)  # scalar
                    good = (q_flat >= thr)                                   # (B,) bool
                    good_f = good.astype(per.dtype).reshape(-1, 1)           # (B,1)

                    good_cnt = jnp.sum(good.astype(jnp.int32))

                    def _use_filtered(_):
                        bc = jnp.sum(lam * per * good_f) / (jnp.sum(good_f) + 1e-8)
                        return bc

                    def _skip(_):
                        return jnp.array(0.0, dtype=per.dtype)

                    bc_loss = jax.lax.cond(
                        good_cnt >= self.il_qfilter_min_good,
                        _use_filtered,
                        _skip,
                        operand=None,
                    )

                    lam_mean = lam.mean()
                    return bc_loss, lam_mean
                else:
                    raise NotImplementedError(f"Unknown il_loss_type: {self.il_loss_type}")

            bc_loss, lam_mean = jax.lax.cond(
                use_il,
                lambda _: _compute_bc(params, _il_data, _success_ema),
                lambda _: (bc_loss, lam_mean),
                operand=None,
            )

            total_loss = rl_loss

            # aux로 log_probs를 뽑아서 alpha update에 씀
            aux = (log_probs, rl_loss)
            return total_loss, aux

        if self.split_actor_losses:
            # vmap: data/alpha/task_weights는 task로 split되어 들어오고,
            # success_ema, il_data는 모든 task에 동일 broadcast (in_axes=None)
            (loss_values, aux_values), actor_grads = jax.vmap(
                jax.value_and_grad(actor_loss, has_aux=True),
                in_axes=(None, 0, 0, 0, None, None),
                out_axes=0,
            )(self.actor.params, data, alpha_val, task_weights, success_ema, il_data)

            # loss_values: (num_tasks,)
            actor_loss_value = loss_values.mean()

            # aux_values: tuple(log_probs, rl_loss, bc_loss, lam_mean) each with leading task dim
            log_probs, rl_losses, bc_losses, lam_means = aux_values
            # log_probs shape: (num_tasks, per_task_batch, 1)
            # -> flatten back to (batch, 1) for alpha update
            # NOTE: split_actor_losses=True이면 나중에 unsplit_data_by_tasks로 복원하는 기존 로직이 있으니
            # 여기서는 그냥 반환용으로 task-wise log_probs를 유지해도 되지만,
            # 현재 코드 흐름에 맞춰 "그대로" 넘기려면 log_probs를 유지하고,
            # _update_inner에서 unsplit하는 방식을 그대로 쓰면 됨.
            # 아래는 기존 방식 유지: log_probs 그대로 반환
            flat_grads, _ = flatten_util.ravel_pytree(
                jax.tree.map(lambda x: x.mean(axis=0), actor_grads)
            )

            # logs용 scalar
            rl_loss_scalar = rl_losses.mean()
            bc_loss_scalar = bc_losses.mean()
            lam_mean_scalar = lam_means.mean()

        else:
            (actor_loss_value, (log_probs, rl_loss_scalar, bc_loss_scalar, lam_mean_scalar)), actor_grads = (
                jax.value_and_grad(actor_loss, has_aux=True)(
                    self.actor.params, data, alpha_val, task_weights, success_ema, il_data
                )
            )
            flat_grads, _ = flatten_util.ravel_pytree(actor_grads)

        key, optimizer_key = jax.random.split(key)
        actor = self.actor.apply_gradients(
            grads=actor_grads,
            optimizer_extra_args={
                "task_losses": actor_loss_value,
                "key": optimizer_key,
            },
        )

        flat_params_act, _ = flatten_util.ravel_pytree(actor.params)
        logs = {
            "losses/actor_loss": actor_loss_value,
            "losses/actor_rl": rl_loss_scalar,
            "losses/actor_bc": bc_loss_scalar,
            "il/lam_mean": lam_mean_scalar,
            "metrics/actor_grad_magnitude": jnp.linalg.norm(flat_grads),
            "metrics/actor_params_norm": jnp.linalg.norm(flat_params_act),
        }

        return (self.replace(actor=actor, key=key), log_probs, logs)

def update_actor_il(
    self,
    il_data: ReplayBufferSamples,
    success_ema: jax.Array,
) -> tuple[Self, LogDict]:
    # PyTorch: pred = actor(obs, deterministic=True) + MSE + Q-filter(top-p) + good_cnt skip + lambda weighting
    key = self.key

    def bc_loss_fn(params: FrozenDict):
        dist_il = self.actor.apply_fn(params, il_data.observations)

        # (1) deterministic action
        pred = dist_il.mode()  # (B, act_dim)

        # (2) per-sample MSE: (B,1)
        per = jnp.mean((pred - il_data.actions) ** 2, axis=-1, keepdims=True)

        # (3) lambda from success_ema (일단 기존 함수 그대로 사용; 다음 단계에서 PyTorch식으로 바꿀 수 있음)
        il_task_onehot = il_data.observations[..., -self.num_tasks:]
        lam = self.compute_il_lambda_from_success_ema(success_ema, il_task_onehot)  # (B,1)

        # (4) Q-filter top-p among IL batch (success pool에서 뽑혔다고 가정)
        q_ens = self.critic.apply_fn(
            self.critic.params,
            il_data.observations,
            il_data.actions,
        )  # (N,B,1)
        q_min = jnp.min(q_ens, axis=0)     # (B,1)
        q_flat = q_min.reshape(-1)         # (B,)

        thr = jnp.quantile(q_flat, 1.0 - self.il_qfilter_top_p)
        good = (q_flat >= thr)                               # (B,)
        good_f = good.astype(per.dtype).reshape(-1, 1)       # (B,1)
        good_cnt = jnp.sum(good.astype(jnp.int32))

        def _use_filtered(_):
            # PyTorch: (lam * mask * per).sum() / (mask.sum()+eps)
            return jnp.sum(lam * good_f * per) / (jnp.sum(good_f) + 1e-8)

        def _skip(_):
            return jnp.array(0.0, dtype=per.dtype)

        bc = jax.lax.cond(good_cnt >= self.il_qfilter_min_good, _use_filtered, _skip, operand=None)
        return bc, (thr, q_flat.mean(), jnp.where(good, q_flat, 0.0).sum() / (good_cnt + 1e-8), good_cnt, lam.mean())

    (bc_loss, (thr, q_mean_all, q_mean_good, good_cnt, lam_mean)), grads = jax.value_and_grad(bc_loss_fn, has_aux=True)(self.actor.params)

    # PyTorch는 actor optimizer로만 BC step을 함
    actor = self.actor.apply_gradients(
        grads=jax.tree.map(lambda g: self.il_coef * g, grads),  # il_coef 곱해줌
        optimizer_extra_args={"task_losses": bc_loss, "key": key},
    )

    logs = {
        "il/active": jnp.array(1.0, dtype=bc_loss.dtype),
        "il/loss_bc": bc_loss,
        "il/q_thr": thr,
        "il/q_mean_all": q_mean_all,
        "il/q_mean_good": q_mean_good,
        "il/good_cnt": good_cnt.astype(jnp.float32),
        "il/lam_mean": lam_mean,
    }

    return self.replace(actor=actor, key=key), logs

    def update_alpha(
        self,
        log_probs: Float[Array, " batch"],
        task_ids: Float[npt.NDArray, " batch num_tasks"],
    ) -> tuple[Self, LogDict]:
        def alpha_loss(params: FrozenDict) -> Float[Array, ""]:
            log_alpha: jax.Array
            log_alpha = task_ids @ params["params"]["log_alpha"].reshape(-1, 1)  # pyright: ignore [reportAttributeAccessIssue]
            return (-log_alpha * (log_probs + self.target_entropy)).mean()

        alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss)(
            self.alpha.params
        )
        alpha = self.alpha.apply_gradients(grads=alpha_grads)

        return self.replace(alpha=alpha), {
            "losses/alpha_loss": alpha_loss_value,
            "alpha": jnp.exp(alpha.params["params"]["log_alpha"]).sum(),  # pyright: ignore [reportArgumentType]
        }

    @jax.jit
    def _update_inner(self, data: ReplayBufferSamples, success_ema: jax.Array, il_data: ReplayBufferSamples | None = None) -> tuple[Self, LogDict, jax.Array]:
        task_ids = data.observations[..., -self.num_tasks :]

        alpha_vals = self.alpha.apply_fn(self.alpha.params, task_ids)
        if self.use_task_weights:
            task_weights = extract_task_weights(self.alpha.params, task_ids)
        else:
            task_weights = None

        actor_data = critic_data = data
        actor_alpha_vals = critic_alpha_vals = alpha_vals
        actor_task_weights = critic_task_weights = task_weights
        alpha_val_indices = None

        if self.split_critic_losses or self.split_actor_losses:
            split_data, _ = self.split_data_by_tasks(data, task_ids)
            split_alpha_vals, alpha_val_indices = self.split_data_by_tasks(
                alpha_vals, task_ids
            )
            split_task_weights, _ = (
                self.split_data_by_tasks(task_weights, task_ids)
                if task_weights is not None
                else (None, None)
            )

            if self.split_critic_losses:
                critic_data = split_data
                critic_alpha_vals = split_alpha_vals
                critic_task_weights = split_task_weights

            if self.split_actor_losses:
                actor_data = split_data
                actor_alpha_vals = split_alpha_vals
                actor_task_weights = split_task_weights

        self, critic_logs, td_errs = self.update_critic(
            critic_data, critic_alpha_vals, critic_task_weights
        )
        self, log_probs, actor_logs = self.update_actor(
            actor_data, actor_alpha_vals, actor_task_weights,
        )
        if self.split_actor_losses:
            assert alpha_val_indices is not None
            log_probs = self.unsplit_data_by_tasks(log_probs, alpha_val_indices)
        self, alpha_logs = self.update_alpha(log_probs, task_ids)
        # (3) BC-only actor update (PyTorch의 extra step)
        il_logs = {}
        if self.use_success_based_il and (il_data is not None) and (success_ema is not None):
            self, il_logs = self.update_actor_il(il_data, success_ema)
        else:
            il_logs = {"il/active": jnp.array(0.0)}

        # HACK: PCGrad logs
        assert isinstance(self.critic.opt_state, tuple)
        assert isinstance(self.actor.opt_state, tuple)
        critic_optim_logs = (
            {
                f"metrics/critic_{key}": value
                for key, value in self.critic.opt_state[0]._asdict().items()
            }
            if isinstance(self.critic.opt_state[0], PCGradState)
            else {}
        )
        actor_optim_logs = (
            {
                f"metrics/actor_{key}": value
                for key, value in self.actor.opt_state[0]._asdict().items()
            }
            if isinstance(self.actor.opt_state[0], PCGradState)
            else {}
        )

        return self, {
            **critic_logs,
            **actor_logs,
            **alpha_logs,
            **il_logs,
            **critic_optim_logs,
            **actor_optim_logs,
        }, td_errs

    @override
    def update(self, data: ReplayBufferSamples, success_ema: jax.Array | None = None, il_data: ReplayBufferSamples | None = None) -> tuple[Self, LogDict, jax.Array]:
        return self._update_inner(data, success_ema, il_data)
