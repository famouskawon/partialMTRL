"""
Inspired by https://github.com/kevinzakka/robopianist-rl/blob/main/sac.py

MTSAC (JAX/Flax) with PyTorch(SB3)-style 2-step RL+IL update:
  1) SAC critic update
  2) SAC actor (RL-only) update
  3) alpha update (from RL actor log_probs)
  4) extra BC-only actor update (success-based IL), Q-filtered top-p
"""

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


# ============================================================
# Temperature (alpha) per task
# ============================================================
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

    def __call__(self, task_ids: Float[Array, "... num_tasks"]) -> Float[Array, "... 1"]:
        # task_ids is one-hot; returns exp(log_alpha[task])
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
def _eval_action(actor: TrainState, observation: Observation) -> Float[Array, "... action_dim"]:
    return actor.apply_fn(actor.params, observation).mode()


def extract_task_weights(
    alpha_params: FrozenDict, task_ids: Float[np.ndarray, "... num_tasks"]
) -> Float[Array, "... 1"]:
    # Used by original library for task weighting; kept as-is.
    log_alpha = alpha_params["params"]["log_alpha"]  # pyright: ignore [reportAssignmentType]
    task_weights = jax.nn.softmax(-log_alpha)
    task_weights = task_ids @ task_weights.reshape(-1, 1)  # pyright: ignore [reportAssignmentType]
    task_weights *= log_alpha.shape[0]
    return task_weights


# ============================================================
# Config
# ============================================================
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

    # Success-based IL
    use_success_based_il: bool = False
    success_ema_tau: float = 0.01

    # IL weight (lambda) scheduler
    il_weight_mode: str = "sigmoid"   # "sigmoid" | "minmax"
    il_weight_temp: float = 0.1       # sigmoid temperature
    il_weight_power: float = 2.0      # reserved for "minmax"

    # BC loss type + scaling
    il_loss_type: str = "mse"         # "mse" | "nll"
    il_coef: float = 1.0
    il_qfilter_top_p: float = 0.2
    il_qfilter_min_good: int = 8

    # MTSACConfig에 추가
    il_use_shared: bool = False
    il_shared_coef: float = 1.0
    il_shared_min_tasks: int = 2
    il_shared_batch_frac: float = 0.5  # IL 배치 중 shared 비율 (0~1)


# ============================================================
# Algorithm
# ============================================================
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

    success_ema: jax.Array  # stored for convenience; typically updated in trainer/buffer logic

    # ---------------- init ----------------
    @override
    @staticmethod
    def initialize(config: MTSACConfig, env_config: EnvConfig, seed: int = 1) -> "MTSAC":
        assert isinstance(env_config.action_space, gym.spaces.Box), "Non-box spaces currently not supported."
        assert isinstance(env_config.observation_space, gym.spaces.Box), "Non-box spaces currently not supported."

        master_key = jax.random.PRNGKey(seed)
        algorithm_key, actor_init_key, critic_init_key, alpha_init_key = jax.random.split(master_key, 4)

        actor_net = ContinuousActionPolicy(
            int(np.prod(env_config.action_space.shape)),
            config=config.actor_config,
        )
        dummy_obs = jnp.array([env_config.observation_space.sample() for _ in range(config.num_tasks)])
        actor = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(actor_init_key, dummy_obs),
            tx=config.actor_config.network_config.optimizer.spawn(),
        )

        critic_cls = partial(QValueFunction, config=config.critic_config)
        critic_net = Ensemble(critic_cls, num=config.num_critics)
        dummy_action = jnp.array([env_config.action_space.sample() for _ in range(config.num_tasks)])
        critic_init_params = critic_net.init(critic_init_key, dummy_obs, dummy_action)
        critic = CriticTrainState.create(
            apply_fn=critic_net.apply,
            params=critic_init_params,
            target_params=critic_init_params,
            tx=config.critic_config.network_config.optimizer.spawn(),
        )

        alpha_net = MultiTaskTemperature(config.num_tasks, config.initial_temperature)
        dummy_task_ids = jnp.array([np.ones((config.num_tasks,)) for _ in range(config.num_tasks)])
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
            use_inter_task_sampling=config.use_inter_task_sampling,
            use_intra_task_sampling=config.use_intra_task_sampling,
            inter_sampling_beta=config.inter_sampling_beta,
            use_success_based_il=config.use_success_based_il,
            success_ema_tau=config.success_ema_tau,
            il_weight_mode=config.il_weight_mode,
            il_weight_temp=config.il_weight_temp,
            il_weight_power=config.il_weight_power,
            il_loss_type=config.il_loss_type,
            il_coef=config.il_coef,
            success_ema=jnp.zeros(config.num_tasks),
            il_qfilter_top_p=config.il_qfilter_top_p,
            il_qfilter_min_good=config.il_qfilter_min_good,
        )

    # ---------------- replay buffer ----------------
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
            "critic_num_params": sum(x.size for x in jax.tree.leaves(self.critic.params)),
        }

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        action, key = _sample_action(self.actor, observation, self.key)
        return self.replace(key=key), jax.device_get(action)

    @override
    def eval_action(self, observations: Observation) -> Action:
        return jax.device_get(_eval_action(self.actor, observations))

    # ---------------- task splitting helpers ----------------
    def split_data_by_tasks(
        self,
        data: PyTree[Float[Array, "batch data_dim"]],
        task_ids: Float[npt.NDArray, "batch num_tasks"],
    ) -> tuple[PyTree[Float[Array, "num_tasks per_task_batch data_dim"]], jax.Array]:
        tasks = jnp.argmax(task_ids, axis=1)
        sorted_indices = jnp.argsort(tasks)

        def group_by_task_leaf(leaf: Float[Array, "batch data_dim"]) -> Float[Array, "task task_batch data_dim"]:
            leaf_sorted = leaf[sorted_indices]
            return leaf_sorted.reshape(self.num_tasks, -1, leaf.shape[1])

        return jax.tree.map(group_by_task_leaf, data), sorted_indices

    def unsplit_data_by_tasks(
        self,
        split_data: PyTree[Float[Array, "num_tasks per_task_batch data_dim"]],
        sort_indices: jax.Array,
    ) -> PyTree[Float[Array, "batch data_dim"]]:
        def reconstruct_leaf(leaf: Float[Array, "num_tasks per_task_batch data_dim"]) -> Float[Array, "batch data_dim"]:
            batch_size = leaf.shape[0] * leaf.shape[1]
            flat = leaf.reshape(batch_size, leaf.shape[-1])
            inverse_indices = jnp.zeros_like(sort_indices)
            inverse_indices = inverse_indices.at[sort_indices].set(jnp.arange(batch_size))
            return flat[inverse_indices]

        return jax.tree.map(reconstruct_leaf, split_data)

    # ============================================================
    # Critic update (SAC)
    # ============================================================
    def update_critic(
        self,
        data: ReplayBufferSamples,
        alpha_val: Float[Array, "*batch 1"],
        task_weights: Float[Array, "*batch 1"] | None = None,
    ) -> tuple[Self, LogDict, jax.Array]:
        key, critic_loss_key = jax.random.split(self.key)

        # Sample a'
        if self.split_critic_losses:
            next_actions, next_action_log_probs = jax.vmap(
                lambda x: self.actor.apply_fn(self.actor.params, x).sample_and_log_prob(seed=critic_loss_key)
            )(data.observations)
            q_values = jax.vmap(self.critic.apply_fn, in_axes=(None, 0, 0))(
                self.critic.target_params, data.next_observations, next_actions
            )
        else:
            next_actions, next_action_log_probs = self.actor.apply_fn(
                self.actor.params, data.next_observations
            ).sample_and_log_prob(seed=critic_loss_key)
            q_values = self.critic.apply_fn(self.critic.target_params, data.next_observations, next_actions)

        def critic_loss(
            params: FrozenDict,
            _data: ReplayBufferSamples,
            _q_values: Float[Array, "#batch 1"],
            _alpha_val: Float[Array, "#batch 1"],
            _next_action_log_probs: Float[Array, " #batch"],
            _task_weights: Float[Array, "#batch 1"] | None = None,
        ) -> tuple[Float[Array, ""], tuple[Float[Array, ""], Float[Array, " #batch"]]]:
            min_qf_next_target = jnp.min(_q_values, axis=0) - _alpha_val * _next_action_log_probs.reshape(-1, 1)
            next_q_value = jax.lax.stop_gradient(
                _data.rewards + (1 - _data.dones) * self.gamma * min_qf_next_target
            )

            q_pred = self.critic.apply_fn(params, _data.observations, _data.actions)

            if self.max_q_value is not None:
                next_q_value = jnp.clip(next_q_value, -self.max_q_value, self.max_q_value)
                q_pred = jnp.clip(q_pred, -self.max_q_value, self.max_q_value)

            if _task_weights is not None:
                loss = (_task_weights * (q_pred - next_q_value) ** 2).mean()
            else:
                loss = ((q_pred - next_q_value) ** 2).mean()

            td = q_pred - next_q_value
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
            flat_grads, _ = flatten_util.ravel_pytree(jax.tree.map(lambda x: x.mean(axis=0), critic_grads))
            qf_values_mean = qf_values.mean()
            critic_loss_mean = critic_loss_value.mean()
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
            qf_values_mean = qf_values
            critic_loss_mean = critic_loss_value

        key, optimizer_key = jax.random.split(key)
        critic = self.critic.apply_gradients(
            grads=critic_grads,
            optimizer_extra_args={"task_losses": critic_loss_value, "key": optimizer_key},
        )
        critic = critic.replace(
            target_params=optax.incremental_update(
                critic.params,
                critic.target_params,  # pyright: ignore [reportArgumentType]
                self.tau,
            )
        )
        flat_params_crit, _ = flatten_util.ravel_pytree(critic.params)

        logs = {
            "losses/qf_values": qf_values_mean,
            "losses/qf_loss": critic_loss_mean,
            "metrics/critic_grad_magnitude": jnp.linalg.norm(flat_grads),
            "metrics/critic_params_norm": jnp.linalg.norm(flat_params_crit),
        }
        return self.replace(critic=critic, key=key), logs, td_errs

    # ============================================================
    # IL lambda (kept as your current; can switch to PyTorch-minmax later)
    # ============================================================
    def compute_il_lambda_from_success_ema(self, success_ema: jax.Array, task_onehot: jax.Array) -> jax.Array:
        task_idx = jnp.argmax(task_onehot, axis=1)
        s = success_ema[task_idx].reshape(-1, 1)
        lam = 2.0 * jnp.clip(s, 0.0, 1.0)   # (S0=0,S1=1,LAMBDA_MAX=2)랑 동치
        # lam = jax.nn.sigmoid(s / self.il_weight_temp)
        return lam

    # ============================================================
    # Actor update (RL-only SAC)
    # ============================================================
    def update_actor(
        self,
        data: ReplayBufferSamples,
        alpha_val: Float[Array, "batch 1"],
        task_weights: Float[Array, "batch 1"] | None = None,
    ) -> tuple[Self, Float[Array, " batch"], LogDict]:
        key, actor_loss_key = jax.random.split(self.key)

        def actor_loss(
            params: FrozenDict,
            _data: ReplayBufferSamples,
            _alpha_val: Float[Array, "batch 1"],
            _task_weights: Float[Array, "batch 1"] | None,
        ):
            dist = self.actor.apply_fn(params, _data.observations)
            action_samples, log_probs = dist.sample_and_log_prob(seed=actor_loss_key)
            log_probs = log_probs.reshape(-1, 1)

            q_values = self.critic.apply_fn(self.critic.params, _data.observations, action_samples)
            min_qf_values = jnp.min(q_values, axis=0)

            rl_term = (_alpha_val * log_probs - min_qf_values)
            rl_loss = (_task_weights * rl_term).mean() if _task_weights is not None else rl_term.mean()

            total_loss = rl_loss
            aux = (log_probs, rl_loss)
            return total_loss, aux

        if self.split_actor_losses:
            (loss_values, aux_values), actor_grads = jax.vmap(
                jax.value_and_grad(actor_loss, has_aux=True),
                in_axes=(None, 0, 0, 0),
                out_axes=0,
            )(self.actor.params, data, alpha_val, task_weights)

            actor_loss_value = loss_values.mean()
            log_probs, rl_losses = aux_values  # each has leading task dim
            flat_grads, _ = flatten_util.ravel_pytree(jax.tree.map(lambda x: x.mean(axis=0), actor_grads))

            rl_loss_scalar = rl_losses.mean()
        else:
            (actor_loss_value, (log_probs, rl_loss_scalar)), actor_grads = jax.value_and_grad(
                actor_loss, has_aux=True
            )(self.actor.params, data, alpha_val, task_weights)
            flat_grads, _ = flatten_util.ravel_pytree(actor_grads)

        key, optimizer_key = jax.random.split(key)
        actor = self.actor.apply_gradients(
            grads=actor_grads,
            optimizer_extra_args={"task_losses": actor_loss_value, "key": optimizer_key},
        )

        flat_params_act, _ = flatten_util.ravel_pytree(actor.params)
        logs = {
            "losses/actor_loss": actor_loss_value,
            "losses/actor_rl": rl_loss_scalar,
            "metrics/actor_grad_magnitude": jnp.linalg.norm(flat_grads),
            "metrics/actor_params_norm": jnp.linalg.norm(flat_params_act),
        }

        return self.replace(actor=actor, key=key), log_probs, logs

    def q_min(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        """
        obs: (B, obs_dim) numpy
        act: (B, act_dim) numpy
        returns: (B,) numpy  (min over ensemble critics)
        """
        obs_j = jnp.asarray(obs)
        act_j = jnp.asarray(act)
        q_ens = self.critic.apply_fn(self.critic.params, obs_j, act_j)  # (N,B,1)
        q_min = jnp.min(q_ens, axis=0).reshape(-1)                      # (B,)
        return jax.device_get(q_min)

    # ============================================================
    # Actor update (IL-only BC extra step)
    # PyTorch(SB3) style: extra actor optimizer step after SAC updates
    # ============================================================
    def update_actor_il(
        self,
        il_data: ReplayBufferSamples | None,
        il_shared_data: ReplayBufferSamples | None,
        success_ema: jax.Array,
    ) -> tuple[Self, LogDict]:
        key, opt_key = jax.random.split(self.key)

        def _bc_loss_for_batch(_il: ReplayBufferSamples, use_qfilter: bool):
            dist_il = self.actor.apply_fn(self.actor.params, _il.observations)
            pred = dist_il.mode()
            per = jnp.mean((pred - _il.actions) ** 2, axis=-1, keepdims=True)  # (B,1)

            il_task_onehot = _il.observations[..., -self.num_tasks:]
            lam = self.compute_il_lambda_from_success_ema(success_ema, il_task_onehot)  # (B,1)

            if not use_qfilter:
                return jnp.mean(lam * per), {
                    "q_thr": jnp.array(jnp.nan, dtype=per.dtype),
                    "good_cnt": jnp.array(_il.observations.shape[0], dtype=jnp.float32),
                    "q_mean_all": jnp.array(jnp.nan, dtype=per.dtype),
                    "q_mean_good": jnp.array(jnp.nan, dtype=per.dtype),
                    "lam_mean": lam.mean(),
                }

            # ---- Q-filter (specific only) ----
            q_ens = self.critic.apply_fn(self.critic.params, _il.observations, _il.actions)  # (N,B,1)
            q_min = jax.lax.stop_gradient(jnp.min(q_ens, axis=0)).reshape(-1)               # (B,)

            thr = jnp.quantile(q_min, 1.0 - self.il_qfilter_top_p)
            good = (q_min >= thr)
            good_f = good.astype(per.dtype).reshape(-1, 1)
            good_cnt = jnp.sum(good.astype(jnp.int32))

            def _use_filtered(_):
                return jnp.sum(lam * good_f * per) / (jnp.sum(good_f) + 1e-8)

            def _skip(_):
                return jnp.array(0.0, dtype=per.dtype)

            loss = jax.lax.cond(good_cnt >= self.il_qfilter_min_good, _use_filtered, _skip, operand=None)

            q_mean_all = q_min.mean()
            q_mean_good = jnp.where(good, q_min, 0.0).sum() / (good_cnt + 1e-8)
            return loss, {
                "q_thr": thr,
                "good_cnt": good_cnt.astype(jnp.float32),
                "q_mean_all": q_mean_all,
                "q_mean_good": q_mean_good,
                "lam_mean": lam.mean(),
            }

        def total_bc_loss_fn(params: FrozenDict):
            # NOTE: params is unused inside _bc_loss_for_batch currently; if you want strictness,
            # pass params and use in apply_fn. Kept simple: rewrite below to use params.
            # We'll do it properly:
            def _bc_loss_for_batch_params(_il: ReplayBufferSamples, use_qfilter: bool):
                dist_il = self.actor.apply_fn(params, _il.observations)
                pred = dist_il.mode()
                per = jnp.mean((pred - _il.actions) ** 2, axis=-1, keepdims=True)

                il_task_onehot = _il.observations[..., -self.num_tasks:]
                lam = self.compute_il_lambda_from_success_ema(success_ema, il_task_onehot)

                if not use_qfilter:
                    return jnp.mean(lam * per), (jnp.nan, jnp.nan, jnp.nan, _il.observations.shape[0], lam.mean())

                q_ens = self.critic.apply_fn(self.critic.params, _il.observations, _il.actions)
                q_min = jax.lax.stop_gradient(jnp.min(q_ens, axis=0)).reshape(-1)

                thr = jnp.quantile(q_min, 1.0 - self.il_qfilter_top_p)
                good = (q_min >= thr)
                good_f = good.astype(per.dtype).reshape(-1, 1)
                good_cnt = jnp.sum(good.astype(jnp.int32))

                def _use(_):
                    return jnp.sum(lam * good_f * per) / (jnp.sum(good_f) + 1e-8)

                def _skip(_):
                    return jnp.array(0.0, dtype=per.dtype)

                loss = jax.lax.cond(good_cnt >= self.il_qfilter_min_good, _use, _skip, operand=None)
                q_mean_all = q_min.mean()
                q_mean_good = jnp.where(good, q_min, 0.0).sum() / (good_cnt + 1e-8)
                return loss, (thr, q_mean_all, q_mean_good, good_cnt.astype(jnp.float32), lam.mean())

            loss_sp = jnp.array(0.0, dtype=jnp.float32)
            loss_sh = jnp.array(0.0, dtype=jnp.float32)

            # logs
            sp_stats = (jnp.nan, jnp.nan, jnp.nan, jnp.array(0.0), jnp.array(0.0))
            sh_stats = (jnp.nan, jnp.nan, jnp.nan, jnp.array(0.0), jnp.array(0.0))

            if il_data is not None:
                loss_sp, sp_stats = _bc_loss_for_batch_params(il_data, use_qfilter=True)

            if il_shared_data is not None:
                loss_sh, sh_stats = _bc_loss_for_batch_params(il_shared_data, use_qfilter=False)

            total = (self.il_coef * loss_sp) + (getattr(self, "il_shared_coef", 1.0) * loss_sh)
            return total, (loss_sp, loss_sh, sp_stats, sh_stats)

        (total_loss, (loss_sp, loss_sh, sp_stats, sh_stats)), grads = jax.value_and_grad(
            total_bc_loss_fn, has_aux=True
        )(self.actor.params)

        actor = self.actor.apply_gradients(
            grads=grads,
            optimizer_extra_args={"task_losses": total_loss, "key": opt_key},
        )

        (sp_thr, sp_q_all, sp_q_good, sp_good_cnt, sp_lam_mean) = sp_stats
        (sh_thr, sh_q_all, sh_q_good, sh_good_cnt, sh_lam_mean) = sh_stats

        logs = {
            "il/active": jnp.array(1.0, dtype=jnp.float32),
            "il/loss_total": total_loss,
            "il/loss_specific": loss_sp,
            "il/loss_shared": loss_sh,
            "il/spec_q_thr": sp_thr,
            "il/spec_q_mean_all": sp_q_all,
            "il/spec_q_mean_good": sp_q_good,
            "il/spec_good_cnt": sp_good_cnt,
            "il/spec_lam_mean": sp_lam_mean,
            "il/shared_cnt": sh_good_cnt,
            "il/shared_lam_mean": sh_lam_mean,
        }
        return self.replace(actor=actor, key=key), logs

    # ============================================================
    # Alpha update
    # ============================================================
    def update_alpha(
        self,
        log_probs: Float[Array, " batch"],
        task_ids: Float[npt.NDArray, " batch num_tasks"],
    ) -> tuple[Self, LogDict]:
        def alpha_loss(params: FrozenDict) -> Float[Array, ""]:
            log_alpha = task_ids @ params["params"]["log_alpha"].reshape(-1, 1)  # pyright: ignore
            return (-log_alpha * (log_probs + self.target_entropy)).mean()

        alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss)(self.alpha.params)
        alpha = self.alpha.apply_gradients(grads=alpha_grads)

        logs = {
            "losses/alpha_loss": alpha_loss_value,
            "alpha": jnp.exp(alpha.params["params"]["log_alpha"]).sum(),  # pyright: ignore
        }
        return self.replace(alpha=alpha), logs

    # ============================================================
    # One update step (critic -> actor(RL) -> alpha -> actor(IL extra))
    # ============================================================
    @jax.jit
    def _update_inner(
        self,
        data: ReplayBufferSamples,
        success_ema: jax.Array,
        il_data: ReplayBufferSamples | None = None,
        il_shared_data: ReplayBufferSamples | None = None,   # shared-overlap
    ) -> tuple[Self, LogDict, jax.Array]:
        task_ids = data.observations[..., -self.num_tasks:]

        alpha_vals = self.alpha.apply_fn(self.alpha.params, task_ids)
        task_weights = extract_task_weights(self.alpha.params, task_ids) if self.use_task_weights else None

        actor_data = critic_data = data
        actor_alpha_vals = critic_alpha_vals = alpha_vals
        actor_task_weights = critic_task_weights = task_weights
        alpha_val_indices = None

        if self.split_critic_losses or self.split_actor_losses:
            split_data, _ = self.split_data_by_tasks(data, task_ids)
            split_alpha_vals, alpha_val_indices = self.split_data_by_tasks(alpha_vals, task_ids)

            split_task_weights, _ = (
                self.split_data_by_tasks(task_weights, task_ids) if task_weights is not None else (None, None)
            )

            if self.split_critic_losses:
                critic_data = split_data
                critic_alpha_vals = split_alpha_vals
                critic_task_weights = split_task_weights

            if self.split_actor_losses:
                actor_data = split_data
                actor_alpha_vals = split_alpha_vals
                actor_task_weights = split_task_weights

        # (1) critic update
        self, critic_logs, td_errs = self.update_critic(critic_data, critic_alpha_vals, critic_task_weights)

        # (2) actor RL-only update
        self, log_probs, actor_logs = self.update_actor(actor_data, actor_alpha_vals, actor_task_weights)

        # (3) alpha update uses RL log_probs
        if self.split_actor_losses:
            assert alpha_val_indices is not None
            log_probs = self.unsplit_data_by_tasks(log_probs, alpha_val_indices)
        self, alpha_logs = self.update_alpha(log_probs, task_ids)

        # (4) actor IL-only extra step
        if self.use_success_based_il and (il_data is not None or il_shared_data is not None):
            self, il_logs = self.update_actor_il(il_data, il_shared_data, success_ema)
        else:
            il_logs = {"il/active": jnp.array(0.0, dtype=jnp.float32)}
        # PCGrad logs (kept)
        critic_optim_logs = {}
        actor_optim_logs = {}
        if isinstance(self.critic.opt_state, tuple) and isinstance(self.actor.opt_state, tuple):
            if isinstance(self.critic.opt_state[0], PCGradState):
                critic_optim_logs = {f"metrics/critic_{k}": v for k, v in self.critic.opt_state[0]._asdict().items()}
            if isinstance(self.actor.opt_state[0], PCGradState):
                actor_optim_logs = {f"metrics/actor_{k}": v for k, v in self.actor.opt_state[0]._asdict().items()}

        logs = {**critic_logs, **actor_logs, **alpha_logs, **il_logs, **critic_optim_logs, **actor_optim_logs}
        return self, logs, td_errs

    def update(
        self,
        data: ReplayBufferSamples,
        success_ema: jax.Array | None = None,
        il_data: ReplayBufferSamples | None = None,
        il_shared_data: ReplayBufferSamples | None = None,
    ) -> tuple[Self, LogDict, jax.Array]:
        if success_ema is None:
            success_ema = jnp.zeros((self.num_tasks,), dtype=jnp.float32)
        return self._update_inner(data, success_ema, il_data, il_shared_data)