# file: ilrl_mtsac.py
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
from metaworld_algorithms.rl.buffers import MultiTaskReplayBuffer
from metaworld_algorithms.rl.buffers import ILRLMultiTaskReplayBuffer  # (위에서 추가된 클래스)
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

import time
from collections import deque
from typing import Deque, Self, override

import orbax.checkpoint as ocp
from metaworld_algorithms.monitoring.utils import log
from metaworld_algorithms.types import GymVectorEnv, CheckpointMetadata, ReplayBufferCheckpoint
from metaworld_algorithms.checkpoint import get_checkpoint_save_args


@jax.jit
def _compute_td_errors_jit(
    actor_apply_fn,
    actor_params,
    critic_apply_fn,
    critic_params,
    critic_target_params,
    alpha_apply_fn,
    alpha_params,
    data: ReplayBufferSamples,
    gamma: float,
    max_q_value: float | None,
    num_tasks: int,
    key: jax.Array,
) -> Float[Array, "batch"]:
    # task ids from observation tail
    task_ids = data.observations[..., -num_tasks:]
    alpha_val = alpha_apply_fn(alpha_params, task_ids)  # (B,1)

    # sample next action
    key, subkey = jax.random.split(key)
    dist = actor_apply_fn(actor_params, data.next_observations)
    next_actions = dist.sample(seed=subkey)
    next_logp = dist.log_prob(next_actions)
    if next_logp.ndim > 1:
        next_logp = next_logp.sum(axis=-1)
    next_logp = next_logp.reshape(-1, 1)

    # target Q
    q_next = critic_apply_fn(critic_target_params, data.next_observations, next_actions)  # (N,B,1)
    min_q_next = jnp.min(q_next, axis=0) - alpha_val * next_logp
    target = data.rewards + (1 - data.dones) * gamma * min_q_next

    # current Q prediction (use min over ensemble as td reference)
    q_pred = critic_apply_fn(critic_params, data.observations, data.actions)  # (N,B,1)
    min_q_pred = jnp.min(q_pred, axis=0)

    if max_q_value is not None:
        target = jnp.clip(target, -max_q_value, max_q_value)
        min_q_pred = jnp.clip(min_q_pred, -max_q_value, max_q_value)

    td = (min_q_pred - target).reshape(-1)
    return td


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
class ILRLMTSACConfig(AlgorithmConfig):
    actor_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    critic_config: QValueFunctionConfig = QValueFunctionConfig()
    temperature_optimizer_config: OptimizerConfig = OptimizerConfig(max_grad_norm=None)
    initial_temperature: float = 1.0
    num_critics: int = 2
    tau: float = 0.005
    use_task_weights: bool = False
    max_q_value: float | None = 5000

    il_coef: float = 1.0
    il_update_per_step: int = 1


class ILRLMTSAC(OffPolicyAlgorithm[ILRLMTSACConfig]):
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

    il_coef: float = struct.field(pytree_node=False)
    il_update_per_step: int = struct.field(pytree_node=False)

    @override
    @staticmethod
    def initialize(
        config: ILRLMTSACConfig, env_config: EnvConfig, seed: int = 1
    ) -> "ILRLMTSAC":
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

        return ILRLMTSAC(
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

            il_coef=config.il_coef,
            il_update_per_step=config.il_update_per_step,
        )


    def compute_td_errors(self, data: ReplayBufferSamples) -> np.ndarray:
        td = _compute_td_errors_jit(
            self.actor.apply_fn,
            self.actor.params,
            self.critic.apply_fn,
            self.critic.params,
            self.critic.target_params,
            self.alpha.apply_fn,
            self.alpha.params,
            data,
            self.gamma,
            self.max_q_value,
            self.num_tasks,
            self.key,   # key는 deterministic하게 쓰기 싫으면 split해서 넣어도 됨
        )
        return np.asarray(jax.device_get(td))

    @override
    def train(
        self,
        config: OffPolicyTrainingConfig,
        envs: GymVectorEnv,
        env_config: EnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)

        obs, _ = envs.reset()
        done = np.full((envs.num_envs,), False)

        start_step, episodes_ended = 0, 0
        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        # --- IMPORTANT: ILRL 전용 버퍼 사용 ---
        replay_buffer = self.spawn_replay_buffer(env_config, config, seed)
        assert isinstance(replay_buffer, ILRLMultiTaskReplayBuffer), (
            "ILRLMTSAC must use ILRLMultiTaskReplayBuffer. "
            "Check spawn_replay_buffer() override."
        )
        if buffer_checkpoint is not None:
            replay_buffer.load_checkpoint(buffer_checkpoint)

        start_time = time.time()

        # ---- per-env episode success tracking (to update EMA on episode end) ----
        # MT10 벡터 env는 보통 env index == task index 로 가정 (num_envs==num_tasks)
        # (만약 wrapper가 task 배치를 바꾼다면 여기 매핑만 조정)
        per_env_episode_success = np.zeros((envs.num_envs,), dtype=np.float32)

        for global_step in range(start_step, config.total_steps // envs.num_envs):
            total_steps = global_step * envs.num_envs

            if global_step < config.warmstart_steps:
                actions = envs.action_space.sample()
            else:
                self, actions = self.sample_action(obs)

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            done = np.logical_or(terminations, truncations)

            buffer_obs = next_obs
            if "final_obs" in infos:
                buffer_obs = np.where(done[:, None], np.stack(infos["final_obs"]), next_obs)

            # ---- (A) main RL buffer add (same as baseline) ----
            replay_buffer.add(obs, buffer_obs, actions, rewards, done)

            # ---- (B) success tracking + IL buffer add ----
            # metaworld wrappers에서 보통 info에 success가 있음. key가 다르면 여기만 바꾸면 됨.
            # case1) infos["success"] shape (num_envs,)
            # case2) infos["final_info"]["success"] 같은 구조일 수도 있음
            success_vec = None
            if "success" in infos:
                success_vec = np.asarray(infos["success"], dtype=np.float32)
            elif "final_info" in infos and isinstance(infos["final_info"], dict) and "success" in infos["final_info"]:
                success_vec = np.asarray(infos["final_info"]["success"], dtype=np.float32)

            if success_vec is not None:
                # episode 동안 success 한번이라도 찍히면 성공으로 취급
                per_env_episode_success = np.maximum(per_env_episode_success, success_vec)

                # (선택) "성공 transition"을 IL 버퍼에 넣고 싶으면, 성공인 step에서만 저장
                # 여기서는 success_vec[i] == 1인 step을 success transition으로 저장
                for i in range(envs.num_envs):
                    if success_vec[i] > 0.5:
                        replay_buffer.add_il_transition(
                            task_idx=i,
                            obs=obs[i],
                            next_obs=buffer_obs[i],
                            action=actions[i],
                            reward=float(rewards[i]),
                            done=float(done[i]),
                        )

            obs = next_obs

            # ---- (C) episode end 처리: EMA 업데이트 + logging deque ----
            if done.any():
                # EMA는 episode가 끝난 env들만 반영해서 업데이트하는 게 안정적
                task_success = np.zeros((self.num_tasks,), dtype=np.float32)
                for i, env_ended in enumerate(done):
                    if env_ended:
                        task_success[i] = per_env_episode_success[i]
                        per_env_episode_success[i] = 0.0  # reset for next episode

                        global_episodic_return.append(infos["final_info"]["episode"]["r"][i])
                        global_episodic_length.append(infos["final_info"]["episode"]["l"][i])
                        episodes_ended += 1

                replay_buffer.update_success_ema(task_success)

            # ---- baseline logging 유지 ----
            if global_step % 500 == 0 and global_episodic_return:
                print(
                    f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
                )
                if track:
                    log(
                        {
                            "charts/mean_episodic_return": float(np.mean(list(global_episodic_return))),
                            "charts/mean_episodic_length": float(np.mean(list(global_episodic_length))),
                        },
                        step=total_steps,
                    )

            # ---- (D) update: RL batch + optional IL batch ----
            if global_step > config.warmstart_steps:
                # IL batch size는 일단 RL batch와 동일하게 두자 (원하면 config로 뺄 수 있음)
                il_batch_size = config.batch_size

                batch: ILRLReplayBatch = replay_buffer.sample(
                    batch_size=config.batch_size,
                    il_batch_size=il_batch_size,
                )

                # 1) SAC update (RL batch)
                self, logs = self.update(batch.rl)

                # 2) BC update (IL batch가 준비됐을 때만)
                if batch.il is not None and getattr(self, "il_coef", 0.0) > 0.0:
                    # il_updates_per_step 만큼 반복
                    for _ in range(getattr(self, "il_updates_per_step", 1)):
                        self, il_logs = self.update_actor_il(batch.il)  # ILRLMTSAC에 구현해둔 함수
                        logs = logs | il_logs

                # 3) TD priority update (optional, but you want it)
                #    -> 아래 2)에서 추가하는 compute_td_errors를 사용
                # if hasattr(self, "compute_td_errors"):
                #     td_err = np.asarray(self.compute_td_errors(batch.rl))  # (B,)
                #     replay_buffer.update_td_priorities(batch.rl_indices, batch.rl_task_indices, td_err)

                #     # (선택) 로그
                #     logs = logs | {
                #         "data/td_abs_mean": float(np.mean(np.abs(td_err))),
                #     }

                # ---- SPS/logging ----
                if global_step % 100 == 0:
                    sps_steps = (global_step - start_step) * envs.num_envs
                    sps = int(sps_steps / (time.time() - start_time))
                    print("SPS:", sps)
                    if track:
                        log({"charts/SPS": sps} | logs, step=total_steps)

                # ---- Evaluation (baseline과 동일) ----
                if (
                    config.evaluation_frequency > 0
                    and episodes_ended % config.evaluation_frequency == 0
                    and done.any()
                    and global_step > 0
                ):
                    mean_success_rate, mean_returns, mean_success_per_task = env_config.evaluate(envs, self)
                    eval_metrics = {
                        "charts/mean_success_rate": float(mean_success_rate),
                        "charts/mean_evaluation_return": float(mean_returns),
                    } | {
                        f"charts/{task_name}_success_rate": float(success_rate)
                        for task_name, success_rate in mean_success_per_task.items()
                    }
                    print(
                        f"total_steps={total_steps}, mean evaluation success rate: {mean_success_rate:.4f}"
                        + f" return: {mean_returns:.4f}"
                    )

                    if track:
                        log(eval_metrics, step=total_steps)

                    if checkpoint_manager is not None:
                        if not done.all():
                            raise NotImplementedError(
                                "Checkpointing doesn't work when evaluation happens before all envs finish."
                            )
                        checkpoint_manager.save(
                            total_steps,
                            args=get_checkpoint_save_args(
                                self,
                                envs,
                                global_step,
                                episodes_ended,
                                run_timestamp,
                                buffer=replay_buffer,
                            ),
                            metrics={k.removeprefix("charts/"): v for k, v in eval_metrics.items()},
                        )

                    obs, _ = envs.reset()

        return self

    @override
    def spawn_replay_buffer(self, env_config: EnvConfig, config: OffPolicyTrainingConfig, seed: int = 1):
        return ILRLMultiTaskReplayBuffer(
            total_capacity=config.buffer_size,
            num_tasks=self.num_tasks,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,

            task_beta=1.0,
            prio_alpha=0.6,
            success_ema_tau=0.01,
            easy_success_th=0.8,
            il_total_capacity=None,
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
            return loss, q_pred.mean()

        if self.split_critic_losses:
            (critic_loss_value, qf_values), critic_grads = jax.vmap(
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
            flat_grads, _ = flatten_util.ravel_pytree(
                jax.tree.map(lambda x: x.mean(axis=0), critic_grads)
            )
        else:
            (critic_loss_value, qf_values), critic_grads = jax.value_and_grad(
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
        }

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
            _task_weights: Float[Array, "batch 1"] | None = None,
        ):
            action_samples, log_probs = self.actor.apply_fn(
                params, _data.observations
            ).sample_and_log_prob(seed=actor_loss_key)
            log_probs = log_probs.reshape(-1, 1)

            q_values = self.critic.apply_fn(
                self.critic.params, _data.observations, action_samples
            )
            min_qf_values = jnp.min(q_values, axis=0)
            if _task_weights is not None:
                loss = (task_weights * (_alpha_val * log_probs - min_qf_values)).mean()
            else:
                loss = (_alpha_val * log_probs - min_qf_values).mean()
            return loss, log_probs

        if self.split_actor_losses:
            (actor_loss_value, log_probs), actor_grads = jax.vmap(
                jax.value_and_grad(actor_loss, has_aux=True),
                in_axes=(None, 0, 0, 0),
                out_axes=0,
            )(self.actor.params, data, alpha_val, task_weights)
            flat_grads, _ = flatten_util.ravel_pytree(
                jax.tree.map(lambda x: x.mean(axis=0), actor_grads)
            )
        else:
            (actor_loss_value, log_probs), actor_grads = jax.value_and_grad(
                actor_loss, has_aux=True
            )(self.actor.params, data, alpha_val, task_weights)
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
            "losses/actor_loss": actor_loss_value.mean(),
            "metrics/actor_grad_magnitude": jnp.linalg.norm(flat_grads),
            "metrics/actor_params_norm": jnp.linalg.norm(flat_params_act),
        }

        return (self.replace(actor=actor, key=key), log_probs, logs)

    def update_actor_il(self, il_data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        if self.il_coef <= 0.0:
            return self, {"losses/il_bc_loss": 0.0}

        key, il_key = jax.random.split(self.key)

        def bc_loss(params: FrozenDict, _data: ReplayBufferSamples) -> Float[Array, ""]:
            dist = self.actor.apply_fn(params, _data.observations)
            logp = dist.log_prob(_data.actions)
            if logp.ndim > 1:
                logp = logp.sum(axis=-1)
            return (-logp).mean()

        loss_val, grads = jax.value_and_grad(bc_loss)(self.actor.params, il_data)

        # scaling by il_coef
        grads = jax.tree.map(lambda g: self.il_coef * g, grads)

        actor = self.actor.apply_gradients(
            grads=grads,
            optimizer_extra_args={
                "task_losses": loss_val,  # PCGrad 쓰면 나중에 필요시 shape 맞추기
                "key": il_key,
            },
        )

        flat_grads, _ = flatten_util.ravel_pytree(grads)
        flat_params, _ = flatten_util.ravel_pytree(actor.params)

        return self.replace(actor=actor, key=key), {
            "losses/il_bc_loss": loss_val,
            "metrics/il_actor_grad_magnitude": jnp.linalg.norm(flat_grads),
            "metrics/il_actor_params_norm": jnp.linalg.norm(flat_params),
        }

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
    def _update_inner(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
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

        self, critic_logs = self.update_critic(
            critic_data, critic_alpha_vals, critic_task_weights
        )
        self, log_probs, actor_logs = self.update_actor(
            actor_data, actor_alpha_vals, actor_task_weights
        )
        if self.split_actor_losses:
            assert alpha_val_indices is not None
            log_probs = self.unsplit_data_by_tasks(log_probs, alpha_val_indices)
        self, alpha_logs = self.update_alpha(log_probs, task_ids)

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

        il_logs: LogDict = {}
        il_batch = getattr(data, "il_batch", None)  # Step2에서 붙일 예정

        if (il_batch is not None) and (self.il_coef > 0.0):
            def do_il_update(carry, _):
                alg = carry
                alg, logs = alg.update_actor_il(il_batch)
                return alg, logs

            self, il_log_stack = jax.lax.scan(
                do_il_update,
                self,
                xs=None,
                length=self.il_updates_per_step,
            )
            il_logs = jax.tree.map(lambda x: x.mean(), il_log_stack)

        return self, {
            **critic_logs, **actor_logs, **alpha_logs,
            **critic_optim_logs, **actor_optim_logs,
            **il_logs,
        }
    @override
    def update(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        return self._update_inner(data)
