# file: ilrl_mtsac.py
"""Inspired by https://github.com/kevinzakka/robopianist-rl/blob/main/sac.py"""

import dataclasses
from functools import partial
from typing import Self, override, Optional, List

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
    
    # IL-RL specific
    lambda_max: float = 2.0
    s0: float = 0.0
    s1: float = 1.0
    top_p: float = 0.1
    
    # Buffer params
    beta_task: float = 1.0
    success_ema_tau: float = 0.01


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
    
    lambda_max: float = struct.field(pytree_node=False)
    s0: float = struct.field(pytree_node=False)
    s1: float = struct.field(pytree_node=False)
    top_p: float = struct.field(pytree_node=False)
    
    beta_task: float = struct.field(pytree_node=False)
    success_ema_tau: float = struct.field(pytree_node=False)

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
            lambda_max=config.lambda_max,
            s0=config.s0,
            s1=config.s1,
            top_p=config.top_p,
            beta_task=config.beta_task,
            success_ema_tau=config.success_ema_tau,
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
        
        # Trajectory cache for retroactive IL buffering
        # List of list of (obs, next_obs, action, reward, done)
        env_trajectories: List[List] = [[] for _ in range(envs.num_envs)]

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

            # Store transition in trajectory cache
            for i in range(envs.num_envs):
                # We need copies to avoid overwriting if buffers are reused by wrappers
                t = (
                    obs[i].copy(), 
                    buffer_obs[i].copy(), 
                    actions[i].copy(), 
                    float(rewards[i]), 
                    float(done[i])
                )
                env_trajectories[i].append(t)

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
                        
                        # Retroactive IL buffering
                        # If this episode was successful, add ALL transitions to IL buffer
                        if task_success[i] > 0.5:
                            for tr in env_trajectories[i]:
                                tobs, tnext_obs, tact, trew, tdone = tr
                                replay_buffer.add_il_transition(
                                    task_idx=i,
                                    obs=tobs,
                                    next_obs=tnext_obs,
                                    action=tact,
                                    reward=trew,
                                    done=tdone,
                                )
                        
                        # Clear trajectory
                        env_trajectories[i] = []

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
                task_success_ema = jnp.array(replay_buffer.success_ema)
                self, logs, td_err = self.update(batch.rl, batch.il, task_success_ema)
                
                # 1-1) priority는 "샘플된 batch만" 업데이트 (PER 정석)
                td_err_np = np.asarray(jax.device_get(td_err))  # (B,)
                replay_buffer.update_td_priorities(batch.rl_indices, batch.rl_task_indices, td_err_np)
                


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



            task_beta=self.beta_task,
            prio_alpha=0.6,
            success_ema_tau=self.success_ema_tau,
            easy_success_th=0.0,  # Catch all successes
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

    # --- (1) update_critic: td_err 반환 추가 ---
    def update_critic(
        self,
        data: ReplayBufferSamples,
        alpha_val: Float[Array, "*batch 1"],
        task_weights: Float[Array, "*batch 1"] | None = None,
    ) -> tuple[Self, LogDict, Float[Array, "*batch"]]:
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
        ) -> tuple[Float[Array, ""], tuple[Float[Array, ""], Float[Array, "#batch"]]]:
            # next_action_log_probs is (B,) shaped, Q values are (B,1)
            min_qf_next_target = jnp.min(_q_values, axis=0) - _alpha_val * _next_action_log_probs.reshape(-1, 1)

            next_q_value = jax.lax.stop_gradient(
                _data.rewards + (1 - _data.dones) * self.gamma * min_qf_next_target
            )  # (B,1)

            q_pred = self.critic.apply_fn(params, _data.observations, _data.actions)  # (N,B,1)

            if self.max_q_value is not None:
                next_q_value = jnp.clip(next_q_value, -self.max_q_value, self.max_q_value)
                q_pred = jnp.clip(q_pred, -self.max_q_value, self.max_q_value)

            # PER용 TD-error는 "min over ensemble" 기준이 가장 자연스럽다
            min_q_pred = jnp.min(q_pred, axis=0)  # (B,1)
            td_err = jax.lax.stop_gradient((min_q_pred - next_q_value).reshape(-1))  # (B,)

            if _task_weights is not None:
                loss = (_task_weights * (q_pred - next_q_value) ** 2).mean()
            else:
                loss = ((q_pred - next_q_value) ** 2).mean()

            # aux: (q_mean, td_err_vector)
            return loss, (q_pred.mean(), td_err)

        if self.split_critic_losses:
            (critic_loss_value, (qf_values, td_err)), critic_grads = jax.vmap(
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
            (critic_loss_value, (qf_values, td_err)), critic_grads = jax.value_and_grad(
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

        logs = {
            "losses/qf_values": qf_values.mean(),
            "losses/qf_loss": critic_loss_value.mean(),
            "metrics/critic_grad_magnitude": jnp.linalg.norm(flat_grads),
            "metrics/critic_params_norm": jnp.linalg.norm(flat_params_crit),
        }
        return self.replace(critic=critic, key=key), logs, td_err

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
                loss = (_task_weights * (_alpha_val * log_probs - min_qf_values)).mean()
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

    @jax.jit
    def update_actor_il(
        self,
        il_data: ReplayBufferSamples,
        task_success_ema: Float[Array, " num_tasks"],
    ) -> tuple[Self, LogDict]:
        if self.lambda_max <= 0.0:
            return self, {
                "losses/il_bc_loss": 0.0,
                "il/active": 0.0,
                "il/good_ratio": 0.0,
                "il/lambda_mean": 0.0,
            }

        key, il_key = jax.random.split(self.key)

        # 1. Q-filtering
        # We need to evaluate Q(s, a) for the IL batch
        # Use min Q over ensemble
        q_values = self.critic.apply_fn(
            self.critic.params, il_data.observations, il_data.actions
        )  # (N, B, 1)
        q = jnp.min(q_values, axis=0).squeeze(-1)  # (B,)
        
        # Threshold
        thr = jnp.quantile(q, 1.0 - self.top_p)
        good = q >= thr  # (B,) boolean
        mask = good.astype(jnp.float32)
        
        # Stats
        good_cnt = mask.sum()
        good_ratio = mask.mean()
        q_mean_all = q.mean()
        q_mean_good = jnp.sum(q * mask) / (good_cnt + 1e-8)

        # 2. Adaptive Lambda
        # Extract task index (assuming last num_tasks dims are 1-hot)
        task_onehot = il_data.observations[..., -self.num_tasks:]  # (B, N)
        s = task_onehot @ task_success_ema  # (B,)
        
        w = (s - self.s0) / (self.s1 - self.s0)
        w = jnp.clip(w, 0.0, 1.0)
        lam = self.lambda_max * w  # (B,)
        lambda_mean = lam.mean()

        # 3. BC Update
        def bc_loss(params: FrozenDict, _data: ReplayBufferSamples) -> Float[Array, ""]:
            # Use deterministic mode for BC target as per reference implementation
            dist = self.actor.apply_fn(params, _data.observations)
            pred_action = dist.mode()
            
            # MSE loss: mean over action dim
            per_sample_loss = jnp.mean((pred_action - _data.actions) ** 2, axis=-1)  # (B,)
            
            # Weighted masked loss
            # Sum over batch, but normalize by sum of mask to avoid scaling issues?
            # Reference implementation: (lam * mask * per).sum() / (mask.sum() + 1e-8)
            weighted_loss = (lam * mask * per_sample_loss).sum() / (mask.sum() + 1e-8)
            return weighted_loss

        loss_val, grads = jax.value_and_grad(bc_loss)(self.actor.params, il_data)

        actor = self.actor.apply_gradients(
            grads=grads,
            optimizer_extra_args={
                "task_losses": loss_val,
                "key": il_key,
            },
        )

        flat_grads, _ = flatten_util.ravel_pytree(grads)
        flat_params, _ = flatten_util.ravel_pytree(actor.params)
        
        logs = {
            "losses/il_bc_loss": loss_val,
            "metrics/il_actor_grad_magnitude": jnp.linalg.norm(flat_grads),
            "metrics/il_actor_params_norm": jnp.linalg.norm(flat_params),
            "il/active": 1.0,
            "il/good_ratio": good_ratio,
            "il/q_thr": thr,
            "il/q_mean_all": q_mean_all,
            "il/q_mean_good": q_mean_good,
            "il/lambda_mean": lambda_mean,
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

    # --- (2) _update_inner: td_err를 propagate + split일 때 unsplit까지 처리 ---
    @jax.jit
    def _update_inner_no_il(
        self,
        data: ReplayBufferSamples,
    ) -> tuple[Self, LogDict, Float[Array, "batch"]]:
        # --- 여기 내용은 너 원래 _update_inner에서 il 관련 부분만 제거한 것 ---
        task_ids = data.observations[..., -self.num_tasks :]

        alpha_vals = self.alpha.apply_fn(self.alpha.params, task_ids)
        task_weights = extract_task_weights(self.alpha.params, task_ids) if self.use_task_weights else None

        actor_data = critic_data = data
        actor_alpha_vals = critic_alpha_vals = alpha_vals
        actor_task_weights = critic_task_weights = task_weights
        alpha_val_indices = None
        data_sort_indices = None

        if self.split_critic_losses or self.split_actor_losses:
            split_data, data_sort_indices = self.split_data_by_tasks(data, task_ids)
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

        self, critic_logs, td_err = self.update_critic(critic_data, critic_alpha_vals, critic_task_weights)

        self, log_probs, actor_logs = self.update_actor(actor_data, actor_alpha_vals, actor_task_weights)
        if self.split_actor_losses:
            assert alpha_val_indices is not None
            log_probs = self.unsplit_data_by_tasks(log_probs, alpha_val_indices)

        self, alpha_logs = self.update_alpha(log_probs, task_ids)

        if self.split_critic_losses:
            assert data_sort_indices is not None
            td_err = self.unsplit_data_by_tasks(td_err[..., None], data_sort_indices).reshape(-1)

        logs = {
            **critic_logs, **actor_logs, **alpha_logs,
            "data/td_abs_mean": jnp.mean(jnp.abs(td_err)),
        }
        return self, logs, td_err

    @jax.jit
    def _update_inner_with_il(
        self,
        rl_data: ReplayBufferSamples,
        il_data: ReplayBufferSamples,   # ✅ None 절대 금지
        task_success_ema: Float[Array, " num_tasks"],
    ) -> tuple[Self, LogDict, Float[Array, "batch"]]:
        self, logs, td_err = self._update_inner_no_il(rl_data)

        def do_il_update(alg, _):
            alg, il_logs = alg.update_actor_il(il_data, task_success_ema)
            return alg, il_logs

        self, il_log_stack = jax.lax.scan(
            do_il_update,
            self,
            xs=None,
            length=self.il_update_per_step,
        )
        il_logs = jax.tree.map(lambda x: x.mean(), il_log_stack)

        logs = {**logs, **il_logs}
        return self, logs, td_err    


    # --- (3) update: td_err도 반환 ---
    @override
    def update(
        self,
        rl_data: ReplayBufferSamples,
        il_data: ReplayBufferSamples | None = None,
        task_success_ema: Float[Array, " num_tasks"] | None = None,
    ) -> tuple[Self, LogDict, Float[Array, "batch"]]:
        if (il_data is None) or (self.lambda_max <= 0.0) or (self.il_update_per_step <= 0) or (task_success_ema is None):
            return self._update_inner_no_il(rl_data)
        else:
            return self._update_inner_with_il(rl_data, il_data, task_success_ema)