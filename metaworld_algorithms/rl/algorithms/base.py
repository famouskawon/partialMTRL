import abc
import time
from collections import deque
from typing import Deque, Generic, Self, TypeVar, override
import jax
import numpy as np
import numpy.typing as npt
import orbax.checkpoint as ocp
from flax import struct
from jaxtyping import Float

from metaworld_algorithms.checkpoint import get_checkpoint_save_args
from metaworld_algorithms.config.envs import EnvConfig, MetaLearningEnvConfig
from metaworld_algorithms.config.rl import (
    AlgorithmConfig,
    GradientBasedMetaLearningTrainingConfig,
    MetaLearningTrainingConfig,
    OffPolicyTrainingConfig,
    OnPolicyTrainingConfig,
    RNNBasedMetaLearningTrainingConfig,
    TrainingConfig,
)
from metaworld_algorithms.monitoring.utils import log
from metaworld_algorithms.rl.buffers import (
    AbstractReplayBuffer,
    MultiTaskRolloutBuffer,
)
from metaworld_algorithms.types import (
    Action,
    AuxPolicyOutputs,
    CheckpointMetadata,
    GymVectorEnv,
    LogDict,
    MetaLearningAgent,
    Observation,
    ReplayBufferCheckpoint,
    ReplayBufferSamples,
    RNNState,
    Rollout,
    PrioritizedReplayBatch
)

AlgorithmConfigType = TypeVar("AlgorithmConfigType", bound=AlgorithmConfig)
TrainingConfigType = TypeVar("TrainingConfigType", bound=TrainingConfig)
EnvConfigType = TypeVar("EnvConfigType", bound=EnvConfig)
MetaLearningTrainingConfigType = TypeVar(
    "MetaLearningTrainingConfigType", bound=MetaLearningTrainingConfig
)
DataType = TypeVar("DataType", ReplayBufferSamples, Rollout, list[Rollout])


import os, json
import numpy as np
from datetime import datetime

def save_buffer_npz(path: str, replay_buffer, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = replay_buffer.checkpoint()
    payload = {}
    for k, v in ckpt["data"].items():
        payload[f"data/{k}"] = v
    payload["rng_state_json"] = np.array(json.dumps(ckpt["rng_state"]), dtype=object)
    meta = dict(meta)
    meta["saved_at"] = datetime.now().isoformat()
    payload["meta_json"] = np.array(json.dumps(meta), dtype=object)
    np.savez_compressed(path, **payload)
SNAP_EVERY = 200_000  # total_steps 기준

def probe_sample_per_task(replay_buffer, rng: np.random.Generator, k_per_task: int):
    # valid size
    size = replay_buffer.capacity if replay_buffer.full else replay_buffer.pos
    T = replay_buffer.num_tasks

    buf_idx = []
    task_idx = []
    obs = []
    act = []

    for t in range(T):
        idx = rng.integers(0, size, size=(k_per_task,), dtype=np.int32)
        buf_idx.append(idx)
        task_idx.append(np.full((k_per_task,), t, dtype=np.int32))
        obs.append(replay_buffer.obs[idx, t])
        act.append(replay_buffer.actions[idx, t])

    buf_idx = np.concatenate(buf_idx, axis=0)
    task_idx = np.concatenate(task_idx, axis=0)
    obs = np.concatenate(obs, axis=0)
    act = np.concatenate(act, axis=0)
    return buf_idx, task_idx, obs, act




class Algorithm(
    abc.ABC,
    Generic[AlgorithmConfigType, TrainingConfigType, EnvConfigType, DataType],
    struct.PyTreeNode,
):
    """Based on https://github.com/kevinzakka/nanorl/blob/main/nanorl/agent.py"""

    num_tasks: int = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False)

    @staticmethod
    @abc.abstractmethod
    def initialize(
        config: AlgorithmConfigType, env_config: EnvConfigType, seed: int = 1
    ) -> "Algorithm": ...

    @abc.abstractmethod
    def get_num_params(self) -> dict[str, int]: ...

    @abc.abstractmethod
    def train(
        self,
        config: TrainingConfigType,
        envs: GymVectorEnv,
        env_config: EnvConfigType,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self: ...


class MetaLearningAlgorithm(
    Algorithm[
        AlgorithmConfigType,
        MetaLearningTrainingConfigType,
        MetaLearningEnvConfig,
        DataType,
    ],
    Generic[AlgorithmConfigType, MetaLearningTrainingConfigType, DataType],
):
    @staticmethod
    @abc.abstractmethod
    def initialize(
        config: AlgorithmConfigType, env_config: MetaLearningEnvConfig, seed: int = 1
    ) -> "MetaLearningAlgorithm": ...

    @abc.abstractmethod
    def update(self, data: DataType) -> tuple[Self, LogDict]: ...

    @abc.abstractmethod
    def wrap(self) -> MetaLearningAgent: ...

    @abc.abstractmethod
    def train(
        self,
        config: MetaLearningTrainingConfigType,
        envs: GymVectorEnv,
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self: ...


class GradientBasedMetaLearningAlgorithm(
    MetaLearningAlgorithm[
        AlgorithmConfigType, GradientBasedMetaLearningTrainingConfig, list[Rollout]
    ],
    Generic[AlgorithmConfigType],
):
    @abc.abstractmethod
    def sample_action_and_aux(
        self, observation: Observation
    ) -> tuple[Self, Action, AuxPolicyOutputs]: ...

    def spawn_rollout_buffer(
        self,
        env_config: EnvConfig,
        training_config: GradientBasedMetaLearningTrainingConfig,
        seed: int | None = None,
    ) -> MultiTaskRolloutBuffer:
        return MultiTaskRolloutBuffer(
            num_tasks=training_config.meta_batch_size,
            num_rollout_steps=training_config.rollouts_per_task
            * env_config.max_episode_steps,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
        )

    @abc.abstractmethod
    def adapt(self, rollouts: Rollout) -> Self: ...

    @abc.abstractmethod
    def init_ensemble_networks(self) -> Self: ...

    @override
    def train(
        self,
        config: GradientBasedMetaLearningTrainingConfig,
        envs: GymVectorEnv,
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        rollout_buffer = self.spawn_rollout_buffer(env_config, config, seed)

        # NOTE: We assume that eval evns are deterministically initialised and there's no state
        # that needs to be carried over when they're used.
        eval_envs = env_config.spawn_test(seed)

        start_time = time.time()

        steps_per_iter = (
            config.meta_batch_size
            * config.rollouts_per_task
            * env_config.max_episode_steps
            * (config.num_inner_gradient_steps + 1)
        )

        for _iter in range(
            start_step, config.total_steps // steps_per_iter
        ):  # Outer step
            global_step = _iter * steps_per_iter
            print(f"Iteration {_iter}, Global num of steps {global_step}")

            envs.call("sample_tasks")
            self = self.init_ensemble_networks()
            all_rollouts: list[Rollout] = []

            # Sampling step
            # Collect num_inner_gradient_steps D datasets + collect 1 D' dataset
            for _step in range(config.num_inner_gradient_steps + 1):
                print(f"- Collecting inner step {_step}")
                obs, _ = envs.reset()
                rollout_buffer.reset()
                episode_started = np.ones((envs.num_envs,))

                while not rollout_buffer.ready:
                    self, actions, aux_policy_outs = self.sample_action_and_aux(obs)

                    next_obs, rewards, terminations, truncations, infos = envs.step(
                        actions
                    )

                    rollout_buffer.add(
                        obs,
                        actions,
                        rewards,
                        episode_started,
                        value=aux_policy_outs.get("value"),
                        log_prob=aux_policy_outs.get("log_prob"),
                        mean=aux_policy_outs.get("mean"),
                        std=aux_policy_outs.get("std"),
                    )

                    episode_started = np.logical_or(terminations, truncations)
                    obs = next_obs

                    for i, env_ended in enumerate(episode_started):
                        if env_ended:
                            global_episodic_return.append(
                                infos["final_info"]["episode"]["r"][i]
                            )
                            global_episodic_length.append(
                                infos["final_info"]["episode"]["l"][i]
                            )

                rollouts = rollout_buffer.get()
                all_rollouts.append(rollouts)

                # Inner policy update for the sake of sampling close to adapted policy during the
                # computation of the objective.
                if _step < config.num_inner_gradient_steps:
                    print(f"- Adaptation step {_step}")
                    self = self.adapt(rollouts)

            mean_episodic_return = np.mean(list(global_episodic_return))
            print("- Mean episodic return: ", mean_episodic_return)
            if track:
                log(
                    {"charts/mean_episodic_returns": mean_episodic_return},
                    step=global_step,
                )

            # Outer policy update
            print("- Computing outer step")
            self, logs = self.update(all_rollouts)

            # Evaluation
            if global_step % config.evaluation_frequency == 0 and global_step > 0:
                print("- Evaluating on the test set...")
                mean_success_rate, mean_returns, mean_success_per_task = (
                    env_config.evaluate_metalearning(eval_envs, self.wrap())
                )

                eval_metrics = {
                    "charts/mean_success_rate": float(mean_success_rate),
                    "charts/mean_evaluation_return": float(mean_returns),
                } | {
                    f"charts/{task_name}_success_rate": float(success_rate)
                    for task_name, success_rate in mean_success_per_task.items()
                }

                if config.evaluate_on_train:
                    print("- Evaluating on the train set...")
                    _, _, eval_success_rate_per_train_task = (
                        env_config.evaluate_metalearning_on_train(
                            envs=envs,
                            agent=self.wrap(),
                        )
                    )
                    for (
                        task_name,
                        success_rate,
                    ) in eval_success_rate_per_train_task.items():
                        eval_metrics[f"charts/{task_name}_train_success_rate"] = float(
                            success_rate
                        )

                print(
                    f"Mean evaluation success rate: {mean_success_rate:.4f}"
                    + f" return: {mean_returns:.4f}"
                )

                if track:
                    log(eval_metrics, step=global_step)

                if checkpoint_manager is not None:
                    checkpoint_manager.save(
                        global_step,
                        args=get_checkpoint_save_args(
                            self,
                            envs,
                            global_step,
                            episodes_ended,
                            run_timestamp,
                        ),
                        metrics={
                            k.removeprefix("charts/"): v
                            for k, v in eval_metrics.items()
                        },
                    )
                    print("- Saved Model")

            # Logging
            print(logs)
            sps = global_step / (time.time() - start_time)
            print("- SPS: ", sps)
            if track:
                log({"charts/SPS": sps} | logs, step=global_step)

        eval_envs.close()
        del eval_envs

        return self


class RNNBasedMetaLearningAlgorithm(
    MetaLearningAlgorithm[
        AlgorithmConfigType, RNNBasedMetaLearningTrainingConfig, Rollout
    ],
    Generic[AlgorithmConfigType],
):
    @abc.abstractmethod
    def sample_action_and_aux(
        self, state: RNNState, observation: Observation
    ) -> tuple[Self, RNNState, Action, AuxPolicyOutputs]: ...

    def spawn_rollout_buffer(
        self,
        env_config: EnvConfig,
        training_config: RNNBasedMetaLearningTrainingConfig,
        example_state: RNNState,
        seed: int | None = None,
    ) -> MultiTaskRolloutBuffer:
        return MultiTaskRolloutBuffer(
            num_tasks=training_config.meta_batch_size,
            num_rollout_steps=training_config.rollouts_per_task
            * env_config.max_episode_steps,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            rnn_state_dim=example_state.shape[-1],
            seed=seed,
        )

    @abc.abstractmethod
    def init_recurrent_state(self, batch_size: int) -> tuple[Self, RNNState]: ...

    @abc.abstractmethod
    def reset_recurrent_state(
        self, current_state: RNNState, reset_mask: npt.NDArray[np.bool_]
    ) -> tuple[Self, RNNState]: ...

    @override
    def train(
        self,
        config: RNNBasedMetaLearningTrainingConfig,
        envs: GymVectorEnv,
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        _, example_state = self.init_recurrent_state(config.meta_batch_size)
        rollout_buffer = self.spawn_rollout_buffer(
            env_config, config, example_state, seed
        )

        # NOTE: We assume that eval evns are deterministically initialised and there's no state
        # that needs to be carried over when they're used.
        eval_envs = env_config.spawn_test(seed)

        start_time = time.time()

        steps_per_iter = (
            config.meta_batch_size
            * config.rollouts_per_task
            * env_config.max_episode_steps
        )

        for _iter in range(
            start_step, config.total_steps // steps_per_iter
        ):  # Outer step
            global_step = _iter * steps_per_iter
            print(f"Iteration {_iter}, Global num of steps {global_step}")

            envs.call("sample_tasks")
            self, states = self.init_recurrent_state(config.meta_batch_size)
            obs, _ = envs.reset()
            rollout_buffer.reset()
            episode_started = np.ones((envs.num_envs,))

            while not rollout_buffer.ready:
                self, next_states, actions, aux_policy_outs = (
                    self.sample_action_and_aux(states, obs)
                )

                next_obs, rewards, terminations, truncations, infos = envs.step(actions)

                rollout_buffer.add(
                    obs,
                    actions,
                    rewards,
                    episode_started,
                    value=aux_policy_outs.get("value"),
                    log_prob=aux_policy_outs.get("log_prob"),
                    mean=aux_policy_outs.get("mean"),
                    std=aux_policy_outs.get("std"),
                    rnn_state=states,
                )

                episode_started = np.logical_or(terminations, truncations)
                obs = next_obs
                states = next_states

                for i, env_ended in enumerate(episode_started):
                    if env_ended:
                        global_episodic_return.append(
                            infos["final_info"]["episode"]["r"][i]
                        )
                        global_episodic_length.append(
                            infos["final_info"]["episode"]["l"][i]
                        )

            rollouts = rollout_buffer.get()

            mean_episodic_return = np.mean(list(global_episodic_return))
            print("- Mean episodic return: ", mean_episodic_return)
            if track:
                log(
                    {"charts/mean_episodic_returns": mean_episodic_return},
                    step=global_step,
                )

            # Outer policy update
            print("- Computing update")
            self, logs = self.update(rollouts)

            # Evaluation
            if global_step % config.evaluation_frequency == 0 and global_step > 0:
                print("- Evaluating on the test set...")
                mean_success_rate, mean_returns, mean_success_per_task = (
                    env_config.evaluate_metalearning(eval_envs, self.wrap())
                )

                eval_metrics = {
                    "charts/mean_success_rate": float(mean_success_rate),
                    "charts/mean_evaluation_return": float(mean_returns),
                } | {
                    f"charts/{task_name}_success_rate": float(success_rate)
                    for task_name, success_rate in mean_success_per_task.items()
                }

                if config.evaluate_on_train:
                    print("- Evaluating on the train set...")
                    _, _, eval_success_rate_per_train_task = (
                        env_config.evaluate_metalearning_on_train(
                            envs=envs,
                            agent=self.wrap(),
                        )
                    )
                    for (
                        task_name,
                        success_rate,
                    ) in eval_success_rate_per_train_task.items():
                        eval_metrics[f"charts/{task_name}_train_success_rate"] = float(
                            success_rate
                        )

                print(
                    f"Mean evaluation success rate: {mean_success_rate:.4f}"
                    + f" return: {mean_returns:.4f}"
                )

                if track:
                    log(eval_metrics, step=global_step)

                if checkpoint_manager is not None:
                    checkpoint_manager.save(
                        global_step,
                        args=get_checkpoint_save_args(
                            self,
                            envs,
                            global_step,
                            episodes_ended,
                            run_timestamp,
                        ),
                        metrics={
                            k.removeprefix("charts/"): v
                            for k, v in eval_metrics.items()
                        },
                    )
                    print("- Saved Model")

            # Logging
            print(
                {
                    k: v
                    for k, v in logs.items()
                    if not (k.startswith("nn") or k.startswith("data"))
                }
            )
            sps = global_step / (time.time() - start_time)
            print("- SPS: ", sps)
            if track:
                log({"charts/SPS": sps} | logs, step=global_step)

        eval_envs.close()
        del eval_envs

        return self

def make_top20_mask_per_task(q_score: np.ndarray, task_idx: np.ndarray, num_tasks: int, top_p: float = 0.2):
    thr = np.empty((num_tasks,), dtype=np.float32)
    mask = np.zeros_like(q_score, dtype=bool)
    q = q_score.astype(np.float32)

    for t in range(num_tasks):
        m = (task_idx == t)
        if not np.any(m):
            thr[t] = np.nan
            continue
        thr[t] = np.quantile(q[m], 1.0 - top_p)
        mask[m] = (q[m] >= thr[t])
    return mask, thr


def save_probe_npz(path: str, buf_idx, task_idx, q_score, is_top, xyz, thr, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        buf_idx=buf_idx.astype(np.int32),
        task_idx=task_idx.astype(np.int32),
        q_score=q_score.astype(np.float32),
        is_top=is_top.astype(np.uint8),
        xyz=xyz.astype(np.float32),
        thr=thr.astype(np.float32),
        meta_json=np.array(json.dumps(meta), dtype=object),
    )

PROBE_EVERY = 50_000
K_PER_TASK = 1024
TOP_P = 0.2
XYZ_IDXS = (0, 1, 2)  # <- 네 obs에서 x,y,z 위치로 바꿔야 함


class OffPolicyAlgorithm(
    Algorithm[
        AlgorithmConfigType, OffPolicyTrainingConfig, EnvConfig, ReplayBufferSamples
    ],
    Generic[AlgorithmConfigType],
):

    @abc.abstractmethod
    def q_min(self, obs:np.ndarray, act:np.ndarray) -> np.ndarray: ...

    @abc.abstractmethod
    def spawn_replay_buffer(
        self, env_config: EnvConfig, config: OffPolicyTrainingConfig, seed: int = 1
    ) -> AbstractReplayBuffer: ...

    @abc.abstractmethod
    def update(self, data: ReplayBufferSamples) -> tuple[Self, LogDict, npt.NDArray]: ...

    @abc.abstractmethod
    def sample_action(self, observation: Observation) -> tuple[Self, Action]: ...

    @abc.abstractmethod
    def eval_action(self, observations: Observation) -> Action: ...

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
        del env_mask
        pass  # For evaluation interface compatibility

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

        # ============================
        # Success Episode Accumulator
        # ============================
        # env index 별로 현재 에피소드 trajectory를 누적한다.
        ep_obs = [[] for _ in range(envs.num_envs)]
        ep_next_obs = [[] for _ in range(envs.num_envs)]
        ep_actions = [[] for _ in range(envs.num_envs)]
        ep_rewards = [[] for _ in range(envs.num_envs)]
        ep_dones = [[] for _ in range(envs.num_envs)]


        done = np.full((envs.num_envs,), False)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        replay_buffer = self.spawn_replay_buffer(env_config, config, seed)
        if buffer_checkpoint is not None:
            replay_buffer.load_checkpoint(buffer_checkpoint)

        start_time = time.time()

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
                buffer_obs = np.where(
                    done[:, None], np.stack(infos["final_obs"]), next_obs
                )

            # ============================
            # Accumulate full episode transitions (per env)
            # ============================
            for i in range(envs.num_envs):
                ep_obs[i].append(obs[i].copy())
                ep_next_obs[i].append(buffer_obs[i].copy())
                ep_actions[i].append(actions[i].copy())
                ep_rewards[i].append(float(rewards[i]))
                ep_dones[i].append(bool(done[i]))

            replay_buffer.add(obs, buffer_obs, actions, rewards, done)

            obs = next_obs

            for i, env_ended in enumerate(done):
                if env_ended:
                    global_episodic_return.append(
                        infos["final_info"]["episode"]["r"][i]
                    )
                    global_episodic_length.append(
                        infos["final_info"]["episode"]["l"][i]
                    )
                    episodes_ended += 1

                    finfo = infos["final_info"]
                    success_i = float(finfo["success"][i]) if "success" in finfo else 0.0
                    ep_o_last = ep_obs[i][-1]
                    task_onehot = ep_o_last[-self.num_tasks:]
                    task_idx = int(np.argmax(task_onehot))
                    # (A) success episode 저장
                    if success_i >= 1.0:
                        ep_o = np.asarray(ep_obs[i])
                        ep_no = np.asarray(ep_next_obs[i])
                        ep_a = np.asarray(ep_actions[i])
                        ep_r = np.asarray(ep_rewards[i], dtype=np.float32)
                        ep_d = np.asarray(ep_dones[i], dtype=np.bool_)

                        task_onehot = ep_o[-1, -self.num_tasks:]
                        task_idx = int(np.argmax(task_onehot))

                        if hasattr(replay_buffer, "add_success_episode"):
                            replay_buffer.add_success_episode(
                                observations=ep_o,
                                next_observations=ep_no,
                                actions=ep_a,
                                rewards=ep_r,
                                dones=ep_d,
                                task_idx=task_idx,
                            )
                    # (B) success EMA 업데이트 (기존 코드 유지하되 success_i 재사용)
                    if hasattr(replay_buffer, "update_success_ema"):
                        task_success = np.zeros(self.num_tasks)
                        mask = np.zeros(self.num_tasks, dtype=bool)

                        # ⚠️ 여기 아직 env i == task i 가정이 들어있음.
                        # 일단 그대로 두되, 나중에 task_idx로 바꾸는 게 더 안전함.
                        task_success[task_idx] = success_i
                        mask[task_idx] = True

                        replay_buffer.update_success_ema(task_success, mask=mask)

                    # (C) episode accumulator clear
                    ep_obs[i].clear()
                    ep_next_obs[i].clear()
                    ep_actions[i].clear()
                    ep_rewards[i].clear()
                    ep_dones[i].clear()

            if global_step % 500 == 0 and global_episodic_return:
                print(
                    f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
                )

                if track:
                    log(
                        {
                            "charts/mean_episodic_return": np.mean(
                                list(global_episodic_return)
                            ),
                            "charts/mean_episodic_length": np.mean(
                                list(global_episodic_length)
                            ),
                        },
                        step=total_steps,
                    )

            if global_step > config.warmstart_steps:
                # Update the agent with data
                data = replay_buffer.sample(config.batch_size)
                success_ema = replay_buffer.success_ema if hasattr(replay_buffer, "success_ema") else None
                il_data = None
                il_shared_data = None
                if (
                    hasattr(replay_buffer, "success_ready")
                    and hasattr(replay_buffer, "sample_success")
                    and replay_buffer.success_ready(min_size=256)
                ):
                    il_data = replay_buffer.sample_success(batch_size=256)
                    
                # shared overlap 샘플 (버퍼에서 이미 "겹치는 것만" 뽑힘)
                if hasattr(replay_buffer, "sample_success_shared"):
                    # shared 비율은 일단 128 정도로 시작 (256의 절반)
                    il_shared_data, _, _ = replay_buffer.sample_success_shared(
                        batch_size=128,
                        min_tasks=2,   # MT10은 2, MT50도 2~3부터
                        )

                if isinstance(data, PrioritizedReplayBatch):
                    self, logs, td_errs = self.update(
                        data.samples,
                        success_ema=success_ema,
                        il_data=il_data,
                        il_shared_data=il_shared_data,  
                    )
                    td_errs_np = np.asarray(jax.device_get(td_errs)).reshape(-1)
                    replay_buffer.update_td_priorities(data.buf_idx, data.task_idx, td_errs_np)
                else:
                    self, logs, _ = self.update(
                        data,
                        success_ema=success_ema,
                        il_data=il_data,
                        il_shared_data=il_shared_data,
                    )
                
                if (total_steps % SNAP_EVERY) == 0 and total_steps > 0:
                    save_buffer_npz(
                        f"buffer_snaps/buffer_step{total_steps:08d}.npz",
                        replay_buffer,
                        meta={"total_steps": int(total_steps), "seed": int(seed)},
                    )
                if (total_steps % PROBE_EVERY) == 0 and total_steps > 0 and global_step > config.warmstart_steps:
                    rng = np.random.default_rng(seed + total_steps)  # step-dependent, 재현 가능
                    buf_idx, task_idx, ob, ac = probe_sample_per_task(replay_buffer, rng, K_PER_TASK)

                    q_score = np.asarray(self.q_min(ob, ac)).reshape(-1)  # (B,)
                    is_top, thr = make_top20_mask_per_task(q_score, task_idx, self.num_tasks, top_p=TOP_P)

                    xyz = ob[:, XYZ_IDXS]  # (B,3)

                    # (선택) 간단 요약 통계 (W&B에 찍기 좋음)
                    # task별 selected vs all mean distance 같은 걸 만들어도 됨
                    meta = {"total_steps": int(total_steps), "k_per_task": int(K_PER_TASK), "top_p": float(TOP_P)}

                    save_probe_npz(
                        f"buffer_snaps/probe_step{total_steps:08d}.npz",
                        buf_idx, task_idx, q_score, is_top, xyz, thr, meta
                    )

                    # wandb/log에도 요약값만 올리고 싶으면:
                    if track:
                        # 예: 전체에서 selected 비율(항상 0.2 근처지만 task별 결측 등 체크용)
                        log({"probe/selected_frac": float(is_top.mean())}, step=total_steps)

                # Logging
                if global_step % 100 == 0:
                    sps_steps = (global_step - start_step) * envs.num_envs
                    sps = int(sps_steps / (time.time() - start_time))
                    print("SPS:", sps)
                    if track:
                        log({"charts/SPS": sps} | logs, step=total_steps)
                    if track and hasattr(replay_buffer, "update_success_ema"):
                        ema = np.asarray(replay_buffer.success_ema).reshape(-1)
                        p = np.asarray(replay_buffer.get_task_sampling_probs()).reshape(-1)

                        metrics = {
                            "sampling/success_ema_mean": float(ema.mean()),
                            "sampling/success_ema_min": float(ema.min()),
                            "sampling/success_ema_max": float(ema.max()),
                            "sampling/p_entropy": float(-(p * np.log(p + 1e-8)).sum()),  # 샘플링이 얼마나 쏠리는지
                        }
                        for t in range(len(ema)):
                            metrics[f"sampling/success_ema/t{t}"] = float(ema[t])
                            metrics[f"sampling/task_sampling_probs/t{t}"] = float(p[t])

                        log(metrics, step=total_steps)

                # Evaluation
                if (
                    config.evaluation_frequency > 0
                    and episodes_ended % config.evaluation_frequency == 0
                    and done.any()
                    and global_step > 0
                ):
                    mean_success_rate, mean_returns, mean_success_per_task = (
                        env_config.evaluate(envs, self)
                    )
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

                    # Checkpointing
                    if checkpoint_manager is not None:
                        if not done.all():
                            raise NotImplementedError(
                                "Checkpointing currently doesn't work for the case where evaluation is run before all envs have finished their episodes / are about to be reset."
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
                            metrics={
                                k.removeprefix("charts/"): v
                                for k, v in eval_metrics.items()
                            },
                        )

                    # Reset envs again to exit eval mode
                    obs, _ = envs.reset()

        return self


class OnPolicyAlgorithm(
    Algorithm[AlgorithmConfigType, OnPolicyTrainingConfig, EnvConfig, Rollout],
    Generic[AlgorithmConfigType],
):
    @abc.abstractmethod
    def sample_action_and_aux(
        self, observation: Observation
    ) -> tuple[Self, Action, AuxPolicyOutputs]: ...

    @abc.abstractmethod
    def sample_action(self, observation: Observation) -> tuple[Self, Action]: ...

    @abc.abstractmethod
    def eval_action(self, observations: Observation) -> Action: ...

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
        del env_mask
        pass  # For evaluation interface compatibility

    @abc.abstractmethod
    def update(
        self,
        data: Rollout,
        dones: Float[npt.NDArray, "task 1"],
        next_obs: Float[Observation, " task"] | None = None,
    ) -> tuple[Self, LogDict]: ...

    def spawn_rollout_buffer(
        self,
        env_config: EnvConfig,
        training_config: OnPolicyTrainingConfig,
        seed: int | None = None,
    ) -> MultiTaskRolloutBuffer:
        return MultiTaskRolloutBuffer(
            training_config.rollout_steps,
            self.num_tasks,
            env_config.observation_space,
            env_config.action_space,
            seed,
        )

    @override
    def train(
        self,
        config: OnPolicyTrainingConfig,
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

        episode_started = np.ones((envs.num_envs,))
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        rollout_buffer = self.spawn_rollout_buffer(env_config, config, seed)

        start_time = time.time()

        for global_step in range(start_step, config.total_steps // envs.num_envs):
            total_steps = global_step * envs.num_envs

            self, actions, aux_policy_outs = self.sample_action_and_aux(obs)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            rollout_buffer.add(
                obs,
                actions,
                rewards,
                episode_started,
                value=aux_policy_outs.get("value"),
                log_prob=aux_policy_outs.get("log_prob"),
                mean=aux_policy_outs.get("mean"),
                std=aux_policy_outs.get("std"),
            )

            episode_started = np.logical_or(terminations, truncations)
            obs = next_obs

            for i, env_ended in enumerate(episode_started):
                if env_ended:
                    global_episodic_return.append(
                        infos["final_info"]["episode"]["r"][i]
                    )
                    global_episodic_length.append(
                        infos["final_info"]["episode"]["l"][i]
                    )
                    episodes_ended += 1

            if global_step % 500 == 0 and global_episodic_return:
                print(
                    f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
                )
                if track:
                    log(
                        {
                            "charts/mean_episodic_return": np.mean(
                                list(global_episodic_return)
                            ),
                            "charts/mean_episodic_length": np.mean(
                                list(global_episodic_length)
                            ),
                        },
                        step=total_steps,
                    )

            # Logging
            if global_step % 1_000 == 0:
                sps_steps = (global_step - start_step) * envs.num_envs
                sps = int(sps_steps / (time.time() - start_time))
                print("SPS:", sps)

                if track:
                    log({"charts/SPS": sps}, step=total_steps)

            if rollout_buffer.ready:
                rollouts = rollout_buffer.get()
                self, logs = self.update(
                    rollouts,
                    dones=terminations,
                    next_obs=np.where(
                        episode_started[:, None], np.stack(infos["final_obs"]), next_obs
                    ),
                )
                rollout_buffer.reset()

                if track:
                    log(logs, step=total_steps)

                # Evaluation
                if (
                    config.evaluation_frequency > 0
                    and episodes_ended % config.evaluation_frequency == 0
                    and episode_started.any()
                    and global_step > 0
                ):
                    mean_success_rate, mean_returns, mean_success_per_task = (
                        env_config.evaluate(envs, self)
                    )
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

                    # Checkpointing
                    if checkpoint_manager is not None:
                        if not episode_started.all():
                            raise NotImplementedError(
                                "Checkpointing currently doesn't work for the case where evaluation is run before all envs have finished their episodes / are about to be reset."
                            )

                        checkpoint_manager.save(
                            total_steps,
                            args=get_checkpoint_save_args(
                                self,
                                envs,
                                global_step,
                                episodes_ended,
                                run_timestamp,
                            ),
                            metrics={
                                k.removeprefix("charts/"): v
                                for k, v in eval_metrics.items()
                            },
                        )

                    # Reset envs again to exit eval mode
                    obs, _ = envs.reset()
                    episode_started = np.ones((envs.num_envs,))

        return self
