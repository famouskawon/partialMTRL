import abc
from typing import override

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from jaxtyping import Float

from metaworld_algorithms.types import (
    Action,
    Observation,
    RNNState,
    ReplayBufferCheckpoint,
    ReplayBufferSamples,
    Rollout,
    PrioritizedReplayBatch,
)



class AbstractReplayBuffer(abc.ABC):
    """Replay buffer for the single-task environments.

    Each sampling step, it samples a batch for each task, returning a batch of shape (batch_size,).
    When pushing samples to the buffer, the buffer accepts inputs of arbitrary batch dimensions.
    """

    obs: Float[Observation, " buffer_size"]
    actions: Float[Action, " buffer_size"]
    rewards: Float[npt.NDArray, "buffer_size 1"]
    next_obs: Float[Observation, " buffer_size"]
    dones: Float[npt.NDArray, "buffer_size 1"]
    pos: int

    @abc.abstractmethod
    def __init__(
        self,
        capacity: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
    ) -> None: ...

    @abc.abstractmethod
    def reset(self) -> None: ...

    @abc.abstractmethod
    def checkpoint(self) -> ReplayBufferCheckpoint: ...

    @abc.abstractmethod
    def load_checkpoint(self, ckpt: ReplayBufferCheckpoint) -> None: ...

    @abc.abstractmethod
    def add(
        self,
        obs: Observation,
        next_obs: Observation,
        action: Action,
        reward: Float[npt.NDArray, " *batch"],
        done: Float[npt.NDArray, " *batch"],
    ) -> None:
        """Add a batch of samples to the buffer."""
        ...

    @abc.abstractmethod
    def sample(self, batch_size: int) -> ReplayBufferSamples: ...


class ReplayBuffer(AbstractReplayBuffer):
    """Replay buffer for the single-task environments.

    Each sampling step, it samples a batch for each task, returning a batch of shape (batch_size,).
    When pushing samples to the buffer, the buffer accepts inputs of arbitrary batch dimensions.
    """

    obs: Float[Observation, " buffer_size"]
    actions: Float[Action, " buffer_size"]
    rewards: Float[npt.NDArray, "buffer_size 1"]
    next_obs: Float[Observation, " buffer_size"]
    dones: Float[npt.NDArray, "buffer_size 1"]
    pos: int

    def __init__(
        self,
        capacity: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
    ) -> None:
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._obs_shape = np.array(env_obs_space.shape).prod()
        self._action_shape = np.array(env_action_space.shape).prod()
        self.full = False

        self.reset()  # Init buffer

    @override
    def reset(self):
        """Reinitialize the buffer."""
        self.obs = np.zeros((self.capacity, self._obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self._action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self._obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.pos = 0

    @override
    def checkpoint(self) -> ReplayBufferCheckpoint:
        return {
            "data": {
                "obs": self.obs,
                "actions": self.actions,
                "rewards": self.rewards,
                "next_obs": self.next_obs,
                "dones": self.dones,
                "pos": self.pos,
                "full": self.full,
            },
            "rng_state": self._rng.bit_generator.state,
        }

    @override
    def load_checkpoint(self, ckpt: ReplayBufferCheckpoint) -> None:
        for key in ["data", "rng_state"]:
            assert key in ckpt

        for key in ["obs", "actions", "rewards", "next_obs", "dones", "pos", "full"]:
            assert key in ckpt["data"]
            setattr(self, key, ckpt["data"][key])

        self._rng.bit_generator.state = ckpt["rng_state"]

    @override
    def add(
        self,
        obs: Observation,
        next_obs: Observation,
        action: Action,
        reward: Float[npt.NDArray, " *batch"],
        done: Float[npt.NDArray, " *batch"],
    ) -> None:
        """Add a batch of samples to the buffer."""
        if obs.ndim >= 2:
            assert (
                obs.shape[0] == action.shape[0] == reward.shape[0] == done.shape[0]
            ), "Batch size must be the same for all transition data."

            # Flatten any batch dims
            flat_obs = obs.reshape(-1, obs.shape[-1])
            flat_next_obs = next_obs.reshape(-1, next_obs.shape[-1])
            flat_action = action.reshape(-1, action.shape[-1])
            flat_reward = reward.reshape(
                -1, 1
            )  # Keep the last dim as 1 for consistency
            flat_done = done.reshape(-1, 1)  # Keep the last dim as 1 for consistency

            # Calculate number of new transitions
            n_transitions = len(flat_obs)

            # Handle buffer wraparound
            indices = np.arange(self.pos, self.pos + n_transitions) % self.capacity

            # Store the transitions
            self.obs[indices] = flat_obs
            self.next_obs[indices] = flat_next_obs
            self.actions[indices] = flat_action
            self.rewards[indices] = flat_reward
            self.dones[indices] = flat_done

            self.pos = (self.pos + n_transitions) % self.capacity
            if self.pos > self.capacity and not self.full:
                self.full = True
        else:
            self.obs[self.pos] = obs.copy()
            self.actions[self.pos] = action.copy()
            self.next_obs[self.pos] = next_obs.copy()
            self.dones[self.pos] = done.copy().reshape(-1, 1)
            self.rewards[self.pos] = reward.copy().reshape(-1, 1)

            self.pos += 1

        if self.pos > self.capacity and not self.full:
            self.full = True
        self.pos %= self.capacity

    @override
    def sample(self, batch_size: int) -> ReplayBufferSamples:
        sample_idx = self._rng.integers(
            low=0,
            high=max(self.pos if not self.full else self.capacity, batch_size),
            size=(batch_size,),
        )

        batch = (
            self.obs[sample_idx],
            self.actions[sample_idx],
            self.next_obs[sample_idx],
            self.dones[sample_idx],
            self.rewards[sample_idx],
        )

        return ReplayBufferSamples(*batch)


class MultiTaskReplayBuffer(AbstractReplayBuffer):
    """Replay buffer for the multi-task benchmarks.

    Each sampling step, it samples a batch for each task, returning a batch of shape (batch_size, num_tasks,).
    When pushing samples to the buffer, the buffer only accepts inputs with batch shape (num_tasks,).
    """

    obs: Float[Observation, "buffer_size task"]
    actions: Float[Action, "buffer_size task"]
    rewards: Float[npt.NDArray, "buffer_size task 1"]
    next_obs: Float[Observation, "buffer_size task"]
    dones: Float[npt.NDArray, "buffer_size task 1"]
    pos: int

    def __init__(
        self,
        total_capacity: int,
        num_tasks: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
        max_steps: int = 500,
    ) -> None:
        assert total_capacity % num_tasks == 0, (
            "Total capacity must be divisible by the number of tasks."
        )
        self.capacity = total_capacity // num_tasks
        self.num_tasks = num_tasks
        self._rng = np.random.default_rng(seed)
        self._obs_shape = np.array(env_obs_space.shape).prod()
        self._action_shape = np.array(env_action_space.shape).prod()
        self.full = False

        # all needed for reward smoothing --> Reggie's original idea about scale and smoothness mattering
        self.max_steps = max_steps
        self.current_trajectory_start = 0

        self.reset(save_rewards=False)  # Init buffer

    @override
    def reset(self, save_rewards=False):
        """Reinitialize the buffer."""
        self.obs = np.zeros(
            (self.capacity, self.num_tasks, self._obs_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.capacity, self.num_tasks, self._action_shape), dtype=np.float32
        )
        self.rewards = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.next_obs = np.zeros(
            (self.capacity, self.num_tasks, self._obs_shape), dtype=np.float32
        )
        self.dones = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.pos = 0

        if save_rewards:
            self.org_rewards = np.zeros(
                (self.capacity, self.num_tasks, 1), dtype=np.float32
            )
            self.traj_start = 0

    @override
    def checkpoint(self) -> ReplayBufferCheckpoint:
        return {
            "data": {
                "obs": self.obs,
                "actions": self.actions,
                "rewards": self.rewards,
                "next_obs": self.next_obs,
                "dones": self.dones,
                "pos": self.pos,
                "full": self.full,
            },
            "rng_state": self._rng.bit_generator.state,
        }

    @override
    def load_checkpoint(self, ckpt: ReplayBufferCheckpoint) -> None:
        for key in ["data", "rng_state"]:
            assert key in ckpt

        for key in ["obs", "actions", "rewards", "next_obs", "dones", "pos", "full"]:
            assert key in ckpt["data"]
            setattr(self, key, ckpt["data"][key])

        self._rng.bit_generator.state = ckpt["rng_state"]

    @override
    def add(
        self,
        obs: Float[Observation, " task"],
        next_obs: Float[Observation, " task"],
        action: Float[Action, " task"],
        reward: Float[npt.NDArray, " task"],
        done: Float[npt.NDArray, " task"],
    ) -> None:
        """Add a batch of samples to the buffer."""
        # NOTE: assuming batch dim = task dim
        assert (
            obs.ndim == 2 and action.ndim == 2 and reward.ndim <= 2 and done.ndim <= 2
        )
        assert (
            obs.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == done.shape[0]
            == self.num_tasks
        )

        self.obs[self.pos] = obs.copy()
        self.actions[self.pos] = action.copy()
        self.next_obs[self.pos] = next_obs.copy()
        self.dones[self.pos] = done.copy().reshape(-1, 1)
        self.rewards[self.pos] = reward.reshape(-1, 1).copy()

        self.pos = self.pos + 1
        if self.pos == self.capacity:
            self.full = True

        self.pos = self.pos % self.capacity

    def single_task_sample(self, task_idx: int, batch_size: int) -> ReplayBufferSamples:
        assert task_idx < self.num_tasks, "Task index out of bounds."

        sample_idx = self._rng.integers(
            low=0,
            high=max(self.pos if not self.full else self.capacity, batch_size),
            size=(batch_size,),
        )

        batch = (
            self.obs[sample_idx][task_idx],
            self.actions[sample_idx][task_idx],
            self.next_obs[sample_idx][task_idx],
            self.dones[sample_idx][task_idx],
            self.rewards[sample_idx][task_idx],
        )

        return ReplayBufferSamples(*batch)

    @override
    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """Sample a batch of size `single_task_batch_size` for each task.

        Args:
            batch_size (int): The total batch size. Must be divisible by number of tasks

        Returns:
            ReplayBufferSamples: A batch of samples of batch shape (batch_size,).
        """
        assert batch_size % self.num_tasks == 0, (
            "Batch size must be divisible by the number of tasks."
        )
        single_task_batch_size = batch_size // self.num_tasks

        sample_idx = self._rng.integers(
            low=0,
            high=max(
                self.pos if not self.full else self.capacity, single_task_batch_size
            ),
            size=(single_task_batch_size,),
        )

        batch = (
            self.obs[sample_idx],
            self.actions[sample_idx],
            self.next_obs[sample_idx],
            self.dones[sample_idx],
            self.rewards[sample_idx],
        )

        mt_batch_size = single_task_batch_size * self.num_tasks
        batch = map(lambda x: x.reshape(mt_batch_size, *x.shape[2:]), batch)

        return ReplayBufferSamples(*batch)


class MultiTaskRolloutBuffer:
    num_rollout_steps: int
    num_tasks: int
    pos: int

    observations: Float[Observation, "timestep task"]
    actions: Float[Action, "timestep task"]
    rewards: Float[npt.NDArray, "timestep task 1"]
    dones: Float[npt.NDArray, "timestep task 1"]

    values: Float[npt.NDArray, "timestep task 1"]
    log_probs: Float[npt.NDArray, "timestep task 1"]
    means: Float[Action, "timestep task"]
    stds: Float[Action, "timestep task"]
    rnn_states: Float[RNNState, "timestep task"] | None = None

    def __init__(
        self,
        num_rollout_steps: int,
        num_tasks: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        rnn_state_dim: int | None = None,
        dtype: npt.DTypeLike = np.float32,
        seed: int | None = None,
    ) -> None:
        self.num_rollout_steps = num_rollout_steps
        self.num_tasks = num_tasks
        self._rng = np.random.default_rng(seed)
        self._obs_shape = np.array(env_obs_space.shape).prod()
        self._action_shape = np.array(env_action_space.shape).prod()
        self._rnn_state_dim = rnn_state_dim
        self.dtype = dtype
        self.reset()  # Init buffer

    def reset(self) -> None:
        """Reinitialize the buffer."""
        self.observations = np.zeros(
            (self.num_rollout_steps, self.num_tasks, self._obs_shape), dtype=self.dtype
        )
        self.actions = np.zeros(
            (self.num_rollout_steps, self.num_tasks, self._action_shape),
            dtype=self.dtype,
        )
        self.rewards = np.zeros(
            (self.num_rollout_steps, self.num_tasks, 1), dtype=self.dtype
        )
        self.dones = np.zeros(
            (self.num_rollout_steps, self.num_tasks, 1), dtype=self.dtype
        )

        self.log_probs = np.zeros(
            (self.num_rollout_steps, self.num_tasks, 1), dtype=self.dtype
        )
        self.values = np.zeros_like(self.rewards)
        self.means = np.zeros_like(self.actions)
        self.stds = np.zeros_like(self.actions)

        if self._rnn_state_dim is not None:
            self.rnn_states = np.zeros(
                (self.num_rollout_steps, self.num_tasks, self._rnn_state_dim),
                dtype=self.dtype,
            )

        self.pos = 0

    @property
    def ready(self) -> bool:
        return self.pos == self.num_rollout_steps

    def add(
        self,
        obs: Float[Observation, " task"],
        action: Float[Action, " task"],
        reward: Float[npt.NDArray, " task"],
        done: Float[npt.NDArray, " task"],
        value: Float[npt.NDArray, " task"] | None = None,
        log_prob: Float[npt.NDArray, " task"] | None = None,
        mean: Float[Action, " task"] | None = None,
        std: Float[Action, " task"] | None = None,
        rnn_state: Float[RNNState, " task"] | None = None,
    ):
        # NOTE: assuming batch dim = task dim
        assert (
            obs.ndim == 2 and action.ndim == 2 and reward.ndim <= 2 and done.ndim <= 2
        )
        assert (
            obs.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == done.shape[0]
            == self.num_tasks
        )

        self.observations[self.pos] = obs.copy()
        self.actions[self.pos] = action.copy()
        self.rewards[self.pos] = reward.copy().reshape(-1, 1)
        self.dones[self.pos] = done.copy().reshape(-1, 1)

        if value is not None:
            self.values[self.pos] = value.copy()
        if log_prob is not None:
            self.log_probs[self.pos] = log_prob.reshape(-1, 1).copy()
        if mean is not None:
            self.means[self.pos] = mean.copy()
        if std is not None:
            self.stds[self.pos] = std.copy()
        if rnn_state is not None:
            assert self.rnn_states is not None
            self.rnn_states[self.pos] = rnn_state.copy()

        self.pos += 1

    def get(
        self,
    ) -> Rollout:
        return Rollout(
            self.observations,
            self.actions,
            self.rewards,
            self.dones,
            self.log_probs,
            self.means,
            self.stds,
            self.values,
            self.rnn_states,
        )

class InterTaskMultiTaskReplayBuffer(MultiTaskReplayBuffer):
    def __init__(
        self,
        total_capacity: int,
        num_tasks: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
        max_steps: int = 500,
        task_beta: float = 1.0,
        eps: float = 1e-6,
        success_ema_tau: float = 0.01,
        use_inter_task_sampling: bool = False,
        use_intra_task_sampling: bool = False,
        prio_alpha: float = 0.6,
        prio_eps: float = 1e-6,
        # ===== NEW: success episode buffer (IL buffer) =====
        use_success_episode_buffer: bool = True,
        success_total_capacity: int | None = None,  # transitions capacity
        
    ) -> None:
        super().__init__(
            total_capacity=total_capacity,
            num_tasks=num_tasks,
            env_obs_space=env_obs_space,
            env_action_space=env_action_space,
            seed=seed,
            max_steps=max_steps,
        )
        self.task_beta = float(task_beta)
        self.eps = float(eps)
        self.success_ema_tau = float(success_ema_tau)
        self.success_ema = np.zeros((self.num_tasks,), dtype=np.float32)
        self.use_inter_task_sampling = use_inter_task_sampling
        self.prio_alpha = float(prio_alpha)   # 예: 0.6
        self.prio_eps   = float(prio_eps)     # 예: 1e-6
        self.td_priority = np.ones((self.capacity, self.num_tasks), dtype=np.float32)

        # ===== NEW: success episode buffer (flat/global) =====
        self.use_success_episode_buffer = bool(use_success_episode_buffer)
        if success_total_capacity is None:
            # default: 1/4 of main total transitions
            success_total_capacity = max(1, (self.capacity * self.num_tasks) // 4)

        self.suc_capacity = int(success_total_capacity)
        self.suc_pos = 0
        self.suc_full = False

        self.suc_obs = np.zeros((self.suc_capacity, self._obs_shape), dtype=np.float32)
        self.suc_actions = np.zeros((self.suc_capacity, self._action_shape), dtype=np.float32)
        self.suc_next_obs = np.zeros((self.suc_capacity, self._obs_shape), dtype=np.float32)
        self.suc_rewards = np.zeros((self.suc_capacity, 1), dtype=np.float32)
        self.suc_dones = np.zeros((self.suc_capacity, 1), dtype=np.float32)
        self.suc_task_idx = np.zeros((self.suc_capacity,), dtype=np.int32)

    def update_success_ema(
        self,
        task_success: npt.NDArray,
        mask: npt.NDArray | None = None,
    ) -> None:
        task_success = np.asarray(task_success, dtype=np.float32).reshape(-1)
        assert task_success.shape[0] == self.num_tasks

        if mask is None:
            mask = np.ones((self.num_tasks,), dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool).reshape(-1)
            assert mask.shape[0] == self.num_tasks

        tau = self.success_ema_tau
        self.success_ema[mask] = (
            (1 - tau) * self.success_ema[mask] + tau * task_success[mask]
        )

    # -----------------------
    # NEW: 성공 에피소드(전이들) 통째로 추가
    # base.py의 add_success_episode 호출과 "정확히" 맞추는 시그니처
    # -----------------------
    def add_success_episode(
        self,
        observations: np.ndarray,        # (T, obs_dim)
        next_observations: np.ndarray,   # (T, obs_dim)
        actions: np.ndarray,             # (T, act_dim)
        rewards: np.ndarray,             # (T,) or (T,1)
        dones: np.ndarray,               # (T,) bool
        task_idx: int,
    ) -> None:
        if not self.use_success_episode_buffer:
            return
        if task_idx < 0 or task_idx >= self.num_tasks:
            return

        obs = np.asarray(observations, dtype=np.float32)
        nxt = np.asarray(next_observations, dtype=np.float32)
        act = np.asarray(actions, dtype=np.float32)
        rew = np.asarray(rewards, dtype=np.float32).reshape(-1)
        dn = np.asarray(dones).reshape(-1).astype(np.float32)

        T = obs.shape[0]
        if T == 0:
            return

        # 전이 단위로 flat buffer에 push
        for t in range(T):
            i = self.suc_pos
            self.suc_obs[i] = obs[t]
            self.suc_next_obs[i] = nxt[t]
            self.suc_actions[i] = act[t]
            self.suc_rewards[i] = np.array([rew[t]], dtype=np.float32)
            self.suc_dones[i] = np.array([dn[t]], dtype=np.float32)
            self.suc_task_idx[i] = int(task_idx)

            self.suc_pos += 1
            if self.suc_pos >= self.suc_capacity:
                self.suc_full = True
                self.suc_pos = 0

    def success_ready(self, min_size: int) -> bool:
        size = self.suc_capacity if self.suc_full else self.suc_pos
        return size >= min_size

    def sample_success(self, batch_size: int) -> ReplayBufferSamples:
        size = self.suc_capacity if self.suc_full else self.suc_pos
        if size <= 0:
            raise RuntimeError("Success buffer empty.")
        idx = self._rng.integers(low=0, high=max(size, batch_size), size=(batch_size,))
        batch = (
            self.suc_obs[idx],
            self.suc_actions[idx],
            self.suc_next_obs[idx],
            self.suc_dones[idx],
            self.suc_rewards[idx],
        )
        return ReplayBufferSamples(*batch)

    def _get_valid_size(self) -> int:
        return self.capacity if self.full else self.pos

    def get_task_sampling_probs(self) -> np.ndarray:
        # p(task) ∝ (1 - success_ema)^beta
        diff = (1.0 - self.success_ema).clip(0.0, 1.0)
        w = (diff + self.eps) ** self.task_beta
        p = w / (w.sum() + 1e-12)
        return p.astype(np.float32)

    def _sample_tasks(self, n: int) -> npt.NDArray:
        return self._rng.choice(self.num_tasks, size=(n,), replace=True, p=self.get_task_sampling_probs()).astype(np.int32)

    def _sample_indices_for_task(self, task_idx: int, n: int) -> np.ndarray:
        size = self._get_valid_size()
        if size <= 0:
            raise RuntimeError("Main buffer is empty.")

        if (not self.use_inter_task_sampling) or (not hasattr(self, "td_priority")):
            # uniform within task
            return self._rng.integers(low=0, high=max(size, n), size=(n,), dtype=np.int32)

        # PER within task
        pr = self.td_priority[:size, task_idx].astype(np.float64)  # (size,)
        pr = np.maximum(pr, self.prio_eps) ** self.prio_alpha
        s = pr.sum()
        if not np.isfinite(s) or s <= 0:
            return self._rng.integers(low=0, high=max(size, n), size=(n,), dtype=np.int32)
        p = pr / s
        return self._rng.choice(size, size=(n,), replace=True, p=p).astype(np.int32)

    def update_td_priorities(
        self,
        buffer_indices: npt.NDArray,
        task_indices: npt.NDArray,
        td_errors: npt.NDArray,
    ) -> None:
        buffer_indices = np.asarray(buffer_indices).reshape(-1).astype(np.int32)
        task_indices = np.asarray(task_indices).reshape(-1).astype(np.int32)
        td_errors = np.asarray(td_errors).reshape(-1)

        pr = np.abs(td_errors).astype(np.float32) + self.prio_eps
        self.td_priority[buffer_indices, task_indices] = pr

    @override
    def sample(self, batch_size: int) -> ReplayBufferSamples | PrioritizedReplayBatch :
        size = self._get_valid_size()
        if size <= 0:
            raise RuntimeError("Main buffer is empty.")

        # 1) task를 batch_size만큼 먼저 뽑는다
        task_idx = self._sample_tasks(batch_size)          # (B,)
        buf_idx = np.empty((batch_size,), dtype=np.int32)  # (B,)

        # 2) task별로 필요한 개수만큼 index를 뽑는다 (intra: on이면 PER, off이면 uniform)
        unique_tasks, counts = np.unique(task_idx, return_counts=True)
        cursor = 0
        for t, c in zip(unique_tasks, counts):
            idxs = self._sample_indices_for_task(int(t), int(c))  # <-- 여기만!
            buf_idx[cursor:cursor + int(c)] = idxs
            cursor += int(c)

        # 3) task 덩어리 편향 방지용 shuffle
        perm = self._rng.permutation(batch_size)
        task_idx = task_idx[perm]
        buf_idx = buf_idx[perm]

        # 4) gather
        obs = self.obs[buf_idx, task_idx]
        actions = self.actions[buf_idx, task_idx]
        next_obs = self.next_obs[buf_idx, task_idx]
        dones = self.dones[buf_idx, task_idx]
        rewards = self.rewards[buf_idx, task_idx]

        return PrioritizedReplayBatch(
            samples=ReplayBufferSamples(obs, actions, next_obs, dones, rewards),
            buf_idx=buf_idx,
            task_idx=task_idx,
        )

# =========================
# IL-RL buffers (NEW)
# =========================

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from jaxtyping import Float

from metaworld_algorithms.types import ReplayBufferSamples


@dataclass
class ILRLReplayBatch:
    """Container returned by ILRLMultiTaskReplayBuffer.sample().

    - rl: RL update batch (from main buffer, prioritized)
    - il: IL(BC) batch (from success-only IL buffer), can be None if not ready
    - rl_indices: indices in main buffer to allow priority updates
    - rl_task_indices: task index per transition (0..num_tasks-1)
    """
    rl: ReplayBufferSamples
    il: ReplayBufferSamples | None
    rl_indices: npt.NDArray  # (batch,)
    rl_task_indices: npt.NDArray  # (batch,)


class ILRLMultiTaskReplayBuffer(MultiTaskReplayBuffer):
    """
    Multi-task replay buffer with:
      1) task-level sampling biased by difficulty (1 - success_ema)
      2) transition-level prioritized sampling using stored td_priority
      3) separate IL buffer storing ONLY (easy tasks) successful transitions

    Notes:
      - TD-error is NOT computed inside the buffer. Call update_td_priorities(...)
        using TD-errors computed by the algorithm.
      - Success EMA is NOT inferred from reward. Call update_success_ema(...)
        when an episode ends (using env info['success']).
    """

    # ---------- init ----------
    def __init__(
        self,
        total_capacity: int,
        num_tasks: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
        max_steps: int = 500,

        # sampling hyperparams
        task_beta: float = 1.0,          # how strongly to focus hard tasks
        prio_alpha: float = 0.6,         # PER alpha
        prio_eps: float = 1e-6,

        # success tracking
        success_ema_tau: float = 0.01,   # EMA update rate
        easy_success_th: float = 0.8,    # defines "easy tasks" for IL buffer

        # IL buffer
        il_total_capacity: int | None = None,  # if None: = total_capacity//4
    ) -> None:
        super().__init__(
            total_capacity=total_capacity,
            num_tasks=num_tasks,
            env_obs_space=env_obs_space,
            env_action_space=env_action_space,
            seed=seed,
            max_steps=max_steps,
        )

        self.task_beta = float(task_beta)
        self.prio_alpha = float(prio_alpha)
        self.prio_eps = float(prio_eps)

        self.success_ema_tau = float(success_ema_tau)
        self.easy_success_th = float(easy_success_th)

        # TD priority for main buffer: shape (capacity, num_tasks)
        # start with all ones (uniform)
        self.td_priority = np.ones((self.capacity, self.num_tasks), dtype=np.float32)

        # success EMA per task
        self.success_ema = np.zeros((self.num_tasks,), dtype=np.float32)

        # ---------- IL buffer (success-only) ----------
        if il_total_capacity is None:
            il_total_capacity = max(1, (self.capacity * self.num_tasks) // 4)

        self.il_capacity = int(il_total_capacity)
        self.il_pos = 0
        self.il_full = False

        self.il_obs = np.zeros((self.il_capacity, self._obs_shape), dtype=np.float32)
        self.il_actions = np.zeros((self.il_capacity, self._action_shape), dtype=np.float32)
        self.il_next_obs = np.zeros((self.il_capacity, self._obs_shape), dtype=np.float32)
        self.il_dones = np.zeros((self.il_capacity, 1), dtype=np.float32)
        self.il_rewards = np.zeros((self.il_capacity, 1), dtype=np.float32)

        # store task id for IL samples (so you can optionally filter per task later)
        self.il_task_idx = np.zeros((self.il_capacity,), dtype=np.int32)

    # ---------- success EMA update (call from rollout) ----------
    def update_success_ema(self, task_success: npt.NDArray) -> None:
        """
        task_success: (num_tasks,) float/bool, typically success of episode end.
        """
        task_success = np.asarray(task_success, dtype=np.float32).reshape(-1)
        assert task_success.shape[0] == self.num_tasks
        tau = self.success_ema_tau
        self.success_ema = (1 - tau) * self.success_ema + tau * task_success

    # ---------- IL buffer add (call from rollout when you detect successful transition) ----------
    def add_il_transition(
        self,
        task_idx: int,
        obs: npt.NDArray,
        next_obs: npt.NDArray,
        action: npt.NDArray,
        reward: float,
        done: float,
    ) -> None:
        """Store one transition into IL buffer (success-only), for easy tasks only."""
        if task_idx < 0 or task_idx >= self.num_tasks:
            return

        # only keep IL data for easy tasks (high success_ema)
        if self.success_ema[task_idx] < self.easy_success_th:
            return

        i = self.il_pos
        self.il_obs[i] = obs.astype(np.float32, copy=False)
        self.il_next_obs[i] = next_obs.astype(np.float32, copy=False)
        self.il_actions[i] = action.astype(np.float32, copy=False)
        self.il_rewards[i] = np.array([reward], dtype=np.float32)
        self.il_dones[i] = np.array([done], dtype=np.float32)
        self.il_task_idx[i] = int(task_idx)

        self.il_pos += 1
        if self.il_pos >= self.il_capacity:
            self.il_full = True
            self.il_pos = 0

    def il_ready(self, min_size: int) -> bool:
        size = self.il_capacity if self.il_full else self.il_pos
        return size >= min_size

    def sample_il(self, batch_size: int) -> ReplayBufferSamples:
        size = self.il_capacity if self.il_full else self.il_pos
        assert size > 0, "IL buffer empty."
        idx = self._rng.integers(low=0, high=max(size, batch_size), size=(batch_size,))
        batch = (
            self.il_obs[idx],
            self.il_actions[idx],
            self.il_next_obs[idx],
            self.il_dones[idx],
            self.il_rewards[idx],
        )
        return ReplayBufferSamples(*batch)

    # ---------- TD priority update (call from algorithm after computing TD errors) ----------
    def update_td_priorities(
        self,
        buffer_indices: npt.NDArray,      # (batch,)
        task_indices: npt.NDArray,        # (batch,)
        td_errors: npt.NDArray,           # (batch,) or (batch,1)
    ) -> None:
        buffer_indices = np.asarray(buffer_indices).reshape(-1)
        task_indices = np.asarray(task_indices).reshape(-1)
        td_errors = np.asarray(td_errors).reshape(-1)

        assert buffer_indices.shape == task_indices.shape == td_errors.shape
        # priority = |td| + eps
        pr = np.abs(td_errors).astype(np.float32) + self.prio_eps

        # write back
        self.td_priority[buffer_indices, task_indices] = pr

    # ---------- sampling ----------
    def _get_valid_size(self) -> int:
        return self.capacity if self.full else self.pos

    def _sample_tasks(self, num_samples: int) -> npt.NDArray:
        """
        Sample task indices with prob ∝ (1 - success_ema)^beta
        """
        diff = (1.0 - self.success_ema).clip(0.0, 1.0)
        w = (diff + 1e-6) ** self.task_beta
        p = w / w.sum()
        return self._rng.choice(self.num_tasks, size=(num_samples,), replace=True, p=p)

    def _sample_indices_for_task(self, task_idx: int, n: int) -> npt.NDArray:
        """
        Sample buffer indices for a given task using stored priorities (PER-style).
        Additionally enforces "TD-error > task mean" approximately by using priorities.
        """
        size = self._get_valid_size()
        if size <= 0:
            raise RuntimeError("Main buffer is empty.")

        pr = self.td_priority[:size, task_idx].astype(np.float64)  # (size,)
        # PER prob ∝ pr^alpha
        pr = np.maximum(pr, self.prio_eps) ** self.prio_alpha
        pr_sum = pr.sum()
        if not np.isfinite(pr_sum) or pr_sum <= 0:
            # fallback to uniform
            return self._rng.integers(low=0, high=max(size, n), size=(n,))
        p = pr / pr_sum
        return self._rng.choice(size, size=(n,), replace=True, p=p)

    def sample(self, batch_size: int, il_batch_size: int | None = None) -> ILRLReplayBatch:
        """
        Returns ILRLReplayBatch. This does NOT affect baseline because only ILRL algorithm will spawn this buffer.
        RL batch composition:
          - pick tasks using success-biased sampling
          - for each chosen task pick transitions using priority sampling
        """
        # RL batch
        task_idx = self._sample_tasks(batch_size)  # (batch,)
        buf_idx = np.empty((batch_size,), dtype=np.int32)

        # task별로 그룹화
        unique_tasks, counts = np.unique(task_idx, return_counts=True)
        cursor = 0
        for t, c in zip(unique_tasks, counts):
            # 이 task에서 필요한 개수 c만큼을 한 번에 샘플
            idxs = self._sample_indices_for_task(int(t), int(c))  # (c,)
            buf_idx[cursor:cursor+c] = idxs
            cursor += c

        # task_idx와 buf_idx의 pairing을 섞어줘야 “task별 index가 덩어리로 몰리는 편향”을 방지
        perm = self._rng.permutation(batch_size)
        task_idx = task_idx[perm]
        buf_idx = buf_idx[perm]

        # gather transitions
        obs = self.obs[buf_idx, task_idx]
        actions = self.actions[buf_idx, task_idx]
        next_obs = self.next_obs[buf_idx, task_idx]
        dones = self.dones[buf_idx, task_idx]
        rewards = self.rewards[buf_idx, task_idx]

        rl = ReplayBufferSamples(obs, actions, next_obs, dones, rewards)

        # IL batch (optional)
        il = None
        if il_batch_size is not None and il_batch_size > 0 and self.il_ready(il_batch_size):
            il = self.sample_il(il_batch_size)

        return ILRLReplayBatch(
            rl=rl,
            il=il,
            rl_indices=buf_idx,
            rl_task_indices=task_idx.astype(np.int32),
        )