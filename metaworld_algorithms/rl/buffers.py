# buffers.py
# - Fixes:
#   (1) NEVER sample from unfilled region (removes max(size, batch_size) bug everywhere)
#   (2) Multi-task sampling + per-task sampling safe when buffer not full
#   (3) Fix MultiTaskReplayBuffer.single_task_sample indexing bug ([:, task_idx])

import abc
from dataclasses import dataclass
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

# =========================
# Helpers
# =========================
def _valid_size(pos: int, full: bool, capacity: int) -> int:
    return capacity if full else pos


def _rand_indices(
    rng: np.random.Generator, size: int, batch_size: int, dtype=np.int32
) -> np.ndarray:
    if size <= 0:
        raise RuntimeError("Replay buffer is empty (size=0).")
    # IMPORTANT: always sample in [0, size)
    return rng.integers(low=0, high=size, size=(batch_size,), dtype=dtype)


# =========================
# Base buffer interfaces
# =========================
class AbstractReplayBuffer(abc.ABC):
    """Replay buffer for single-task or multi-task envs."""

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
    ) -> None: ...

    @abc.abstractmethod
    def sample(self, batch_size: int) -> ReplayBufferSamples: ...


# =========================
# Single-task replay buffer
# =========================
class ReplayBuffer(AbstractReplayBuffer):
    """Single-task replay buffer."""

    def __init__(
        self,
        capacity: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
    ) -> None:
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._obs_shape = int(np.array(env_obs_space.shape).prod())
        self._action_shape = int(np.array(env_action_space.shape).prod())
        self.full = False
        self.reset()

    @override
    def reset(self):
        self.obs = np.zeros((self.capacity, self._obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self._action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self._obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.pos = 0
        self.full = False

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
        # batched add
        if obs.ndim >= 2:
            assert (
                obs.shape[0] == action.shape[0] == reward.shape[0] == done.shape[0]
            ), "Batch size must match."

            flat_obs = obs.reshape(-1, obs.shape[-1])
            flat_next_obs = next_obs.reshape(-1, next_obs.shape[-1])
            flat_action = action.reshape(-1, action.shape[-1])
            flat_reward = reward.reshape(-1, 1)
            flat_done = done.reshape(-1, 1)

            n_transitions = len(flat_obs)
            indices = np.arange(self.pos, self.pos + n_transitions) % self.capacity

            self.obs[indices] = flat_obs
            self.next_obs[indices] = flat_next_obs
            self.actions[indices] = flat_action
            self.rewards[indices] = flat_reward
            self.dones[indices] = flat_done

            self.pos = (self.pos + n_transitions) % self.capacity
            if n_transitions > 0 and (self.full or (self.pos == 0 and n_transitions >= self.capacity)):
                self.full = True
        else:
            self.obs[self.pos] = obs.copy()
            self.actions[self.pos] = action.copy()
            self.next_obs[self.pos] = next_obs.copy()
            self.dones[self.pos] = done.copy().reshape(-1, 1)
            self.rewards[self.pos] = reward.copy().reshape(-1, 1)

            self.pos += 1
            if self.pos >= self.capacity:
                self.full = True
            self.pos %= self.capacity

    @override
    def sample(self, batch_size: int) -> ReplayBufferSamples:
        size = _valid_size(self.pos, self.full, self.capacity)
        sample_idx = _rand_indices(self._rng, size, batch_size)

        batch = (
            self.obs[sample_idx],
            self.actions[sample_idx],
            self.next_obs[sample_idx],
            self.dones[sample_idx],
            self.rewards[sample_idx],
        )
        return ReplayBufferSamples(*batch)


# =========================
# Multi-task replay buffer
# =========================
class MultiTaskReplayBuffer(AbstractReplayBuffer):
    """Multi-task replay buffer.

    Stored as (capacity_per_task, num_tasks, dim).
    sample(batch_size): returns flattened (batch_size, dim) with batch_size divisible by num_tasks.
    """

    def __init__(
        self,
        total_capacity: int,
        num_tasks: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
        max_steps: int = 500,
    ) -> None:
        assert total_capacity % num_tasks == 0, "total_capacity must be divisible by num_tasks."
        self.capacity = int(total_capacity // num_tasks)  # per-task length
        self.num_tasks = int(num_tasks)
        self._rng = np.random.default_rng(seed)
        self._obs_shape = int(np.array(env_obs_space.shape).prod())
        self._action_shape = int(np.array(env_action_space.shape).prod())
        self.full = False

        # optional fields used elsewhere
        self.max_steps = int(max_steps)
        self.current_trajectory_start = 0

        self.reset(save_rewards=False)

    @override
    def reset(self, save_rewards: bool = False):
        self.obs = np.zeros((self.capacity, self.num_tasks, self._obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.num_tasks, self._action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.num_tasks, self._obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.pos = 0
        self.full = False

        if save_rewards:
            self.org_rewards = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
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
        assert obs.ndim == 2 and action.ndim == 2 and reward.ndim <= 2 and done.ndim <= 2
        assert obs.shape[0] == action.shape[0] == reward.shape[0] == done.shape[0] == self.num_tasks

        self.obs[self.pos] = obs.copy()
        self.actions[self.pos] = action.copy()
        self.next_obs[self.pos] = next_obs.copy()
        self.dones[self.pos] = done.copy().reshape(-1, 1)
        self.rewards[self.pos] = reward.reshape(-1, 1).copy()

        self.pos += 1
        if self.pos >= self.capacity:
            self.full = True
        self.pos %= self.capacity

    def single_task_sample(self, task_idx: int, batch_size: int) -> ReplayBufferSamples:
        assert 0 <= task_idx < self.num_tasks, "Task index out of bounds."
        size = _valid_size(self.pos, self.full, self.capacity)
        sample_idx = _rand_indices(self._rng, size, batch_size)

        # FIX: correct indexing is [:, task_idx] after selecting sample_idx
        batch = (
            self.obs[sample_idx][:, task_idx],
            self.actions[sample_idx][:, task_idx],
            self.next_obs[sample_idx][:, task_idx],
            self.dones[sample_idx][:, task_idx],
            self.rewards[sample_idx][:, task_idx],
        )
        return ReplayBufferSamples(*batch)

    @override
    def sample(self, batch_size: int) -> ReplayBufferSamples:
        assert batch_size % self.num_tasks == 0, "batch_size must be divisible by num_tasks."
        single_task_batch_size = batch_size // self.num_tasks

        size = _valid_size(self.pos, self.full, self.capacity)
        sample_idx = _rand_indices(self._rng, size, single_task_batch_size)

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


# =========================
# Multi-task rollout buffer (unchanged logic)
# =========================
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
        self.num_rollout_steps = int(num_rollout_steps)
        self.num_tasks = int(num_tasks)
        self._rng = np.random.default_rng(seed)
        self._obs_shape = int(np.array(env_obs_space.shape).prod())
        self._action_shape = int(np.array(env_action_space.shape).prod())
        self._rnn_state_dim = rnn_state_dim
        self.dtype = dtype
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.num_rollout_steps, self.num_tasks, self._obs_shape), dtype=self.dtype)
        self.actions = np.zeros((self.num_rollout_steps, self.num_tasks, self._action_shape), dtype=self.dtype)
        self.rewards = np.zeros((self.num_rollout_steps, self.num_tasks, 1), dtype=self.dtype)
        self.dones = np.zeros((self.num_rollout_steps, self.num_tasks, 1), dtype=self.dtype)

        self.log_probs = np.zeros((self.num_rollout_steps, self.num_tasks, 1), dtype=self.dtype)
        self.values = np.zeros_like(self.rewards)
        self.means = np.zeros_like(self.actions)
        self.stds = np.zeros_like(self.actions)

        if self._rnn_state_dim is not None:
            self.rnn_states = np.zeros((self.num_rollout_steps, self.num_tasks, self._rnn_state_dim), dtype=self.dtype)

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
        assert obs.ndim == 2 and action.ndim == 2 and reward.ndim <= 2 and done.ndim <= 2
        assert obs.shape[0] == action.shape[0] == reward.shape[0] == done.shape[0] == self.num_tasks

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

    def get(self) -> Rollout:
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


# =========================
# Inter-task + PER multi-task buffer
# =========================
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
        # success episode buffer (IL buffer)
        use_success_episode_buffer: bool = True,
        success_total_capacity: int | None = None,
        # heuristic: force minimum task sampling probability (observation-only)
        force_task_idx: int | None = None,
        force_min_prob: float | None = None,
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

        self.use_inter_task_sampling = bool(use_inter_task_sampling)
        self.use_intra_task_sampling = bool(use_intra_task_sampling)

        self.prio_alpha = float(prio_alpha)
        self.prio_eps = float(prio_eps)
        self.td_priority = np.ones((self.capacity, self.num_tasks), dtype=np.float32)

        # success episode buffer (flat/global)
        self.use_success_episode_buffer = bool(use_success_episode_buffer)
        if success_total_capacity is None:
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
        self.force_task_idx = force_task_idx
        self.force_min_prob = None if force_min_prob is None else float(force_min_prob)

        # InterTaskMultiTaskReplayBuffer.__init__ 내부 (success buffer 초기화 이후에 추가)
        # ---- Shared overlap (LSH on obs) ----
        self.use_shared_overlap_il = False  # config flag (원하면 인자로 빼도 됨)

        # LSH params
        self.lsh_bits = 24                 # 16~32 추천. 24면 충돌/희소 밸런스 좋음.
        self.lsh_use_action = False        # obs만으로 먼저 가자 (action 포함은 다음 단계)
        self.lsh_seed = 0 if seed is None else int(seed)

        # random projection matrix: (obs_dim, bits)
        rng = np.random.default_rng(self.lsh_seed + 12345)
        self._lsh_W_obs = rng.standard_normal((self._obs_shape, self.lsh_bits)).astype(np.float32)

        # success transition -> hash code 저장
        self.suc_hash = np.zeros((self.suc_capacity,), dtype=np.uint64)

        # global hash -> task bitmask (MT50이면 uint64로 충분)
        # key: uint64 hash, value: uint64 bitmask of tasks that have this hash
        self._hash_task_mask: dict[int, int] = {}

    def _lsh_hash_obs(self, obs: np.ndarray) -> np.uint64:
        """obs: (obs_dim,) float32"""
        # proj: (bits,)
        proj = obs.astype(np.float32, copy=False) @ self._lsh_W_obs
        bits = (proj > 0).astype(np.uint8)  # 0/1

        # pack bits into uint64
        h = np.uint64(0)
        for i in range(self.lsh_bits):
            h |= (np.uint64(bits[i]) << np.uint64(i))
        return h

    @staticmethod
    def _popcount_u64(x: int) -> int:
        # python int bit_count works
        return int(x).bit_count()

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
        self.success_ema[mask] = (1 - tau) * self.success_ema[mask] + tau * task_success[mask]

    def add_success_episode(
        self,
        observations: np.ndarray,        # (T, obs_dim)
        next_observations: np.ndarray,   # (T, obs_dim)
        actions: np.ndarray,             # (T, act_dim)
        rewards: np.ndarray,             # (T,) or (T,1)
        dones: np.ndarray,               # (T,) bool/float
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

        for t in range(T):
            i = self.suc_pos
            self.suc_obs[i] = obs[t]
            self.suc_next_obs[i] = nxt[t]
            self.suc_actions[i] = act[t]
            self.suc_rewards[i] = np.array([rew[t]], dtype=np.float32)
            self.suc_dones[i] = np.array([dn[t]], dtype=np.float32)
            self.suc_task_idx[i] = int(task_idx)

            # ---- Shared overlap (LSH) : 반드시 여기 (루프 안) ----
            if self.use_shared_overlap_il:
                h = self._lsh_hash_obs(obs[t])
                self.suc_hash[i] = h
                key = int(h)
                prev = self._hash_task_mask.get(key, 0)
                self._hash_task_mask[key] = prev | (1 << int(task_idx))

            self.suc_pos += 1
            if self.suc_pos >= self.suc_capacity:
                self.suc_full = True
                self.suc_pos = 0
        

    def success_ready(self, min_size: int) -> bool:
        size = self.suc_capacity if self.suc_full else self.suc_pos
        return size >= min_size

    def sample_success(self, batch_size: int) -> ReplayBufferSamples:
        size = self.suc_capacity if self.suc_full else self.suc_pos
        idx = _rand_indices(self._rng, size, batch_size)

        batch = (
            self.suc_obs[idx],
            self.suc_actions[idx],
            self.suc_next_obs[idx],
            self.suc_dones[idx],
            self.suc_rewards[idx],
        )
        return ReplayBufferSamples(*batch)

    def _get_valid_size(self) -> int:
        return _valid_size(self.pos, self.full, self.capacity)

    def get_task_sampling_probs(self) -> np.ndarray:
        diff = (1.0 - self.success_ema).clip(0.0, 1.0)
        w = (diff + self.eps) ** self.task_beta
        p = w / (w.sum() + 1e-12)  # (num_tasks,)

        # ---- Heuristic (observation-only): enforce minimum probability for one task ----
        if self.force_task_idx is not None and self.force_min_prob > 0.0:
            t = int(self.force_task_idx)
            if 0 <= t < self.num_tasks:
                min_p = float(self.force_min_prob)

                # work in float64 for numerical stability
                p = p.astype(np.float64, copy=True)

                if p[t] < min_p:
                    other_sum = 1.0 - p[t]
                    if other_sum <= 1e-12:
                        # degenerate case: already all mass on t (or numerical issue)
                        p[:] = 0.0
                        p[t] = 1.0
                    else:
                        # scale down others proportionally to make room for min_p
                        scale = (1.0 - min_p) / other_sum
                        p *= scale
                        p[t] = min_p

                # re-normalize
                p = p / (p.sum() + 1e-12)

        return p.astype(np.float32)

    def _sample_tasks(self, n: int) -> npt.NDArray:
        return self._rng.choice(
            self.num_tasks, size=(n,), replace=True, p=self.get_task_sampling_probs()
        ).astype(np.int32)

    def _sample_indices_for_task(self, task_idx: int, n: int) -> np.ndarray:
        size = self._get_valid_size()
        if size <= 0:
            raise RuntimeError("Main buffer is empty.")

        if (not self.use_inter_task_sampling) or (not hasattr(self, "td_priority")):
            return _rand_indices(self._rng, size, n)

        pr = self.td_priority[:size, task_idx].astype(np.float64)
        pr = np.maximum(pr, self.prio_eps) ** self.prio_alpha
        s = pr.sum()
        if not np.isfinite(s) or s <= 0:
            return _rand_indices(self._rng, size, n)

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
    def sample(self, batch_size: int) -> ReplayBufferSamples | PrioritizedReplayBatch:
        size = self._get_valid_size()
        if size <= 0:
            raise RuntimeError("Main buffer is empty.")

        task_idx = self._sample_tasks(batch_size)          # (B,)
        buf_idx = np.empty((batch_size,), dtype=np.int32)  # (B,)

        unique_tasks, counts = np.unique(task_idx, return_counts=True)
        cursor = 0
        for t, c in zip(unique_tasks, counts):
            idxs = self._sample_indices_for_task(int(t), int(c))
            buf_idx[cursor:cursor + int(c)] = idxs
            cursor += int(c)

        perm = self._rng.permutation(batch_size)
        task_idx = task_idx[perm]
        buf_idx = buf_idx[perm]

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

    def sample_success_shared(
        self,
        batch_size: int,
        min_tasks: int = 2,          # "겹친다"의 최소 기준: 2개 task 이상에서 등장
        max_tries_mult: int = 50,    # rejection sampling 최대 시도 배수
    ) -> tuple[ReplayBufferSamples, np.ndarray, np.ndarray]:
        """
        Returns:
        samples: ReplayBufferSamples (obs, act, next_obs, done, reward)
        idx: (B,) indices into success buffer
        task_idx: (B,) task indices for each sample
        """
        size = self.suc_capacity if self.suc_full else self.suc_pos
        if size <= 0:
            raise RuntimeError("Success buffer is empty.")
        if not self.use_shared_overlap_il:
            raise RuntimeError("Shared overlap IL is disabled.")

        idxs = np.empty((batch_size,), dtype=np.int32)
        out_task = np.empty((batch_size,), dtype=np.int32)

        got = 0
        tries = 0
        max_tries = int(batch_size * max_tries_mult)

        while got < batch_size and tries < max_tries:
            tries += 1
            j = int(self._rng.integers(low=0, high=size, dtype=np.int32))
            h = int(self.suc_hash[j])
            mask = self._hash_task_mask.get(h, 0)
            if mask == 0:
                continue
            if self._popcount_u64(mask) < int(min_tasks):
                continue

            idxs[got] = j
            out_task[got] = int(self.suc_task_idx[j])
            got += 1

        if got < batch_size:
            # fallback: 남은 건 그냥 success에서 랜덤으로 채우기 (학습 중 early phase에서 발생 가능)
            rest = batch_size - got
            fill = _rand_indices(self._rng, size, rest)
            idxs[got:] = fill
            out_task[got:] = self.suc_task_idx[fill].astype(np.int32)

        batch = (
            self.suc_obs[idxs],
            self.suc_actions[idxs],
            self.suc_next_obs[idxs],
            self.suc_dones[idxs],
            self.suc_rewards[idxs],
        )
        return ReplayBufferSamples(*batch), idxs, out_task

# =========================
# IL-RL buffers
# =========================
@dataclass
class ILRLReplayBatch:
    rl: ReplayBufferSamples
    il: ReplayBufferSamples | None
    rl_indices: npt.NDArray        # (batch,)
    rl_task_indices: npt.NDArray   # (batch,)


class ILRLMultiTaskReplayBuffer(MultiTaskReplayBuffer):
    """
    Multi-task replay buffer with:
      1) task-level sampling biased by difficulty (1 - success_ema)
      2) transition-level prioritized sampling using stored td_priority
      3) separate IL buffer storing ONLY successful transitions (easy tasks)
    """

    def __init__(
        self,
        total_capacity: int,
        num_tasks: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
        max_steps: int = 500,

        task_beta: float = 1.0,
        prio_alpha: float = 0.6,
        prio_eps: float = 1e-6,

        success_ema_tau: float = 0.01,
        easy_success_th: float = 0.8,

        il_total_capacity: int | None = None,
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

        self.td_priority = np.ones((self.capacity, self.num_tasks), dtype=np.float32)
        self.success_ema = np.zeros((self.num_tasks,), dtype=np.float32)

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
        self.il_task_idx = np.zeros((self.il_capacity,), dtype=np.int32)

    def update_success_ema(self, task_success: npt.NDArray) -> None:
        task_success = np.asarray(task_success, dtype=np.float32).reshape(-1)
        assert task_success.shape[0] == self.num_tasks
        tau = self.success_ema_tau
        self.success_ema = (1 - tau) * self.success_ema + tau * task_success

    def add_il_transition(
        self,
        task_idx: int,
        obs: npt.NDArray,
        next_obs: npt.NDArray,
        action: npt.NDArray,
        reward: float,
        done: float,
    ) -> None:
        if task_idx < 0 or task_idx >= self.num_tasks:
            return
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
        idx = _rand_indices(self._rng, size, batch_size)

        batch = (
            self.il_obs[idx],
            self.il_actions[idx],
            self.il_next_obs[idx],
            self.il_dones[idx],
            self.il_rewards[idx],
        )
        return ReplayBufferSamples(*batch)

    def update_td_priorities(
        self,
        buffer_indices: npt.NDArray,
        task_indices: npt.NDArray,
        td_errors: npt.NDArray,
    ) -> None:
        buffer_indices = np.asarray(buffer_indices).reshape(-1)
        task_indices = np.asarray(task_indices).reshape(-1)
        td_errors = np.asarray(td_errors).reshape(-1)
        assert buffer_indices.shape == task_indices.shape == td_errors.shape

        pr = np.abs(td_errors).astype(np.float32) + self.prio_eps
        self.td_priority[buffer_indices, task_indices] = pr

    def _get_valid_size(self) -> int:
        return _valid_size(self.pos, self.full, self.capacity)

    def _sample_tasks(self, num_samples: int) -> npt.NDArray:
        diff = (1.0 - self.success_ema).clip(0.0, 1.0)
        w = (diff + 1e-6) ** self.task_beta
        p = w / (w.sum() + 1e-12)
        return self._rng.choice(self.num_tasks, size=(num_samples,), replace=True, p=p).astype(np.int32)

    def _sample_indices_for_task(self, task_idx: int, n: int) -> npt.NDArray:
        size = self._get_valid_size()
        if size <= 0:
            raise RuntimeError("Main buffer is empty.")

        pr = self.td_priority[:size, task_idx].astype(np.float64)
        pr = np.maximum(pr, self.prio_eps) ** self.prio_alpha
        pr_sum = pr.sum()
        if not np.isfinite(pr_sum) or pr_sum <= 0:
            return _rand_indices(self._rng, size, n, dtype=np.int32)

        p = pr / pr_sum
        return self._rng.choice(size, size=(n,), replace=True, p=p).astype(np.int32)

    def sample(self, batch_size: int, il_batch_size: int | None = None) -> ILRLReplayBatch:
        size = self._get_valid_size()
        if size <= 0:
            raise RuntimeError("Main buffer is empty.")

        task_idx = self._sample_tasks(batch_size)  # (B,)
        buf_idx = np.empty((batch_size,), dtype=np.int32)

        unique_tasks, counts = np.unique(task_idx, return_counts=True)
        cursor = 0
        for t, c in zip(unique_tasks, counts):
            idxs = self._sample_indices_for_task(int(t), int(c))
            buf_idx[cursor:cursor + int(c)] = idxs
            cursor += int(c)

        perm = self._rng.permutation(batch_size)
        task_idx = task_idx[perm]
        buf_idx = buf_idx[perm]

        obs = self.obs[buf_idx, task_idx]
        actions = self.actions[buf_idx, task_idx]
        next_obs = self.next_obs[buf_idx, task_idx]
        dones = self.dones[buf_idx, task_idx]
        rewards = self.rewards[buf_idx, task_idx]
        rl = ReplayBufferSamples(obs, actions, next_obs, dones, rewards)

        il = None
        if il_batch_size is not None and il_batch_size > 0 and self.il_ready(il_batch_size):
            il = self.sample_il(il_batch_size)

        return ILRLReplayBatch(
            rl=rl,
            il=il,
            rl_indices=buf_idx,
            rl_task_indices=task_idx.astype(np.int32),
        )