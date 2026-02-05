
import gymnasium as gym
import numpy as np
import os

# Set headless backend
os.environ["MUJOCO_GL"] = "egl"
os.environ["EGL_LOG_LEVEL"] = "fatal"

def make_env(seed=1):
    return gym.make_vec(
        "Meta-World/MT10",
        seed=seed,
        use_one_hot=True,
        terminate_on_success=False,
        max_episode_steps=200,
        vector_strategy="async",
        num_goals=50,
        render_mode="rgb_array",
    )

print("--- Test 1: Repeated reset with sample_tasks ---")
envs = make_env(seed=1)
for i in range(5):
    obs, _ = envs.reset()
    task_idx = np.argmax(obs[0, -10:])
    print(f"Iter {i}: Task {task_idx}")
    envs.call("sample_tasks")

envs.close()

print("\n--- Test 2: Re-creating env with different seeds ---")
for s in range(1, 6):
    envs = make_env(seed=s)
    obs, _ = envs.reset()
    task_idx = np.argmax(obs[0, -10:])
    print(f"Seed {s}: Task {task_idx}")
    envs.close()
