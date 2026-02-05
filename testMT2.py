import gymnasium as gym

env = gym.make_vec(
    "Meta-World/MT10",
    vector_strategy="async",
    use_one_hot=True,
    num_goals=50,
)
obs, info = env.reset()
print(obs.shape)  # (num_envs, obs_dim)
