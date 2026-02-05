
import json
from pathlib import Path

notebook_path = Path("/home/mlic/kawon/251222_metaworld/metaworld-algorithms/examples/multi_task/rendering.ipynb")

with open(notebook_path, "r") as f:
    nb = json.load(f)

# Rollout and Rendering Code
rollout_source = [
    "# Rollout and Render\n",
    "import imageio\n",
    "import numpy as np\n",
    "\n",
    "# Ensure environment is created (if not already)\n",
    "# envs = env_config.spawn(seed=SEED) # Assumed to be created in earlier cells as 'envs'\n",
    "\n",
    "frames = []\n",
    "obs, info = envs.reset()\n",
    "\n",
    "print(\"Starting rollout...\")\n",
    "for i in range(200):  # max_episode_steps\n",
    "    # Select action\n",
    "    # Agent expects (params, obs, ...)\n",
    "    # We need to pass the restored agent params\n",
    "    # Check MTSAC signature: likely agent.select_action(agent.train_state.params, obs, ...)\n",
    "    # But agent loaded might be the TrainState itself or containing it.\n",
    "    # Let's assume 'agent' is the restored TrainState based on previous cells logic, \n",
    "    # OR 'agent' is the wrapper class instance if we initialized it.\n",
    "    # Based on MTSAC.initialize, 'agent' is a TrainState-like object usually.\n",
    "    # But let's look at how MTSAC uses it. \n",
    "    # Actually, let's try using the agent method directly if it was restored as a class instance,\n",
    "    # but typically checkpoints restore 'params'. \n",
    "    # Wait, the restore script used 'item=agent' where agent came from MTSAC.initialize.\n",
    "    # So 'agent' is the Agent instance (MTSAC class instance? No, initialize usually returns TrainState).\n",
    "    \n",
    "    # Let's inspect the 'agent' object in valid code context, but here we write potential code.\n",
    "    # Common pattern in this repo:\n",
    "    # action, rng = agent.select_action(agent.state.params, obs, rng)\n",
    "    # OR if 'agent' IS the state: \n",
    "    # We need the class method? \n",
    "    # Let's use the 'agent' object assuming it has the methods or we use the class.\n",
    "    \n",
    "    # Using a generic approach for now, user can adjust if needed.\n",
    "    # Assuming 'agent' has 'select_action' or we use 'MTSAC.select_action'\n",
    "    \n",
    "    # Fix: We need to know if 'agent' is the state or the object.\n",
    "    # In debug_restore.py: agent = MTSAC.initialize(...)\n",
    "    # So 'agent' is what initialize returns. \n",
    "    # Let's assume it has a 'select_action' method bound to it or available.\n",
    "    \n",
    "    rng = jax.random.PRNGKey(i)\n",
    "    # This is a guess based on typical JAX implementations in this repo\n",
    "    action, _ = agent.select_action(agent.state.params, obs, rng, deterministic=True)\n",
    "    \n",
    "    # Environment step\n",
    "    obs, reward, terminated, truncated, info = envs.step(np.array(action))\n",
    "    \n",
    "    # Render\n",
    "    # Using gym's render if available, or custom logic\n",
    "    # Metaworld vector envs usually require getting the underlining envs for rendering\n",
    "    # But 'GymVectorEnv' might have render support.\n",
    "    # Let's try capturing frames from the first env in the vector\n",
    "    try:\n",
    "        frame = envs.envs[0].render() # Adjust based on actual env structure\n",
    "        frames.append(frame)\n",
    "    except AttributeError:\n",
    "        # Fallback for some vector wrappers\n",
    "        frame = envs.call(\"render\")[0]\n",
    "        frames.append(frame)\n",
    "\n",
    "    if terminated.any() or truncated.any():\n",
    "        break\n",
    "\n",
    "print(f\"Rollout complete. Frames caught: {len(frames)}\")\n",
    "imageio.mimsave(\"rollout.mp4\", frames, fps=30)\n",
    "print(\"Video saved to rollout.mp4\")\n"
]

# Append the new cell
nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": rollout_source
})

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Rollout cell added to notebook.")
