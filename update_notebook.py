
import json
from pathlib import Path

notebook_path = Path("/home/mlic/kawon/251222_metaworld/metaworld-algorithms/examples/multi_task/rendering.ipynb")

with open(notebook_path, "r") as f:
    nb = json.load(f)

# 1. Restore agent_dir variable
# The target is the cell where we applied the checkpoint loading fix
found_restore = False
for cell in nb["cells"]:
    source = "".join(cell["source"])
    if "checkpoint_utils.construct_restore_args(agent)" in source:
        # Check if 'agent_dir =' is missing
        if "agent_dir =" not in source:
            print("Found checkpointer cell. Restoring agent_dir definition...")
            # Prepend agent_dir definition
            prefix_code = [
                "step_dir = CKPT_ROOT / str(STEP)\n",
                "agent_dir = step_dir / \"agent\"\n",
                "\n"
            ]
            cell["source"] = prefix_code + cell["source"]
            found_restore = True
        else:
            print("agent_dir already defined in checkpointer cell.")
            found_restore = True # Treat as found
        break

# 2. Append Rollout Cell
rollout_source = [
    "# Rollout and Render\n",
    "import imageio\n",
    "import numpy as np\n",
    "\n",
    "frames = []\n",
    "obs, info = envs.reset()\n",
    "print(\"Starting rollout...\")\n",
    "\n",
    "for i in range(200):  # max_episode_steps\n",
    "    rng = jax.random.PRNGKey(i)\n",
    "    # agent is the MTSAC instance restored\n",
    "    # Select action using the agent's updated params\n",
    "    # Note: 'agent.actor' is a TrainState, so agent.actor.params are the params\n",
    "    action, _ = agent.select_action(agent.actor.params, obs, rng, deterministic=True)\n",
    "    \n",
    "    # Environment step\n",
    "    obs, reward, terminated, truncated, info = envs.step(np.array(action))\n",
    "    \n",
    "    # Render\n",
    "    # Try rendering from vector env\n",
    "    try:\n",
    "        # Accessing the first env in the vector wrapper to render\n",
    "        frame = envs.envs[0].render()\n",
    "        frames.append(frame)\n",
    "    except Exception as e:\n",
    "        print(f\"Render failed at step {i}: {e}\")\n",
    "        break\n",
    "\n",
    "    if terminated.any() or truncated.any():\n",
    "        break\n",
    "\n",
    "print(f\"Rollout complete. Frames caught: {len(frames)}\")\n",
    "if frames:\n",
    "    imageio.mimsave(\"rollout.mp4\", frames, fps=30)\n",
    "    print(\"Video saved to rollout.mp4\")\n"
]

# Check if rollout cell already exists to avoid duplication
found_rollout = False
for cell in nb["cells"]:
    if "imageio.mimsave" in "".join(cell["source"]):
        print("Rollout logic already present. Updating it...")
        cell["source"] = rollout_source
        found_rollout = True
        break

if not found_rollout:
    print("Appending new rollout cell...")
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": rollout_source
    })

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
