
import json
from pathlib import Path

notebook_path = Path("/home/mlic/kawon/251222_metaworld/metaworld-algorithms/examples/multi_task/rendering.ipynb")

with open(notebook_path, "r") as f:
    nb = json.load(f)

# The corrected code including path definitions
corrected_code = [
    "step_dir = CKPT_ROOT / str(STEP)\n",
    "agent_dir = step_dir / \"agent\"\n",
    "\n",
    "from orbax.checkpoint import checkpoint_utils\n",
    "\n",
    "agent_ckptr = ocp.PyTreeCheckpointer()\n",
    "# Construct restore_args ensuring mesh/sharding is set to current available device (e.g. CPU)\n",
    "restore_args = checkpoint_utils.construct_restore_args(agent)\n",
    "agent = agent_ckptr.restore(str(agent_dir), item=agent, restore_args=restore_args)\n",
    "\n",
    "print(\"agent restored from:\", agent_dir)"
]

found = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        # Identify the cell we modified previously
        if "checkpoint_utils.construct_restore_args(agent)" in source:
            print("Found target cell. Restoring agent_dir definition...")
            cell["source"] = corrected_code
            found = True
            break

if found:
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Notebook fixed successfully: {notebook_path}")
else:
    print("Target cell not found! Please check the notebook manually.")
    exit(1)
