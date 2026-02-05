
import os
import sys
from pathlib import Path
import jax
import orbax.checkpoint as ocp
import metaworld_algorithms
from metaworld_algorithms.envs import MetaworldConfig
from metaworld_algorithms.rl.algorithms.mtsac import MTSAC, MTSACConfig
from metaworld_algorithms.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig
from metaworld_algorithms.config.nn import VanillaNetworkConfig
from metaworld_algorithms.config.optim import OptimizerConfig

# Setup paths
REPO_ROOT = Path("/home/mlic/kawon/251222_metaworld/metaworld-algorithms")
sys.path.insert(0, str(REPO_ROOT))

print("JAX Version:", jax.__version__)
print("JAX Devices:", jax.devices())
print("Orbax Version:", ocp.__version__)

# Configuration (copied from notebook)
SEED = 1
env_config = MetaworldConfig(
    env_id="MT10",
    use_one_hot=True,
    terminate_on_success=False,
    max_episode_steps=200,
    reward_func_version="v2",
    num_goals=50,
    reward_normalization_method=None,
    normalize_observations=False,
)

algo_config = MTSACConfig(
    num_tasks=10,
    gamma=0.99,
    actor_config=ContinuousActionPolicyConfig(
        network_config=VanillaNetworkConfig(
            optimizer=OptimizerConfig(max_grad_norm=1.0)
        )
    ),
    critic_config=QValueFunctionConfig(
        network_config=VanillaNetworkConfig(
            optimizer=OptimizerConfig(max_grad_norm=1.0),
        )
    ),
    num_critics=2,
    use_inter_task_sampling=True,
    use_intra_task_sampling=True,
    use_success_based_il=True,
    success_ema_tau=0.01,
    il_weight_mode="sigmoid",
    il_weight_temp=0.1,
    il_weight_power=2.0,
    il_loss_type="mse",
    il_coef=1.0,
    il_qfilter_top_p=0.2,
    il_qfilter_min_good=8,
)

# Initialize Agent
print("Initializing Agent...")
agent = MTSAC.initialize(algo_config, env_config, seed=SEED)
print("Agent initialized.")

# Checkpoint Path
CKPT_ROOT = Path("/home/mlic/kawon/251222_metaworld/metaworld-algorithms/examples/multi_task/run_results8/mt10_custom_mtsac_1/checkpoints")
STEP = 9_599_990
agent_dir = CKPT_ROOT / str(STEP) / "agent"

print(f"Restoring from: {agent_dir}")

# Attempt Restore
try:
    agent_ckptr = ocp.PyTreeCheckpointer()
    
    # Try basic restore first
    print("Attempting basic restore with item=agent...")
    restored_agent = agent_ckptr.restore(str(agent_dir), item=agent)
    print("Success!")
    
except Exception as e:
    print(f"Basic restore failed: {e}")
    print("Attempting restore with explicit restore_args for sharding...")
    
    # Construct restore_args ensuring mesh/sharding is set to current available device
    from orbax.checkpoint import checkpoint_utils
    
    # Create restore_args from the target agent
    # We want to force restoration to match the 'agent' structure and sharding (which is likely SingleDevice on CPU now)
    restore_args = checkpoint_utils.construct_restore_args(agent)
    
    try:
        restored_agent = agent_ckptr.restore(str(agent_dir), item=agent, restore_args=restore_args)
        print("Success with explicit restore_args!")
    except Exception as e2:
        print(f"Restore with args failed: {e2}")
        import traceback
        traceback.print_exc()

