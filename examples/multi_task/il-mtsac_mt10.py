# file: il-mtsac_mt10.py
from dataclasses import dataclass
from pathlib import Path

import tyro

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
)
from metaworld_algorithms.config.nn import VanillaNetworkConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import OffPolicyTrainingConfig
from metaworld_algorithms.envs import MetaworldConfig
from metaworld_algorithms.rl.algorithms import ILRLMTSACConfig
from metaworld_algorithms.run import Run

@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./run_results")
    resume: bool = False
    
    # IL-RL Hyperparameters
    il_update_per_step: int = 1
    lambda_max: float = 2.0
    s0: float = 0.0
    s1: float = 1.0
    top_p: float = 0.1
    task_beta: float = 1.0
    success_ema_tau: float = 0.01

def main() -> None:
    args = tyro.cli(Args)

    run = Run(
        run_name="mt10_mtsac_ilrl",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(
            env_id="MT10",
            terminate_on_success=False,
        ),
        algorithm=ILRLMTSACConfig(
            num_tasks=10,
            actor_config=ContinuousActionPolicyConfig(
                network_config=VanillaNetworkConfig(
                    optimizer=OptimizerConfig(max_grad_norm=None),
                )
            ),
            critic_config=QValueFunctionConfig(
                network_config=VanillaNetworkConfig(
                    optimizer=OptimizerConfig(max_grad_norm=None),
                )
            ),
            temperature_optimizer_config=OptimizerConfig(max_grad_norm=None),
            initial_temperature=1.0,
            num_critics=2,
            tau=0.005,
            use_task_weights=False,
            max_q_value=None,

            il_update_per_step=args.il_update_per_step,
            lambda_max=args.lambda_max,
            s0=args.s0,
            s1=args.s1,
            top_p=args.top_p,
            beta_task=args.task_beta,
            success_ema_tau=args.success_ema_tau,
        ),
        training_config=OffPolicyTrainingConfig(
            total_steps=int(2e7),
            buffer_size=int(1e6),
            batch_size=1280,
        ),
        checkpoint=True,
        resume=args.resume,
    )

    if args.track:
        assert args.wandb_project is not None and args.wandb_entity is not None
        run.enable_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=run,
            resume="allow",
        )

    run.start()


if __name__ == "__main__":
    main()
