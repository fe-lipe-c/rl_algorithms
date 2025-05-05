"""
How to run this script
----------------------
`python RPS_run.py --enable-new-api-stack --sheldon_cooper_mode=False`

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`
"""

import argparse
import ray
from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import register_env
from RPS_env import RockPaperScissors

# parser = add_rllib_example_script_args(
#     default_iters=40,
#     default_reward=500,
#     default_timesteps=200_000,
# )

# parser.add_argument("--sheldon_cooper_mode", type=bool, default=False)
# parser.set_defaults(
#     enable_new_api_stack=True,
#     num_agents=2,
# )


def get_cli_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--framework", default="torch")
    parser.add_argument("--enable-new-api-stack", action="store_true", default=True)
    parser.add_argument("--as-test", action="store_true")
    parser.add_argument(
        "--num-gpus",
        type=float,
        default=1,
        help="Number of GPUs to use for the trainer process",
    )
    parser.add_argument(
        "--num-gpus-per-worker",
        type=float,
        default=0,
        help="Number of GPUs to use per worker",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # args = parser.parse_args()
    args = get_cli_args()

    assert args.num_agents == 2
    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"

    # def env_creator(env_config):
    #     return RockPaperScissors(sheldon_cooper_mode=args.sheldon_cooper_mode)

    try:
        ray.init(num_gpus=args.num_gpus + (args.num_gpus_per_worker * args.num_agents))
    except Exception as e:
        print(f"Warning: Error initializing Ray with GPUs: {e}")
        print("Falling back to CPU-only mode")
        ray.init()

    if args.framework == "torch" and args.num_gpus > 0:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # register_env("rock_paper_scissors_v10", env_creator)
    register_env(
        "rock_paper_scissors_v10",
        lambda config: RockPaperScissors(
            num_agents=max(2, config.get("num_agents", 2))
        ),
    )

    base_config = (
        PPOConfig()
        .environment("rock_paper_scissors_v10", env_config={"num_agents": 2})
        .framework(args.framework)
        .resources(num_gpus=args.num_gpus)
        .env_runners(
            num_gpus_per_env_runner=args.num_gpus_per_worker,
            sample_timeout_s=600,  # Increase timeout to 10 minutes
        )
        .multi_agent(
            policies=["learnable_policy", "random"],
            policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
        )
        .training(
            model={"free_log_std": False},
            gamma=0.99,
            lr=3e-4,
            # lr_schedule=[[0, 3e-4], [2000000, 5e-5]],  # Learning rate annealing
            lambda_=0.95,
            # kl_coeff=0.0,  # Start with 0 to rely on clipping
            # kl_target=0.01,
            clip_param=0.1,
            # vf_clip_param=10.0,
            entropy_coeff=0.0,  # Lower entropy for continuous control
            train_batch_size=1000000,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
        )
        .training(model={"custom_action_dist": "multi_categorical"})
    )

    if args.enable_new_api_stack:
        config = base_config.rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    p: RLModuleSpec() for p in ["learnable_policy", "random"]
                },
            ),
            model_config=DefaultModelConfig(
                # Configure the model for our auction environment
                fcnet_hiddens=[32, 32],
                fcnet_activation="tanh",
                vf_share_layers=False,  # Separate value function
            ),
        )

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            verbose=2,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
        ),
    )
    results = tuner.fit()
    ray.shutdown()

    # run_rllib_example_script_experiment(base_config, args)
