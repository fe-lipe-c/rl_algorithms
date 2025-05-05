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

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import register_env
from RPS_env import RockPaperScissors

parser = add_rllib_example_script_args(
    default_iters=40,
    default_reward=500,
    default_timesteps=200_000,
)

parser.add_argument("--sheldon_cooper_mode", type=bool, default=False)
parser.set_defaults(
    enable_new_api_stack=True,
    num_agents=2,
)

if __name__ == "__main__":
    args = parser.parse_args()

    assert args.num_agents == 2

    def env_creator(env_config):
        return RockPaperScissors(sheldon_cooper_mode=args.sheldon_cooper_mode)

    register_env("rock_paper_scissors_v10", env_creator)

    base_config = (
        PPOConfig()
        .environment("rock_paper_scissors_v10")
        .multi_agent(
            policies=["learnable_policy", "random"],
            policy_mapping_fn=lambda agent_id, *args, **kwargs: [
                "learnable_policy",
                "random",
            ][int(agent_id.split("_")[1]) % 2],
            policies_to_train=["learnable_policy"],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "learnable_policy": RLModuleSpec(),
                    "random": RLModuleSpec(),
                    # "random": RLModuleSpec(module_class=RandomRLModule),
                }
            ),
        )
    )

    run_rllib_example_script_experiment(base_config, args)
