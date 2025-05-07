from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.examples.envs.classes.multi_agent import MultiAgentCartPole
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import register_env


parser = add_rllib_example_script_args(
    default_iters=10, default_reward=500.0, default_timesteps=200000
)
parser.set_defaults(
    enable_new_api_stack=True,
    num_agents=2,
)


if __name__ == "__main__":
    args = parser.parse_args()

    assert args.num_agents == 2, "Must set --num-agents=2 when running this script!"

    # Simple environment with n independent cartpole entities.
    register_env(
        "multi_agent_cartpole",
        lambda _: MultiAgentCartPole({"num_agents": args.num_agents}),
    )

    base_config = (
        PPOConfig()
        .environment("multi_agent_cartpole")
        .multi_agent(
            policies={"learnable_policy", "random"},
            # Map to either random behavior or PPO learning behavior based on
            # the agent's ID.
            policy_mapping_fn=lambda agent_id, *args, **kwargs: [
                "learnable_policy",
                "random",
            ][agent_id % 2],
            # We need to specify this here, b/c the `forward_train` method of
            # `RandomRLModule` (ModuleID="random") throws a not-implemented error.
            policies_to_train=["learnable_policy"],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "learnable_policy": RLModuleSpec(),
                    "random": RLModuleSpec(module_class=RandomRLModule),
                }
            ),
        )
    )

    run_rllib_example_script_experiment(base_config, args)
