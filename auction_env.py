import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class FirstPriceAuctionEnv(MultiAgentEnv):
    """
    A simple first-price auction environment for multiple agents.
    """

    def __init__(self, num_agents=2, obs_len=6, config=None):
        super().__init__()

        self.obs_len = obs_len

        self.num_agents = num_agents
        # Define all agent IDs that might even show up in your episodes.
        self.possible_agents = [f"bidder_{i}" for i in range(num_agents)]
        # bidders do not change throuhout the episode
        self.agents = self.possible_agents

        # Define the private values for each agent.
        self.private_values = [np.random.beta(10, 10) for _ in range(num_agents)]

        # Define the observation and action spaces for each agent.
        self.observation_spaces = {
            Box(low=0, high=1, shape=(self.obs_len,), dtype=np.float32)
            for _ in range(num_agents)
        }
        self.action_spaces = {
            MultiDiscrete(nvec=[100, 100], start=[1, 1]) for _ in range(num_agents)
        }

    def reset(self, *, seed=None, options=None):
        return {
            f"bidder_{i}": np.zeros(self.obs_len) for i in range(self.num_agents)
        }, {}

    def step(self, actions):
        for i, bid in enumerate(actions.values()):
            self.current_bids[i] = bid

        self.winning_bid = max(self.current_bids)
        self.winner = self.current_bids.index(self.winning_bid)

        rewards = {f"bidder_{i}": 0 for i in range(self.num_agents)}
        for i in range(self.num_agents):
            if i == self.winner:
                rewards[f"agent_{i}"] = self.private_values[i] - self.winning_bid

        done = True  # End the episode after one round of bidding

        return {f"bidder_{i}": 0 for i in range(self.num_agents)}, rewards, done, {}
