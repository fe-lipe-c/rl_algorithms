import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.annotations import OldAPIStack, override


class RockPaperScissors(MultiAgentEnv):
    """
    A simple Rock-Paper-Scissors environment for 2 agents.

    Both players always move simultaneously over a course of 10 timesteps in total.
    The winner of each timestep receives reward of +1, the losing player 1-.

    The observation of each player is the last opponent's action.
    """

    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    LIZARD = 3
    SPOCK = 4

    WIN_MATRIX = {
        (ROCK, ROCK): (0, 0),
        (ROCK, PAPER): (-1, 1),
        (ROCK, SCISSORS): (1, -1),
        (ROCK, LIZARD): (1, -1),
        (ROCK, SPOCK): (-1, 1),
        (PAPER, ROCK): (1, -1),
        (PAPER, PAPER): (0, 0),
        (PAPER, SCISSORS): (-1, 1),
        (PAPER, LIZARD): (-1, 1),
        (PAPER, SPOCK): (1, -1),
        (SCISSORS, ROCK): (-1, 1),
        (SCISSORS, PAPER): (1, -1),
        (SCISSORS, SCISSORS): (0, 0),
        (SCISSORS, LIZARD): (1, -1),
        (SCISSORS, SPOCK): (-1, 1),
        (LIZARD, ROCK): (-1, 1),
        (LIZARD, PAPER): (1, -1),
        (LIZARD, SCISSORS): (-1, 1),
        (LIZARD, LIZARD): (0, 0),
        (LIZARD, SPOCK): (1, -1),
        (SPOCK, ROCK): (1, -1),
        (SPOCK, PAPER): (-1, 1),
        (SPOCK, SCISSORS): (1, -1),
        (SPOCK, LIZARD): (-1, 1),
        (SPOCK, SPOCK): (0, 0),
    }

    def __init__(
        self,
        num_agents=2,
        max_steps=10,
        render_mode="human",
        seed=None,
    ):
        super().__init__()

        self._num_agents = num_agents

        self.agents = [f"player_{i}" for i in range(self.num_agents)]
        self.possible_agents = [f"player_{i}" for i in range(self.num_agents)]

        # The observations are always the last taken actions. Hence observation
        # and action spaces are identical

        self.observation_spaces = {
            "player_0": MultiDiscrete(nvec=[3, 3], start=[0, 0]),
            "player_1": MultiDiscrete(nvec=[3, 3], start=[0, 0]),
        }
        self.action_spaces = {
            "player_0": MultiDiscrete(nvec=[3, 3], start=[0, 0]),
            "player_1": MultiDiscrete(nvec=[3, 3], start=[0, 0]),
        }

        self.last_move = None
        self.num_moves = 0

        self.terminateds = set()
        self.truncateds = set()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # @override(MultiAgentEnv)
    def reset(self, *, seed=None, options=None):
        self.terminateds = set()
        self.truncateds = set()
        obs, infos = {"player_0": [0, 0], "player_1": [0, 0]}, {}

        self.num_moves = 0

        # Return starting observations
        return obs, infos

    # @override(MultiAgentEnv)
    def step(self, action_dict):
        self.num_moves += 1

        if len(action_dict) == 0:
            raise ValueError(
                "The environment is expecting an action for at least one agent."
            )

        move1 = action_dict["player_0"]
        move2 = action_dict["player_1"]

        observations = {
            "player_0": move2,
            "player_1": move1,
        }

        # Compute the rewards for each player based on the win-matrix
        r1, r2 = self.WIN_MATRIX[(move1[0], move2[0])]
        rewards = {"player_0": r1, "player_1": r2}

        # Terminate the entire episode for all agents, once 10 moves have been made.
        terminateds = {"__all__": self.num_moves >= 10}
        print(f"--------Step: {self.num_moves} -------------********\n")
        print(f"--------Actions: {move1} -------------********\n")
        print(f"--------Actions: {move2} -------------********\n")
        print(f"--------Actions type: {type(move2)} -------------********\n")
        print(f"--------observations: {observations} -------------********\n")

        return (
            observations,
            rewards,
            terminateds,
            {"player_0": False, "player_1": False},
            {"player_0": {}, "player_1": {}},
        )

    # @override(MultiAgentEnv)
    def render(self, mode="human"):
        pass


# dir(RockPaperScissors)
# teste = RockPaperScissors()

# teste.action_spaces

# obs, info = teste.reset()
# obs
# teste.step({"player_0": 3, "player_1": 2})

# # teste.step({"player_0": 0, "player_1": 1})
# teste.observe({"player_0": 0, "player_1": 1})
