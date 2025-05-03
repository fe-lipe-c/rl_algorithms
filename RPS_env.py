from gymnasium.spaces import Box, Discrete, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class RockPaperScissors(MultiAgentEnv):
    """
    A simple Rock-Paper-Scissors environment for 2 agents.

    Both players always move simultaneously over a course of 10 timesteps in total.
    The winner of each timestep receives reward of +1, the losing player 1-.

    The observation of each player is the last opponent's action.

    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    LIZARD = 3
    SPOCK = 4

    WIN_MATRIX = {
            (ROCK,ROCK): (0, 0),
            (ROCK,PAPER): (-1, 1),
            (ROCK,SCISSORS): (1, -1),
            (ROCK,LIZARD): (1, -1),
            (ROCK,SPOCK): (-1, 1),
            (PAPER,ROCK): (1, -1),
            (PAPER,PAPER): (0, 0),
            (PAPER,SCISSORS): (-1, 1),
            (PAPER,LIZARD): (-1, 1),
            (PAPER,SPOCK): (1, -1),
            (SCISSORS,ROCK): (-1, 1),
            (SCISSORS,PAPER): (1, -1),
            (SCISSORS,SCISSORS): (0, 0),
            (SCISSORS,LIZARD): (1, -1),
            (SCISSORS,SPOCK): (-1, 1),
            (LIZARD,ROCK): (-1, 1),
            (LIZARD,PAPER): (1, -1),
            (LIZARD,SCISSORS): (-1, 1),
            (LIZARD,LIZARD): (0, 0),
            (LIZARD,SPOCK): (1, -1),
            (SPOCK,ROCK): (1, -1),
            (SPOCK,PAPER): (-1, 1),
            (SPOCK,SCISSORS): (1, -1),
            (SPOCK,LIZARD): (-1, 1),
            (SPOCK,SPOCK): (0, 0),
            }
    """
    def __init__(self, config=None):
    super().__init__()

    self.agents = self.possible_agents = ["player_0", "player_1"]

    # The observations are always the last taken actions. Hence observation
    # and action spaces are identical

    self.observation_spaces = self.action_spaces = {'player_1': Discrete(5), 'player_2': Discrete(5),}
