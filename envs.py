"""
Custom Maze Environment for Reinforcement Learning
A 5x5 grid-world maze environment that follows the OpenAI Gym interface.
"""

from typing import Tuple, Dict, Optional, Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from IPython.display import HTML

import gym
from gym import spaces
from gym.error import DependencyNotInstalled

import pygame
from pygame import gfxdraw


class Maze(gym.Env):
    """
    A 5x5 grid-world maze environment with walls.

    The agent starts at position (0, 0) and must navigate to the goal at (4, 4).
    The maze contains walls that block certain movements.

    Parameters
    ----------
    exploring_starts : bool, optional (default=False)
        If True, the agent starts at a random position (excluding the goal).
        If False, the agent always starts at (0, 0).
    shaped_rewards : bool, optional (default=False)
        If True, rewards are shaped based on distance to goal.
        If False, rewards are -1 for each step until reaching the goal.
    size : int, optional (default=5)
        The size of the square maze grid.

    Attributes
    ----------
    action_space : gym.spaces.Discrete
        Discrete action space with 4 actions: {0: UP, 1: RIGHT, 2: DOWN, 3: LEFT}
    observation_space : gym.spaces.MultiDiscrete
        State space representing (row, col) positions in the grid.
    """

    def __init__(self, exploring_starts: bool = False,
                 shaped_rewards: bool = False, size: int = 5) -> None:
        super().__init__()
        self.exploring_starts = exploring_starts
        self.shaped_rewards = shaped_rewards
        self.state = (size - 1, size - 1)
        self.goal = (size - 1, size - 1)
        self.maze = self._create_maze(size=size)
        self.distances = self._compute_distances(self.goal, self.maze)
        self.action_space = spaces.Discrete(n=4)
        self.action_space.action_meanings = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: "LEFT"}
        self.observation_space = spaces.MultiDiscrete([size, size])

        self.screen = None
        self.agent_transform = None

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        Execute one step in the environment.

        Parameters
        ----------
        action : int
            Action to take: {0: UP, 1: RIGHT, 2: DOWN, 3: LEFT}

        Returns
        -------
        state : Tuple[int, int]
            The new state after taking the action
        reward : float
            Reward received for the transition
        done : bool
            Whether the episode has terminated (reached goal)
        info : Dict
            Additional information (empty dict)
        """
        reward = self.compute_reward(self.state, action)
        self.state = self._get_next_state(self.state, action)
        done = self.state == self.goal
        info = {}
        return self.state, reward, done, info

    def reset(self) -> Tuple[int, int]:
        """
        Reset the environment to start a new episode.

        Returns
        -------
        state : Tuple[int, int]
            The initial state of the new episode
        """
        if self.exploring_starts:
            while self.state == self.goal:
                self.state = tuple(self.observation_space.sample())
        else:
            self.state = (0, 0)
        return self.state

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.

        Parameters
        ----------
        mode : str, optional (default='human')
            Rendering mode: 'human' or 'rgb_array'

        Returns
        -------
        rgb_array : np.ndarray or None
            RGB array representation of the environment
        """
        assert mode in ['human', 'rgb_array']

        screen_size = 600
        scale = screen_size / 5

        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((screen_size, screen_size))

        surf = pygame.Surface((screen_size, screen_size))
        surf.fill((22, 36, 71))


        for row in range(5):
            for col in range(5):

                state = (row, col)
                for next_state in [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]:
                    if next_state not in self.maze[state]:

                        # Add the geometry of the edges and walls (i.e. the boundaries between
                        # adjacent squares that are not connected).
                        row_diff, col_diff = np.subtract(next_state, state)
                        left = (col + (col_diff > 0)) * scale - 2 * (col_diff != 0)
                        right = ((col + 1) - (col_diff < 0)) * scale + 2 * (col_diff != 0)
                        top = (5 - (row + (row_diff > 0))) * scale - 2 * (row_diff != 0)
                        bottom = (5 - ((row + 1) - (row_diff < 0))) * scale + 2 * (row_diff != 0)

                        gfxdraw.filled_polygon(surf, [(left, bottom), (left, top), (right, top), (right, bottom)], (255, 255, 255))

        # Add the geometry of the goal square to the viewer.
        left, right, top, bottom = scale * 4 + 10, scale * 5 - 10, scale - 10, 10
        gfxdraw.filled_polygon(surf, [(left, bottom), (left, top), (right, top), (right, bottom)], (40, 199, 172))

        # Add the geometry of the agent to the viewer.
        agent_row = int(screen_size - scale * (self.state[0] + .5))
        agent_col = int(scale * (self.state[1] + .5))
        gfxdraw.filled_circle(surf, agent_col, agent_row, int(scale * .6 / 2), (228, 63, 90))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def compute_reward(self, state: Tuple[int, int], action: int) -> float:
        """
        Compute the reward for taking an action from a state.

        Parameters
        ----------
        state : Tuple[int, int]
            Current state
        action : int
            Action to take

        Returns
        -------
        reward : float
            Reward value (either shaped or sparse)
        """
        next_state = self._get_next_state(state, action)
        if self.shaped_rewards:
            return - (self.distances[next_state] / self.distances.max())
        return - float(state != self.goal)

    def simulate_step(self, state: Tuple[int, int], action: int):
        """
        Simulate a step without changing the environment's internal state.
        Useful for planning algorithms.

        Parameters
        ----------
        state : Tuple[int, int]
            State to simulate from
        action : int
            Action to simulate

        Returns
        -------
        next_state : Tuple[int, int]
            Resulting state
        reward : float
            Reward for the transition
        done : bool
            Whether the episode would terminate
        info : Dict
            Additional information
        """
        reward = self.compute_reward(state, action)
        next_state = self._get_next_state(state, action)
        done = next_state == self.goal
        info = {}
        return next_state, reward, done, info

    def _get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        if action == 0:
            next_state = (state[0] - 1, state[1])
        elif action == 1:
            next_state = (state[0], state[1] + 1)
        elif action == 2:
            next_state = (state[0] + 1, state[1])
        elif action == 3:
            next_state = (state[0], state[1] - 1)
        else:
            raise ValueError("Action value not supported:", action)
        if next_state in self.maze[state]:
            return next_state
        return state

    @staticmethod
    def _create_maze(size: int) -> Dict[Tuple[int, int], Iterable[Tuple[int, int]]]:
        """
        Create the maze structure with walls.

        Returns a dictionary mapping each state to its valid neighboring states.
        """
        maze = {(row, col): [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
                for row in range(size) for col in range(size)}

        left_edges = [[(row, 0), (row, -1)] for row in range(size)]
        right_edges = [[(row, size - 1), (row, size)] for row in range(size)]
        upper_edges = [[(0, col), (-1, col)] for col in range(size)]
        lower_edges = [[(size - 1, col), (size, col)] for col in range(size)]
        walls = [
            [(1, 0), (1, 1)], [(2, 0), (2, 1)], [(3, 0), (3, 1)],
            [(1, 1), (1, 2)], [(2, 1), (2, 2)], [(3, 1), (3, 2)],
            [(3, 1), (4, 1)], [(0, 2), (1, 2)], [(1, 2), (1, 3)],
            [(2, 2), (3, 2)], [(2, 3), (3, 3)], [(2, 4), (3, 4)],
            [(4, 2), (4, 3)], [(1, 3), (1, 4)], [(2, 3), (2, 4)],
        ]

        obstacles = upper_edges + lower_edges + left_edges + right_edges + walls

        for src, dst in obstacles:
            maze[src].remove(dst)

            if dst in maze:
                maze[dst].remove(src)

        return maze

    @staticmethod
    def _compute_distances(goal: Tuple[int, int],
                           maze: Dict[Tuple[int, int], Iterable[Tuple[int, int]]]) -> np.ndarray:
        """
        Compute the shortest distance from each state to the goal using Dijkstra's algorithm.
        Used for reward shaping.
        """
        distances = np.full((5, 5), np.inf)
        visited = set()
        distances[goal] = 0.

        while visited != set(maze):
            sorted_dst = [(v // 5, v % 5) for v in distances.argsort(axis=None)]
            closest = next(x for x in sorted_dst if x not in visited)
            visited.add(closest)

            for neighbour in maze[closest]:
                distances[neighbour] = min(distances[neighbour], distances[closest] + 1)
        return distances


def display_video(frames):
    """
    Display a video from a sequence of frames in a Jupyter notebook.

    Parameters
    ----------
    frames : list of np.ndarray
        List of image frames to display as a video

    Returns
    -------
    HTML
        HTML5 video element for Jupyter notebook display

    Note
    ----
    Adapted from: https://colab.research.google.com/github/deepmind/dm_control/blob/master/tutorial.ipynb
    """
    orig_backend = plt.get_backend()
    plt.use('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.use(orig_backend)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
        im.set_data(frame)
        return [im]
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                    interval=50, blit=True, repeat=False)
    return HTML(anim.to_html5_video())