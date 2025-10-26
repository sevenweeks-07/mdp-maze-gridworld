# Reinforcement Learning Maze Tutorial

A simple reinforcement learning project that implements a custom Maze environment and demonstrates fundamental RL concepts.

## Overview

This project provides a hands-on introduction to Reinforcement Learning (RL) using a 5×5 grid-world maze. The agent (red circle) must navigate through walls to reach the goal (teal square).

## Project Structure

```
RL/
├── envs.py          # Custom Maze environment implementation
├── lesson.ipynb     # Tutorial notebook with RL concepts
└── README.md        # Project documentation
```

## Features

- **Custom Maze Environment**: Implements OpenAI Gym interface for easy integration
- **Configurable Settings**:
  - Exploring starts (random initial positions)
  - Shaped rewards (distance-based rewards vs sparse rewards)
- **Interactive Tutorial**: Jupyter notebook with mathematical explanations and visualizations

## Installation

### Prerequisites

```bash
python >= 3.10
```

### Dependencies

```bash
pip install gym numpy matplotlib pygame ipython
```

## Usage

### Running the Tutorial

1. Open the Jupyter notebook:
```bash
jupyter notebook maze_rl.ipynb
```

2. Run the cells to explore:
   - Environment basics (states, actions, rewards)
   - Trajectories and episodes
   - Discounted returns
   - Policy definitions

### Using the Maze Environment

```python
from envs import Maze

# Create environment
env = Maze()

# Reset to initial state
state = env.reset()

# Take an action
action = 2  # DOWN
next_state, reward, done, info = env.step(action)

# Render the environment
frame = env.render(mode='rgb_array')
```

## Environment Details

### State Space
- **Type**: Discrete 2D grid
- **Shape**: 5×5 (25 states)
- **Representation**: `(row, col)` tuple

### Action Space
- **Type**: Discrete
- **Actions**: 4
  - `0`: UP
  - `1`: RIGHT
  - `2`: DOWN
  - `3`: LEFT

### Rewards
- **Default**: `-1` for each step (encourages shorter paths)
- **Shaped** (optional): Distance-based rewards to the goal

### Episode Termination
- Episode ends when the agent reaches the goal state `(4, 4)`

## Key Concepts Covered

The tutorial notebook covers these fundamental RL concepts:

1. **States and Actions**: Understanding the observation and action spaces
2. **Trajectories**: Sequences of state-action-reward transitions
3. **Episodes**: Complete runs from start to goal
4. **Rewards**: Immediate feedback signals
5. **Returns**: Discounted cumulative rewards (G_t = Σ γ^k * r_{t+k})
6. **Policies**: Mappings from states to actions (π(a|s))

## Customization

### Creating a Custom Maze

Modify the `_create_maze()` method in `envs.py` to change the wall configuration:

```python
walls = [
    [(1, 0), (1, 1)],  # Add more wall segments
    # ... custom walls
]
```

### Adjusting Environment Parameters

```python
# Random starting positions
env = Maze(exploring_starts=True)

# Distance-based shaped rewards
env = Maze(shaped_rewards=True)

# Different grid size (requires code modification)
env = Maze(size=10)
```

## Learning Algorithms

This environment is compatible with various RL algorithms:
- **Monte Carlo Methods**
- **Temporal Difference Learning** (TD, SARSA, Q-Learning)
- **Policy Gradient Methods**
- **Deep RL** (DQN, A3C, PPO)

## Contributing

This is a side project for learning purposes. Feel free to:
- Report bugs or issues
- Suggest improvements
- Fork and experiment with the code

## Acknowledgments

- Environment rendering adapted from OpenAI Gym examples
- Video display function from DeepMind's dm_control tutorial
