# MultiMCTS

MultiMCTS is a Python package that implements the [Monte Carlo Tree Search](https://wikipedia.org/wiki/Monte_Carlo_tree_search) algorithm for board games played by any number of players. With MultiMCTS, you can create AI for any game *merely by knowing the rules* -- **no strategy needed!**


## Features

- Efficient (largely C-compiled) MCTS implementation
- Support for any number of players/teams
- Easily create AI for any board game


## Usage

### For your game, you will need to:
- Represent a game state in code
- Identify all legal moves
- Determine if the game is over and who won

### You do NOT need to:
- Understand strategy
- Have domain knowledge
- Evaluate the favorability of a game state

You can install with pip.
```bash
pip install multimcts
```

To use MultiMCTS, you must first define your game by subclassing `GameState` and implementing the required methods (see the [Tic-Tac-Toe](https://github.com/taylorvance/multimcts/blob/main/examples/tictactoe.py) example):
- `get_current_team` -- Returns the current team.
- `get_legal_moves` -- Returns a list of legal moves. Moves can be any data structure.
- `make_move` -- Returns a copy of the current state after performing the given move (one from get_legal_moves).
- `is_terminal` -- Returns whether the game is over.
- `get_reward` -- Returns the reward given to the team that played the game-ending move.

Then you can use MCTS to search for the best move. It will search until your defined limit is reached. The following shows how to simulate a game using MCTS:
```python
from multimcts import MCTS, GameState

class MyGameState(GameState):
    # your implementation here...
    pass

mcts = MCTS()                               # Create an MCTS agent.
state = MyGameState()                       # Set up a new game.

while not state.is_terminal():              # Continue until the game is over.
    print(state)                            # Print the current game state (implementing GameState.__repr__ might be helpful).
    state = mcts.search(state, max_time=1)  # Play the best move found after 1 second.

print(state)                                # Print the final game state.
```


### Development

1. Clone the MultiMCTS repo.
    ```bash
    git clone https://github.com/taylorvance/multimcts.git
    cd multimcts/
    ```
1. Make changes to `multimcts/mcts.pyx` or other files. Then build a distribution.
    ```bash
    python setup.py sdist
    ```
1. Install the updated package to use in your projects.
    ```bash
    pip install dist/multimcts-0.1.0.tar.gz
    ```
    Replace `multimcts-0.1.0` with the actual filename and version from the `dist/` directory.
