# MultiMCTS

MultiMCTS is a Python package that implements the [Monte Carlo Tree Search](https://wikipedia.org/wiki/Monte_Carlo_tree_search) algorithm for board games played by any number of players. With MultiMCTS, you can create AI for any board game *merely by knowing the rules* -- no strategy needed!

## Features

- Efficient (largely C-compiled) MCTS implementation
- Support for any number of players/teams
- Easily create AI for any board game

## Game Implementation

### For your game, you will need to:
- Represent the game state in code
- Identify all legal moves
- Determine if the game is over and who won

### You do NOT need to:
- Have domain knowledge
- Understand any strategy
- Evaluate the favorability of a game state

## Usage

To use MultiMCTS, you must define your game by subclassing `GameState` and implementing the required methods (see tictactoe.py for an example):
- get_current_team
- get_legal_moves
- make_move
- is_terminal
- get_reward

```python
from multimcts import MCTS, GameState

class MyGameState(GameState):
    # your implementation here...

mcts = MCTS(max_time=1)         # Create an MCTS agent. This one will spend 1 second searching for the best move.
state = MyGameState()           # Set up a new game.
while not state.is_terminal():  # Continue until the game is over.
    print(state)                # Print the current game state (implementing GameState.__repr__ might be helpful).
    state = mcts.search(state)  # Use the best move found so far.
print(state)                    # Print the final game state.
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
