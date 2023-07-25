from multimcts import MCTS
from tictactoe import TicTacToeState


if __name__ == "__main__":
    mcts = MCTS(exploration_bias=0.43) # This bias was chosen empirically.

    state = TicTacToeState()
    print(state)

    while not state.is_terminal():
        state = mcts.search(state, max_iterations=10000)
        print(state)
