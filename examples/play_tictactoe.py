from multimcts import MCTS
from tictactoe import TicTacToeState


if __name__ == "__main__":
    mcts = MCTS(exploration_bias=0.43) # This bias was chosen empirically.

    state = TicTacToeState()
    print(state)

    while not state.is_terminal():
        if state.get_current_team() == 1:
            # Player's turn: Prompt for a move.
            moves = state.get_legal_moves()
            for i,move in enumerate(moves):
                print(f'{i+1}. {str(move)}')
            move = moves[int(input("> ")) - 1]
            state = state.make_move(move)
        else:
            # AI's turn: Use MCTS to find the best move.
            state = mcts.search(state, max_iterations=10000)

        print(state)
