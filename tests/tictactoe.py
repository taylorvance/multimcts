from multimcts import GameState

class TicTacToeState(GameState):
    def __init__(self, board=None, current_team=1):
        self.board = board or [[0,0,0], [0,0,0], [0,0,0]]
        self.current_team = current_team

    def get_current_team(self):
        return self.current_team

    def get_legal_moves(self):
        return [(i,j) for i in range(3) for j in range(3) if self.board[i][j] == 0]

    def make_move(self, move):
        board = [list(row) for row in self.board]
        board[move[0]][move[1]] = self.current_team
        return TicTacToeState(board, -self.current_team)

    def is_terminal(self):
        # Check rows and columns.
        for i in range(3):
            if abs(sum(self.board[i])) == 3 or abs(sum([row[i] for row in self.board])) == 3:
                return True
        # Check diagonals.
        if abs(sum(self.board[i][i] for i in range(3))) == 3 or abs(sum(self.board[i][2-i] for i in range(3))) == 3:
            return True
        # Check for draw.
        if all(self.board[i][j] != 0 for i in range(3) for j in range(3)):
            return True
        return False

    def get_reward(self):
        win_sum = -self.current_team * 3
        # Check rows and columns.
        for i in range(3):
            if sum(self.board[i]) == win_sum or sum(row[i] for row in self.board) == win_sum:
                return 1
        # Check diagonals.
        if sum(self.board[i][i] for i in range(3)) == win_sum or sum(self.board[i][2-i] for i in range(3)) == win_sum:
            return 1
        # Cat's game.
        return 0

    def __repr__(self):
        return '\n'.join(''.join({1:'X', -1:'O', 0:'-'}[val] for val in row) for row in self.board)
