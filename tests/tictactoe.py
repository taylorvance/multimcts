from random import shuffle
from multimcts import GameState

class TicTacToeState(GameState):
    def __init__(self, board=None, current_team=1):
        self.board = board or [[0,0,0], [0,0,0], [0,0,0]]
        self.current_team = current_team

    def get_current_team(self):
        return self.current_team

    def get_legal_moves(self, randomize=False):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    moves.append((i, j))
        if randomize:
            shuffle(moves)
        return iter(moves)

    def make_move(self, move):
        board = [list(row) for row in self.board]
        board[move[0]][move[1]] = self.current_team
        return TicTacToeState(board, -self.current_team)

    def is_terminal(self):
        for i in range(3):
            if abs(sum(self.board[i])) == 3 or abs(sum([row[i] for row in self.board])) == 3:
                return True
        if abs(sum(self.board[i][i] for i in range(3))) == 3 or abs(sum(self.board[i][2-i] for i in range(3))) == 3:
            return True
        if not self.has_move():
            return True
        return False

    def get_reward(self):
        win_sum = -self.current_team * 3
        for i in range(3):
            if sum(self.board[i]) == win_sum or sum(row[i] for row in self.board) == win_sum:
                return 1
        if sum(self.board[i][i] for i in range(3)) == win_sum or sum(self.board[i][2-i] for i in range(3)) == win_sum:
            return 1
        return 0

    def __hash__(self):
        return hash(str(self.board)+str(self.current_team))

    def __eq__(self, other):
        return self.board == other.board and self.current_team == other.current_team

    def __repr__(self):
        return '\n'.join(''.join({1:'X', -1:'O', 0:'-'}[val] for val in row) for row in self.board)
