from time import time
from random import shuffle, choice
from collections import defaultdict
from typing import Union, Any

from libc.math cimport log, sqrt, INFINITY


class GameState():
    def get_current_team(self) -> Union[int,str]:
        """The identifier of the current player's team."""
        raise NotImplementedError("GameState must implement get_current_team.")

    def get_legal_moves(self) -> list[Any]:
        """Returns a list of all legal moves from this state."""
        raise NotImplementedError("GameState must implement get_legal_moves.")

    def make_move(self, move:Any) -> 'GameState':
        """Returns a new GameState, which is the result of applying the given move to this state.
        Note: The current state (self) should NOT be modified. Rather, modify a copy of it.
        """
        raise NotImplementedError("GameState must implement make_move.")

    def is_terminal(self) -> bool:
        """Checks if the game is over."""
        raise NotImplementedError("GameState must implement is_terminal.")

    def get_reward(self) -> Union[float, dict[Union[int,str], float]]:
        """Returns the reward earned by the team that played the game-ending move (i.e. the team from the previous state).
        Typically 1 for win, -1 for loss, 0 for draw.
        Alternatively, returns a dict of teams/rewards: {team1:reward1, team2:reward2, ...}
        Note: This method is only called on terminal states.
        """
        raise NotImplementedError("GameState must implement get_reward.")

    def pick_move(self) -> Any:
        """Returns a legal move from this state, used during the MCTS simulation step.
        You may override this method with a heuristic for guiding simulations. Otherwise, games are played out randomly.
        """
        return choice(self.get_legal_moves())


class Node():
    """Represents a game state node in the MCTS search tree.

    Args:
        state (GameState): The game state at the current node.
        parent (Node): The parent of the current node in the search tree.

    Attributes:
        children (list): The child nodes of the current node. These represent legal moves that have been visited.
        num_visits (int): The number of times the node has been visited.
        total_reward (dict): The total reward obtained from simulations through the node. Keys are teams; values are rewards.
        is_terminal (bool): Whether the node represents a terminal state.
        is_fully_expanded (bool): Whether all children of the node have been visited.
        remaining_moves (list): A list of moves that have not yet been tried.
    """
    def __init__(self, state:GameState, parent:'Node'=None):
        self.state = state
        self.parent = parent

        self.children:list['Node'] = []
        self.num_visits:int = 0
        self.total_reward = defaultdict(float)

        self.is_terminal:bool = self.state.is_terminal()
        self.is_fully_expanded:bool = self.is_terminal

        if self.is_fully_expanded:
            self.remaining_moves = []
        else:
            self.remaining_moves = self.state.get_legal_moves()
            shuffle(self.remaining_moves)


class MCTS():
    def __init__(self, exploration_bias:float=1.414):
        """Initializes an MCTS agent.

        Args:
            exploration_bias (float): The exploration bias, often denoted as C in the UCB formula.
                It determines the balance between exploration (choosing a move with uncertain outcome) and exploitation (choosing a move with known high reward).
                Default is 1.414, which is sqrt(2) and often used in practice.
        """
        self.exploration_bias = exploration_bias

    def search(self, state:GameState, *, max_time:Union[int,float]=None, max_iterations:int=None) -> GameState:
        """Searches for this state's best move until some limit has been reached.

        Args:
            state (GameState): The game state for which to find the best move.
            max_time (int|float): The maximum time to search, in seconds.
            max_iterations (int): The maximum number of selections/simulations to perform.
        Returns:
            GameState: A new game state which is the result of applying the best move to the given state.
        """
        if max_time is None and max_iterations is None:
            raise ValueError("One or more of max_time/max_iterations is required.")

        node = Node(state)

        cdef double end_time
        cdef int i
        if max_time is not None:
            end_time = time() + max_time
        if max_iterations is not None:
            i = max_iterations

        while True:
            child = self.select(node)
            reward = self.simulate(child)
            self.backpropagate(child, reward)

            if max_time is not None and time() >= end_time:
                break
            if max_iterations is not None:
                i -= 1
                if i <= 0:
                    break

        return self.get_best_child(node).state

    def select(self, node:Node) -> Node:
        """Step 1: Selection
        Traverse the tree for the node we most want to simulate.
        Looks for an unexplored child of this node's best child's best child's...best child.
        """
        while not node.is_terminal:
            if not node.is_fully_expanded:
                return self.expand(node)
            else:
                node = self.get_best_child(node)

        return node

    @staticmethod
    def expand(node:Node) -> Node:
        """Step 2: Expansion
        Add a new child to this node.
        """
        try:
            move = node.remaining_moves.pop()
        except IndexError:
            raise IndexError("Tried to expand a node with no remaining moves.")

        child = Node(node.state.make_move(move), node)
        node.children.append(child)

        if len(node.remaining_moves) == 0:
            node.is_fully_expanded = True

        return child

    def simulate(self, node:Node) -> dict:
        """Step 3: Simulation (aka playout, rollout)
        Play out a random game, from this node to termination, and return the final reward.
        """
        state = node.state

        if node.is_terminal:
            terminal_team = node.parent.state.get_current_team()
        else:
            while not state.is_terminal():
                terminal_team = state.get_current_team()
                move = state.pick_move()
                state = state.make_move(move)

        reward = state.get_reward()
        if not isinstance(reward, dict):
            reward = {terminal_team: reward}

        return reward

    @staticmethod
    def backpropagate(node:Node, reward:dict):
        """Step 4: Backpropagation
        Update all ancestors with the reward from this terminal node.
        """
        # Remove 0-values for efficiency
        reward = {k:v for k,v in reward.items() if v!=0}

        while node is not None:
            node.num_visits += 1

            for key in reward:
                node.total_reward[key] += reward[key]

            node = node.parent

    def get_best_child(self, node:Node) -> Node:
        cur_team = node.state.get_current_team()

        # Initialize UCB variables.
        cdef int visits
        cdef double reward, score
        cdef double exploration_bias = self.exploration_bias
        cdef double ln_parent_visits = log(node.num_visits)

        cdef double max_score = -INFINITY
        best:Node = None

        for child in node.children:
            # Relative reward is this child's reward minus its siblings' rewards.
            reward = (2 * child.total_reward[cur_team]) - sum(x for x in child.total_reward.values())
            visits = child.num_visits

            """UCB := (x / n) + C * sqrt(ln(N) / n)
            x=reward for this node
            n=number of simulations for this node
            N=number of simulations for parent node
            C=exploration bias
            """
            score = (reward / visits) + exploration_bias * sqrt(ln_parent_visits / visits)

            if score > max_score:
                max_score = score
                best = child

        return best
