# distutils: language=c++
# cython: language_level=3
# cython: profile=False

from time import time
from random import shuffle, choice
from typing import Union, Dict, List, Any

from libc.math cimport log, sqrt, INFINITY
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.pair cimport pair
cimport cython


Move = Any # MCTS does not care about the contents of a move; it merely passes the output of get_legal_moves() to make_move(), both of which are handled by the user.
Team = Union[int,str]
Rewards = Dict[Team,float]


class GameState:
    def get_current_team(self) -> Team:
        """The identifier of the current player's team."""
        raise NotImplementedError("GameState must implement get_current_team.")

    def get_legal_moves(self) -> List[Move]:
        """Returns a list of all legal moves from this state."""
        raise NotImplementedError("GameState must implement get_legal_moves.")

    def make_move(self, move:Move) -> 'GameState':
        """Returns a new GameState, which is the result of applying the given move to this state.
        Note: The current state (self) should NOT be modified. Rather, modify a copy of it.
        """
        raise NotImplementedError("GameState must implement make_move.")

    def is_terminal(self) -> bool:
        """Checks if the game is over."""
        raise NotImplementedError("GameState must implement is_terminal.")

    def get_reward(self) -> Union[float,Rewards]:
        """Returns the reward earned by the team that played the game-ending move (i.e. the team from the previous state).
        Typically 1 for win, -1 for loss, 0 for draw.
        Alternatively, returns a dict of teams/rewards: {team1:reward1, team2:reward2, ...}
        Note: This method is only called on terminal states.
        """
        raise NotImplementedError("GameState must implement get_reward.")


cdef class Node:
    """Represents a game state node in the MCTS search tree.

    Args:
        state (GameState): The game state at the current node.
        parent (Node): The parent of the current node in the search tree.
        move (Move): The move that was played from the parent node to get to this node.

    Attributes:
        children (list): The child nodes of the current node. These represent legal moves that have been visited.
        num_visits (int): The number of times the node has been visited.
        total_reward (dict): The total reward obtained from simulations through the node. Keys are teams; values are rewards.
        is_terminal (bool): Whether the node represents a terminal state.
        is_fully_expanded (bool): Whether all children of the node have been visited.
        remaining_moves (list): A list of moves that have not yet been tried.
    """
    cdef _state, _move
    cdef string _team
    cdef Node _parent
    cdef _children, _remaining_moves
    cdef map[string,float] _total_reward
    cdef int _num_visits
    cdef double log_visits
    cdef bint _is_terminal
    cdef bint _is_fully_expanded

    def __init__(self, state:GameState, parent:'Node'=None, move:Move=None):
        self._state = state
        self._parent = parent
        self._move = move
        self._team = str(self._state.get_current_team()).encode()

        self._children:List['Node'] = []
        self._num_visits = 0
        self.log_visits = -INFINITY
        self._total_reward = map[string,float]()

        self._is_terminal = self._state.is_terminal()
        if self._is_terminal:
            self._is_fully_expanded = True
            self._remaining_moves = []
        else:
            self._is_fully_expanded = False
            self._remaining_moves = self._state.get_legal_moves()
            shuffle(self._remaining_moves)#.do a c-shuffle? need memoryview?

    @property
    def state(self) -> GameState: return self._state
    @property
    def parent(self) -> 'Node': return self._parent
    @property
    def move(self) -> Move: return self._move
    @property
    def children(self) -> List['Node']: return self._children
    @property
    def remaining_moves(self) -> List[Move]: return self._remaining_moves

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def visit(self):
        self._num_visits += 1
        self.log_visits = log(self._num_visits)


cdef class MCTS:
    cdef double exploration_bias

    def __init__(self, exploration_bias:float=1.414):
        """Initializes an MCTS agent.

        Args:
            exploration_bias (float): The exploration bias, often denoted as C in the UCB formula.
                It determines the balance between exploration (choosing a move with uncertain outcome) and exploitation (choosing a move with known high reward).
                The default √2 is often used in practice. However, the optimal value depends on the game and is usually found by experimentation.
        """
        self.exploration_bias = exploration_bias

    @property
    def exploration_bias(self) -> float:
        return self.exploration_bias

    def search(self, state:GameState, *, max_time:Union[int,float]=None, max_iterations:int=None, heuristic=None, return_type:str="state") -> Union[GameState,Move,Node]:
        """Searches for this state's best move until some limit has been reached.

        Args:
            state (GameState): The game state for which to find the best move.
            max_time (int|float): The maximum time to search, in seconds.
            max_iterations (int): The maximum number of selections/simulations to perform.
            heuristic (callable): A function that takes a state and returns a move. See simulate() for more information.
            return_type (str): One of "state", "move", or "node".
        Returns:
            GameState: A new game state which is the result of applying the best move to the given state.
        """
        return_type = return_type.lower()
        VALID_RETURN_TYPES = {"state","move","node"}
        if return_type not in VALID_RETURN_TYPES:
            raise ValueError(f'Invalid return type: {return_type}, must be one of {VALID_RETURN_TYPES}')

        if max_time is None and max_iterations is None:
            raise ValueError("One or more of max_time/max_iterations is required.")

        node = Node(state)

        cdef double exploration_bias = self.exploration_bias

        cdef double end_time
        cdef int i
        if max_time is not None:
            end_time = time() + max_time
        if max_iterations is not None:
            i = max_iterations

        while True:
            child = self.select(node, exploration_bias)
            reward = self.simulate(child, heuristic=heuristic)
            self.backpropagate(child, reward)

            if max_time is not None and time() >= end_time:
                break
            if max_iterations is not None:
                i -= 1
                if i <= 0:
                    break

        best = self.get_best_child(node, exploration_bias)

        if return_type == "state":
            return best.state
        elif return_type == "move":
            return best.move
        elif return_type == "node":
            return best

    def select(self, node:Node, exploration_bias:float) -> Node:
        """Step 1: Selection
        Traverse the tree for the node we most want to simulate.
        Looks for an unexplored child of this node's best child's best child's...best child.
        """
        while not node._is_terminal:
            if not node._is_fully_expanded:
                return self.expand(node)
            else:
                node = self.get_best_child(node, exploration_bias)

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

        child = Node(state=node.state.make_move(move), parent=node, move=move)
        node.children.append(child)

        if len(node.remaining_moves) == 0:
            node._is_fully_expanded = True

        return child

    def simulate(self, node:Node, *, heuristic=None) -> Rewards:
        """Step 3: Simulation (aka playout/rollout)
        Play out a game, from the given node to termination, and return the final reward.
        A heuristic function may be used to guide the simulation. Otherwise, moves are chosen randomly.

        Args:
            node (Node): The node from which to begin the simulation.
            heuristic (callable): A function that takes a state and returns a move.
        """
        state = node.state

        if node._is_terminal:
            terminal_team = node.parent.state.get_current_team()
        else:
            while not state.is_terminal():
                terminal_team = state.get_current_team()
                if heuristic is not None:
                    move = heuristic(state)
                else:
                    move = choice(state.get_legal_moves())
                state = state.make_move(move)

        reward = state.get_reward()
        if not isinstance(reward, dict):
            reward = {terminal_team: reward}

        return reward

    @staticmethod
    def backpropagate(node:Node, reward:Rewards):
        """Step 4: Backpropagation
        Update all ancestors with the reward from this terminal node.
        """
        cdef double val
        cdef string ckey
        cdef map[string,float] creward = map[string,float]()
        for key,val in reward.items():
            if val == 0:
                continue
            ckey = str(key).encode()
            creward[ckey] = val

        cdef pair[string,float] item
        while node is not None:
            for item in creward:
                ckey = item.first
                if node._total_reward.count(ckey) == 0:
                    node._total_reward[ckey] = 0
                node._total_reward[ckey] += item.second
            node.visit()
            node = node.parent

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_best_child(self, node:Node, exploration_bias:float) -> Node:
        """Find the child with the highest Upper Confidence Bound (UCB) score.
        ucb = (x / n) + C * sqrt(ln(N) / n)
        x=reward for this node
        n=number of simulations for this node
        N=number of simulations for parent node
        C=exploration bias
        """
        # Initialize UCB variables.
        cdef int visits
        cdef double reward, ucb
        cdef double C = exploration_bias
        cdef double ln_parent_visits = node.log_visits

        cdef string cur_team = node._team

        cdef Node child, best_child
        cdef pair[string,float] item

        cdef double best_score = -INFINITY

        for child in node.children:
            visits = child._num_visits
            if visits == 0: # This should never happen but we're skipping div by 0 checks so just to be safe...
                continue

            # Relative reward is this child's reward minus its siblings' rewards.
            reward = 2 * child._total_reward[cur_team]
            for item in child._total_reward:
                reward -= item.second

            ucb = (reward / visits) + C * sqrt(ln_parent_visits / visits)

            if ucb > best_score:
                best_score = ucb
                best_child = child

        return best_child
