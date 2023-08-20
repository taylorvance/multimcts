# distutils: language=c++
# cython: language_level=3
# cython: profile=False

from time import time
from random import shuffle, choice
from typing import Union, Dict, List, Tuple, Hashable

cimport cython
from libc.math cimport log, sqrt, INFINITY
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.pair cimport pair


Move = Hashable # Moves can be any type, as long as they can be added to a set and used as dict keys.
Team = Union[int,str]
Rewards = Dict[Team,float]


class GameState:
    def get_current_team(self) -> Team:
        """The identifier of the current player's team."""
        raise NotImplementedError("GameState must implement get_current_team.")

    def get_legal_moves(self) -> List[Move]:
        """List of all legal moves from this state."""
        raise NotImplementedError("GameState must implement get_legal_moves.")

    def make_move(self, move:Move) -> GameState:
        """A new GameState--the result of applying the given move to this state.
        Note: The current state (self) should NOT be modified. Rather, modify a copy of it.
        """
        raise NotImplementedError("GameState must implement make_move.")

    def is_terminal(self) -> bool:
        """Is the game over?"""
        raise NotImplementedError("GameState must implement is_terminal.")

    def get_reward(self) -> Union[float,Rewards]:
        """The reward earned by the team that played the game-ending move (i.e. the team from the previous state).
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
        children (Dict[Move,Node]): The child nodes of the current node. These represent legal moves that have been visited.
        visits (int): The number of times the node has been visited.
        rewards (dict): All teams' rewards obtained from simulations through the node. Keys are teams; values are rewards.
        is_terminal (bool): Whether the node represents a terminal state.
        is_fully_expanded (bool): Whether all children of the node have been visited.
        remaining_moves (list): A list of moves that have not yet been tried.
    """
    cdef state, move, children, remaining_moves
    cdef Node parent
    cdef string team
    cdef map[string,double] rewards, rave_rewards
    cdef int visits, rave_visits
    cdef bint is_terminal, is_fully_expanded
    cdef double sqrtlog_visits, invsqrt_visits, avg_reward, avg_rave_reward

    def __init__(self, state:GameState, parent:Node=None, move:Move=None):
        self.state = state
        self.parent = parent
        self.move = move

        self.children:Dict[Move,Node] = {}
        self.visits = 0
        self.rave_visits = 0
        self.rewards = map[string,double]()
        self.rave_rewards = map[string,double]()
        self.is_terminal = self.state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        if self.is_fully_expanded:
            self.remaining_moves = []
        else:
            self.remaining_moves = self.state.get_legal_moves()
            shuffle(self.remaining_moves)

        # The following are cached for performance.
        self.team = str(self.state.get_current_team()).encode()
        if self.parent is not None:
            self.rewards[self.parent.team] = 0
            self.rave_rewards[self.parent.team] = 0
        self.sqrtlog_visits = 0
        self.invsqrt_visits = 0
        self.avg_reward = 0
        self.avg_rave_reward = 0

    def get_state(self) -> GameState: return self.state
    def get_team(self) -> str: return self.team.decode()
    def get_move(self) -> Move: return self.move

    def get_parent(self) -> Node: return self.parent
    def get_children(self) -> Dict[Move,Node]: return self.children

    def get_visits(self) -> int: return self.visits
    def get_rewards(self) -> dict: return {item.first.decode():item.second for item in self.rewards}
    def get_avg_reward(self) -> float: return self.avg_reward

    def get_rave_visits(self) -> int: return self.rave_visits
    def get_rave_rewards(self) -> dict: return {item.first.decode():item.second for item in self.rave_rewards}
    def get_avg_rave_reward(self) -> float: return self.avg_rave_reward

    @cython.cdivision(True)
    cdef void visit(self, map[string,double] crewards, set moves):
        """Update this node's visits and rewards, and cache some variables for more efficient UCB calculation."""
        self.visits += 1

        self.sqrtlog_visits = sqrtlog(self.visits)
        self.invsqrt_visits = invsqrt(self.visits)

        # Update regular rewards.
        cdef pair[string,double] item
        cdef double total_reward = 0
        for item in crewards:
            if self.rewards.count(item.first) == 0:
                self.rewards[item.first] = 0
            self.rewards[item.first] += item.second
            total_reward += self.rewards[item.first]
        if self.parent is not None:
            # Average reward: (reward for my parent's team - rewards for all other teams) / num visits to this node
            self.avg_reward = ((2*self.rewards[self.parent.team]) - total_reward) / self.visits

        # Update RAVE rewards.
        if len(moves) == 0:
            return
        cdef int rave_visits = 0
        cdef double total_rave_reward = 0
        for move in list(self.children.keys()) + self.remaining_moves:
            if move in moves:
                rave_visits += 1
        if rave_visits > 0:
            self.rave_visits += rave_visits
            for item in crewards:
                if self.rave_rewards.count(item.first) == 0:
                    self.rave_rewards[item.first] = 0
                self.rave_rewards[item.first] += rave_visits * item.second
                total_rave_reward += self.rave_rewards[item.first]
            if self.parent is not None:
                self.avg_rave_reward = ((2*self.rave_rewards[self.parent.team]) - total_rave_reward) / self.rave_visits

    cpdef double ucb(self, double exploration_bias):
        """Upper Confidence Bound
        ucb = (x / n) + C * sqrt(ln(N) / n)
        x=reward for this node
        n=number of simulations for this node
        N=number of simulations for parent node
        C=exploration bias
        """
        return self.avg_reward + self.uncertainty(exploration_bias)

    cpdef double uncertainty(self, double exploration_bias):
        return exploration_bias * self.parent.sqrtlog_visits * self.invsqrt_visits

    @cython.cdivision(True)
    cpdef double rave_ratio(self, double bias):
        """Determines the relative weight of RAVE rewards in the final score calculation.
        As this node is visited more times, the RAVE effect diminishes.
        Note: The number of RAVE visits has little bearing on this value (the function's curve looks the same for all large values of rave_visits).
        https://www.desmos.com/calculator/drlccftt6a
        """
        if self.rave_visits == 0 or bias == 0:
            return 0
        elif self.visits == 0:
            return 1
        cdef int v = self.visits, rv = self.rave_visits
        return rv / (v + rv + 4*v*rv/(bias*bias))

    cpdef double score(self, double exploration_bias, double rave_bias):
        cdef double rave_ratio = self.rave_ratio(rave_bias)
        return (rave_ratio * self.avg_rave_reward) + ((1-rave_ratio) * self.avg_reward) + self.uncertainty(exploration_bias)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef Node best_child(self, double exploration_bias, double rave_bias):
        cdef Node child, best_child = next(iter(self.children.values()))
        cdef double score, best_score = -INFINITY
        for move, child in self.children.items():
            score = child.score(exploration_bias, rave_bias)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    cdef Node select(self, double exploration_bias, double rave_bias):
        cdef Node node = self
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = node.best_child(exploration_bias, rave_bias)
            else:
                return node.expand()
        return node

    cdef Node expand(self):
        move = self.remaining_moves.pop()
        if len(self.remaining_moves) == 0:
            self.is_fully_expanded = True
        cdef Node child = Node(state=self.state.make_move(move), parent=self, move=move)
        self.children[move] = child
        return child

    cdef Tuple[map[string,double], set] simulate(self):
        cdef set moves = set()
        state = self.state

        if self.is_terminal:
            # If terminal, no simulation needed. Just get the reward.
            reward = state.get_reward()
            if not isinstance(reward, dict):
                reward = {self.parent.state.get_current_team(): reward}
        elif callable(getattr(state, 'simulate', False)):
            # Else if this GameState has its own simulate method (aka rollout policy), use it.
            (reward, moves) = state.simulate()
        else:
            # Else, use a random rollout policy.
            while not state.is_terminal():
                prev_state = state
                move = choice(state.get_legal_moves())
                moves.add(move)
                state = state.make_move(move)
            reward = state.get_reward()
            if not isinstance(reward, dict):
                reward = {prev_state.get_current_team(): reward}

        # Convert rewards from Python dict to C map.
        cdef map[string,double] crewards = map[string,double]()
        for team in reward:
            if reward[team] != 0:
                crewards[str(team).encode()] = float(reward[team])

        return (crewards, moves)

    cdef void backpropagate(self, map[string,double] crewards, set moves):
        self.visit(crewards, moves)
        if self.parent is not None:
            self.parent.backpropagate(crewards, moves)

    cdef void execute_round(self, double exploration_bias, double rave_bias):
        """Step 1,2: Selection,Expansion"""
        cdef Node node = self.select(exploration_bias, rave_bias)

        """Step 3: Simulation"""
        cdef map[string,double] crewards
        cdef set moves
        (crewards, moves) = node.simulate()

        """Step 4: Backpropagation"""
        if rave_bias == 0:
            moves = set()
        node.backpropagate(crewards, moves)


cdef class MCTS:
    cdef double _exploration_bias, _rave_bias

    def __init__(self, exploration_bias:float=1.414, rave_bias:float=0):
        """Initializes an MCTS agent.
        Args:
            exploration_bias (float): The exploration bias, which balances exploration (favoring untested moves) and exploitation (favoring good moves).
                The default √2 is often used in practice. However, the optimal value depends on the game and is usually found by experimentation.
            rave_bias (float): The RAVE bias, which balances RAVE rewards (from unchosen but simulated moves) with regular rewards (from chosen moves).
        """
        self._exploration_bias = exploration_bias
        self._rave_bias = rave_bias

    @property
    def exploration_bias(self) -> float: return self._exploration_bias
    @property
    def rave_bias(self) -> float: return self._rave_bias

    def search(self, state:GameState, *, max_iterations:int=None, max_time:Union[int,float]=None, return_type:str="state") -> Union[GameState,Move,Node]:
        """Searches for this state's best move until some limit has been reached.
        Args:
            state (GameState): The game state for which to find the best move.
            max_iterations (int): The maximum number of simulations to perform.
            max_time (int|float): The maximum time to search, in seconds.
            return_type (str): One of "state", "move", or "node".
        Returns:
            GameState: A new game state which is the result of applying the best move to the given state.
        """
        if max_iterations is None and max_time is None:
            raise ValueError('At least one of [max_iterations,max_time] is required.')
        if return_type not in {'state','move','node'}:
            raise ValueError(f'Invalid return_type "{return_type}" must be one of {{"state","move","node"}}')

        # Initialize stopping conditions.
        cdef int end_iteration
        if max_iterations is not None:
            end_iteration = max_iterations
        cdef double end_time
        if max_time is not None:
            end_time = time() + max_time

        cdef Node node = Node(state)
        cdef int i = 0
        while True:
            node.execute_round(self._exploration_bias, self._rave_bias)

            # Check stopping conditions.
            i += 1
            if max_iterations is not None:
                if end_iteration <= i:
                    break
            if max_time is not None:
                if end_time <= time():
                    break

        # We've performed the full search above using the desired biases.
        # For the final pick we will be a bit more exploitative and a bit less RAVEy.
        cdef Node best = node.best_child(self._exploration_bias*0.9, self._rave_bias*0.9)

        if return_type == "state":
            return best.state
        elif return_type == "move":
            return best.move
        elif return_type == "node":
            return best


# Cache some math to speed up Node visits.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double sqrtlog(int x):
    return SQRTLOG[x] if 0 <= x < CACHE_MAX else sqrt(log(x))
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double invsqrt(int x):
    return INVSQRT[x] if 0 <= x < CACHE_MAX else 1 / sqrt(x)

cdef int CACHE_MAX = 10000
cdef double[10000] SQRTLOG, INVSQRT
cdef int x = 0
with cython.cdivision(True), cython.boundscheck(False), cython.wraparound(False):
    SQRTLOG[0] = 0
    INVSQRT[0] = 0
    for x in range(1,CACHE_MAX):
        SQRTLOG[x] = sqrt(log(x))
        INVSQRT[x] = 1 / sqrt(x)
