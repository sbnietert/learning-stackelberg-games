from typing import Optional, Callable, Self, Literal

import copy
import itertools

import numpy as np
from scipy import optimize

EPSILON = 1e-9


class Polytope:
    """
    Polytope represented via constraints {x : Ax <= b}
    """

    def __init__(self, A: np.ndarray, b: np.ndarray):
        assert b.shape == (A.shape[0], 1)

        self.dimension = A.shape[1]
        self.A = A
        self.b = b

    def copy(self) -> Self:
        return type(self)(self.A.copy(), self.b.copy())

    def contains(self, x: np.ndarray):
        assert x.shape == (self.dimension, 1)
        return (self.A @ x <= self.b).all()

    def intersect(self, other: "Polytope"):
        return Polytope(
            np.concatenate([self.A, other.A]), np.concatenate([self.b, other.b])
        )

    def sample_interior_point(self, epsilon=1e-5) -> Optional[np.ndarray]:
        """
        Solves for the Chebyshev center of the polytope
        """
        A_norms = np.linalg.norm(self.A, axis=1).reshape(-1, 1)
        A_chebyshev = np.concatenate([self.A, A_norms], axis=1)
        c_chebyshev = np.concatenate([np.zeros(self.dimension), -np.ones(1)])
        result = optimize.linprog(
            c=c_chebyshev,
            A_ub=A_chebyshev,
            b_ub=self.b,
            bounds=(None, None),
        )
        if result.success and result.x[-1] > epsilon:
            x = result.x[:-1].reshape(-1, 1)
            assert (self.A @ x <= self.b - EPSILON).all(), (result.x[-1], result.slack)
            return x
        else:
            return None


class SSG:
    """
    Stackelberg security game implementation
    """

    def __init__(
        self,
        n_targets: int,
        constraints: Polytope,
        leader_payoffs: np.ndarray,
        follower_payoffs: np.ndarray,
    ):
        assert constraints.dimension == n_targets
        assert leader_payoffs.shape == (2, n_targets)
        assert follower_payoffs.shape == (2, n_targets)
        assert (leader_payoffs[1] > leader_payoffs[0]).all()
        assert (follower_payoffs[1] < follower_payoffs[0]).all()

        self.n_targets = n_targets
        self.constraints = constraints
        self.leader_payoffs = leader_payoffs
        self.follower_payoffs = follower_payoffs

        leader_delta = float(np.min(leader_payoffs[1] - leader_payoffs[0]))
        follower_delta = float(np.min(follower_payoffs[0] - follower_payoffs[1]))
        self.slope_bound = 1.0/min(leader_delta,follower_delta)

        self.call_count = 0

    def is_feasible(self, x: np.ndarray):
        """
        Checks feasibility of leader action x against constraints
        """
        assert x.shape == (self.n_targets, 1)
        return self.constraints.contains(x)

    def get_best_response(self, x: np.ndarray, boost_index=None, boost_epsilon=1e-5):
        """
        Implements best response oracle, returning follower's best response to leader action x
        """
        self.call_count += 1
        follower_payoffs = (
            self.follower_payoffs[1] * x.T + self.follower_payoffs[0] * (1.0 - x.T)
        ).squeeze()
        if boost_index is not None:
            follower_payoffs[boost_index] += boost_epsilon
        return int(np.argmax(follower_payoffs))

    def get_leader_payoff(self, x: np.ndarray):
        y = self.get_best_response(x)
        return self.leader_payoffs[1][y] * x[y] + self.leader_payoffs[0][y] * (
            1.0 - x[y]
        )
    

class RepeatedSSG:
    ssg: SSG
    follower_discount_factor: float
    discounting_type: Literal["geometric", "hyperbolic"]
    # note differing role of discount factor for the two types

    time: int = 0
    last_reset_time: int = 0
    leader_utility: float = 0
    follower_discounted_utility: float = 0 # since last reset

    def __init__(self, ssg: SSG, follower_discount_factor: float, discounting_type="geometric"):
        self.ssg = ssg
        self.follower_discount_factor = follower_discount_factor
        self.discounting_type = discounting_type

    def __str__(self):
        return f't:{self.time}, last reset: {self.last_reset_time}, leader util:{self.leader_utility:.2f}, follower util (since last reset):{self.follower_discounted_utility:.2f}'

    def copy(self) -> Self:
        return copy.copy(self)
    
    def reset(self, discount_factor: Optional[float] = None) -> Self:
        if discount_factor is None:
            return type(self)(self.ssg, self.follower_discount_factor, self.discounting_type)
        return type(self)(self.ssg, discount_factor, self.discounting_type)

    def step(self, x: np.ndarray, y: int):
        leader_payoffs = self.ssg.leader_payoffs
        follower_payoffs = self.ssg.follower_payoffs
        u = x[y] * leader_payoffs[1][y] + (1.0 - x[y]) * leader_payoffs[0][y]
        v = x[y] * follower_payoffs[1][y] + (1.0 - x[y]) * follower_payoffs[0][y]
        self.leader_utility += float(u)
        dt = self.time - self.last_reset_time
        if self.discounting_type == "geometric":
            discount_multiplier = self.follower_discount_factor ** dt
        elif self.discounting_type == "hyperbolic":
            discount_multiplier = 1.0/(1.0 + self.follower_discount_factor*dt)
        self.follower_discounted_utility += discount_multiplier * float(v)
        self.time += 1
    
    def reset_follower_utility(self):
        self.follower_discounted_utility = 0
        self.last_reset_time = self.time

class RepeatedSSGAlg:
    """
    Generic class from which all leader algorithms for repeated SSGs will be instantiated
    """

    # top-level params, set at initialization and unchanged
    game: RepeatedSSG
    time_horizon: int # total number of follower response queries

    _prev_query: Optional[np.ndarray] = None

    # type definition for follower response oracle
    FollowerResponseOracle = Callable[[np.ndarray, Self], int] 
    get_response: FollowerResponseOracle

    def __init__(self, game: RepeatedSSG, time_horizon: int, get_response: FollowerResponseOracle):
        self.game = game.reset()
        self.time_horizon = time_horizon
        self.get_response = get_response

    def __str__(self) -> str:
        return str(self.game)
    
    # needs to be a deep copy, should be overwritten for subclasses if needed
    def copy(self, get_response: FollowerResponseOracle, reset_follower_utility=False) -> Self:
        obj = copy.copy(self)
        obj.game = obj.game.copy()
        obj.get_response = get_response
        if self._prev_query is not None:
            obj._prev_query = self._prev_query.copy()
        if reset_follower_utility:
            obj.game.reset_follower_utility()
        return obj

    def query_follower(self, x: np.ndarray) -> int:
        self._prev_query = x
        y = self.get_response(x, self)
        self.game.step(x, y)
        return y
    
    def repeat_prev_query(self):
        assert(self._prev_query is not None)
        self.query_follower(self._prev_query)

    # leader algorithm should be implemented here, returns true until all follower queries have been used up
    # at most one follower query should be used per step (not enforced formally)
    def step(self) -> bool:
        return self.game.time < self.time_horizon

    def run(self, num_rounds: Optional[int] = None, verbose: bool = False) -> Self:
        if verbose: print(self)
        t = 0
        while((num_rounds is None or t < num_rounds) and self.step()):
            if verbose: print(self)
            t += 1
        return self


class SearchThenCommitSSGAlg(RepeatedSSGAlg):
    """
    Generic class from which all search-then-commit leader algorithms for repeated SSGs will be instantiated
    Search phase operates in batches of specified size to address non-myopic agents
    """

    batch_size: int
    search_result: Optional[np.ndarray] = None
    exploit_leader_utility: float = 0 # tracks leader utility after search completed

    def __init__(self, game: RepeatedSSG, time_horizon: int, batch_size: int, get_response: RepeatedSSGAlg.FollowerResponseOracle):
        super().__init__(game, time_horizon, get_response)
        self.batch_size = batch_size


    def __str__(self) -> str:
        s = ""
        if self.search_result is not None:
            s += f'Search complete\nSearch result: {self.search_result.flatten()}\n'
        s += super().__str__()
        return s

    def copy(self, get_response: RepeatedSSGAlg.FollowerResponseOracle, reset_follower_utility=False):
        obj = super().copy(get_response, reset_follower_utility)
        if self.search_result is not None:
            obj.search_result = self.search_result.copy()
        return obj

    def end_search(self, result: np.ndarray):
        self.search_result = result

    # at most one follower query per search step, should set "search_result" once search has terminated
    def search_step(self):
        assert False
    
    def step(self) -> bool:
        if self.search_result is None:
            if self.game.time % self.batch_size == 0:
                    self.search_step()
            else:
                self.repeat_prev_query()
        else:
            u0 = self.game.leader_utility
            self.query_follower(self.search_result)
            self.exploit_leader_utility += self.game.leader_utility - u0
        return self.game.time < self.time_horizon
    
# generate a follower response oracle which follows a list of specified moves before best responding
def _gen_follow_sequence_then_best_respond_oracle(seq) -> RepeatedSSGAlg.FollowerResponseOracle:
    i = 0
    def get_response(x: np.ndarray, alg: RepeatedSSGAlg):
        nonlocal i
        if i < len(seq):
            i += 1
            return seq[i-1]
        else:
            return alg.game.ssg.get_best_response(x)
    return get_response

# generate a follower response oracle which chooses the best policy among those with a bounded lookahead and episode length
# lookahead: num rounds forward for which the agent will enumerate all possible responses
# cutoff: num rounds after which the agent will stop simulating the future
# if "lie_threshold" is set, the oracle outputs an alert whenever it chooses an action that is suboptimal by this amount
def gen_non_myopic_with_bounded_lookahead_oracle(lookahead: int, cutoff: int, lie_threshold: Optional[float] = None, utility_reset: bool = True) -> RepeatedSSGAlg.FollowerResponseOracle:
    assert lookahead <= cutoff
    if lookahead == 0:
        def get_response(x: np.ndarray, alg: RepeatedSSGAlg) -> int:
            return alg.game.ssg.get_best_response(x)
        return get_response
    # note that, while x is unused below, it is implicitly maintained within the state of clinch
    def get_response(x: np.ndarray, alg: RepeatedSSGAlg) -> int:
        nonlocal lookahead
        targets = set(range(alg.game.ssg.n_targets))
        max_follower_utility = 0.0
        best_y: Optional[int] = None

        # print('beginning simulations')
        for seq in itertools.product(targets, repeat=lookahead):
            get_response = _gen_follow_sequence_then_best_respond_oracle(seq)
            # print('copying alg and simulating')
            # print(alg.threads[0].search_result)
            follower_util = alg.copy(get_response, reset_follower_utility=utility_reset).run(num_rounds=cutoff).game.follower_discounted_utility
            # print(alg.threads[0].search_result)
            # print('simulation complete')
            if follower_util > max_follower_utility:
                max_follower_utility = follower_util
                best_y = seq[0]
        # print('all simulations complete')

        assert best_y is not None
        if lie_threshold is not None:
            br = alg.game.ssg.get_best_response(x)
            if best_y is not br:
                V = alg.game.ssg.follower_payoffs
                br_payoff = x[br] * V[1][br] + (1.0 - x[br]) * V[0][br]
                y_payoff = x[best_y] * V[1][best_y] + (1.0 - x[best_y]) * V[0][best_y]
                if br_payoff - y_payoff > lie_threshold:
                    print(f'LIED: x: {x}, BR: {br}, y: {best_y}')
                    print(f'agent payoff for BR: {br_payoff}')
                    print(f'agent payoff for y: {y_payoff}')
                    print(alg)

        return best_y
    return get_response