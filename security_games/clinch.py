import numpy as np

from security_games.utils import SSG, RepeatedSSG, Polytope, RepeatedSSGAlg, SearchThenCommitSSGAlg
from typing import Optional, Set, Literal, Self


def conserve_mass(game: SSG, x: np.ndarray, x_lower: np.ndarray, epsilon: float):
    for i in range(game.n_targets):
        lb, ub = x_lower[i], x[i]
        while ub - lb > epsilon:
            m = (lb + ub) / 2
            x[i] = m
            if game.get_best_response(x) == i:
                lb = m
            else:
                ub = m
        x[i] = lb
    return x


def step_hit_and_run_mcmc(P: Polytope, x: np.ndarray, z: np.ndarray, epsilon: float):
    if not P.contains(x - epsilon * z) and not P.contains(x + epsilon * z):
        return x

    lb, ub = -epsilon, epsilon
    while P.contains(x + lb * z):
        lb *= 2
    while P.contains(x + ub * z):
        ub *= 2

    w = np.random.uniform(lb, ub)
    while not P.contains(x + w * z):
        w = np.random.uniform(lb, ub)

    return x + w * z


def sample_approximate_centroid(
    P: Polytope, x_lb: np.ndarray, active_set: set[int], epsilon, iters=3000
):
    A_augmented = np.concatenate((P.A, -np.eye(P.dimension)), axis=0)
    b_augmented = np.concatenate((P.b, -x_lb), axis=0)
    P_augmented = Polytope(A_augmented, b_augmented)
    inactive_mask = np.array([i not in active_set for i in range(P.dimension)])

    x = x_lb.copy()

    incr = 1e-7
    while P.contains(x + incr * np.ones(x.shape)):
        x += incr * np.ones(x.shape)
        incr *= 2

    iterates = []
    for _ in range(iters):
        z = np.random.normal(size=x.shape)
        z[inactive_mask] = 0.0
        z /= np.linalg.norm(z)

        x = step_hit_and_run_mcmc(P_augmented, x, z, epsilon)
        iterates.append(x.copy())

    return np.mean(iterates, axis=0)


def clinch(game: SSG, epsilon: float = 1e-3, clinch_simplex=False, verbose=True):
    game.call_count = 0

    if clinch_simplex:
        assert np.allclose(
            game.constraints.A,
            np.concatenate([-np.eye(game.n_targets), np.ones((1, game.n_targets))]),
        )
        assert np.allclose(
            game.constraints.b,
            np.concatenate([np.zeros((game.n_targets, 1)), np.ones((1, 1))]),
        )

    active_set = set(range(game.n_targets))
    x_lb = np.zeros((game.n_targets, 1))

    x = x_lb
    y = game.get_best_response(x)
    while y in active_set:
        if not clinch_simplex:
            x = sample_approximate_centroid(game.constraints, x_lb, active_set, epsilon)
        else:
            remainder = 1 - np.sum(x_lb)
            x = x_lb + remainder * np.ones(x_lb.shape) / game.n_targets
        y = game.get_best_response(x)
        x_lb[y] = x[y]

        for i in range(game.n_targets):
            e_i = np.zeros(x_lb.shape)
            e_i[i] = 1.0
            if not game.constraints.contains(x_lb + epsilon * e_i):
                active_set.remove(i)

    result = conserve_mass(game, x, x_lb, epsilon)

    if verbose:
        print(f"Result: {result}")
        print(f"Queries: {game.call_count}")

    return result

def round_clinch_result(x: np.ndarray, ssg: SSG, epsilon: float, minimum_width: float) -> np.ndarray:
    candidate_targets = [y for y in range(len(x)) if x[y] > minimum_width / 2]
    U = ssg.leader_payoffs
    candidate_utilities = [x[y] * U[1][y] + (1.0 - x[y]) * U[0][y] for y in candidate_targets]
    best_target = candidate_targets[np.argmax(candidate_utilities)]
    rounded = x.copy()
    rounded[best_target] -= minimum_width * epsilon / 2
    return rounded

# policy for repeated game, used for non-myopic simulations
class ClinchThenCommit(SearchThenCommitSSGAlg):
    # top-level params, set at initialization and unchanged
    epsilon: float # search accuracy
    is_simplex: bool

    search_stage: Literal["initial search","conserve mass"] = "initial search"

    # initial search state, updated throughout "initial search" stage
    active_search_targets: Set[int]
    x_lb: np.ndarray

    # conserve mass state, updated at end of "initial search" stage and throughout "conserve mass" stage
    current_result: np.ndarray
    conserve_mass_target: int = 0

    def __init__(self, game: RepeatedSSG, time_horizon: int, is_simplex: bool, get_response: RepeatedSSGAlg.FollowerResponseOracle):
        batch_size = 1
        super().__init__(game, time_horizon, batch_size, get_response)
        self.epsilon = 1.0/time_horizon
        self.is_simplex = is_simplex
        n = game.ssg.n_targets
        if is_simplex:
            assert np.allclose(
                game.ssg.constraints.A,
                np.concatenate([-np.eye(n), np.ones((1, n))]),
            )
            assert np.allclose(
                game.ssg.constraints.b,
                np.concatenate([np.zeros((n, 1)), np.ones((1, 1))]),
            )

        self.active_search_targets = set(range(n))
        self.x_lb = np.zeros((n, 1))
        self.current_result = np.zeros(self.x_lb.shape)

    def __str__(self) -> str:
        s = ""
        if self.search_result is None:
            s += "Search in progress\n"
            match self.search_stage:
                case "initial search":
                    s += f"Active targets: {self.active_search_targets}\n"
                    s += f"x_lb: {self.x_lb}\n"
                case "conserve mass":
                    s += f"x_lb: {self.x_lb}\n"
                    s += f"Current target: {self.conserve_mass_target}\n"
                    s += f"Current result: {self.current_result.flatten()}\n"
        return s + super().__str__()
    
    def copy(self, get_response: RepeatedSSGAlg.FollowerResponseOracle) -> Self:
        obj = super().copy(get_response)
        obj.active_search_targets = self.active_search_targets.copy()
        obj.x_lb = self.x_lb.copy()
        obj.current_result = self.current_result.copy()
        return obj

    # NOTE: each step function calls query_follower at most once

    def initial_search_step(self):
        game = self.game
        x_lb = self.x_lb
        active_targets = self.active_search_targets

        if self.is_simplex:
            remainder = 1 - np.sum(x_lb)
            x = x_lb + remainder * np.ones(x_lb.shape) / game.ssg.n_targets
        else:
            x = sample_approximate_centroid(game.ssg.constraints, x_lb, active_targets, self.epsilon)

        y = self.query_follower(x)
        x_lb[y] = x[y]

        for i in range(game.ssg.n_targets):
            e_i = np.zeros(x_lb.shape)
            e_i[i] = 1.0
            if not game.ssg.constraints.contains(x_lb + self.epsilon * e_i):
                active_targets.remove(i)

        if y not in active_targets:
            self.search_stage = "conserve mass"
            self.current_result = x
    
    def conserve_mass_step(self):
        t = self.conserve_mass_target
        if self.current_result[t] - self.x_lb[t] > self.epsilon:
            query = self.current_result.copy()
            m = (self.current_result[t] + self.x_lb[t]) / 2
            query[t] = m
            y = self.query_follower(query)
            if y == t:
                self.x_lb[t] = m
            else:
                self.current_result[t] = m
        else:
            self.current_result[t] = self.x_lb[t]
            self.conserve_mass_target = t+1
            if self.conserve_mass_target >= self.game.ssg.n_targets:
                self.end_search(self.current_result)

    def search_step(self):
        match self.search_stage:
            case "initial search":
                self.initial_search_step()
            case "conserve mass":
                self.conserve_mass_step()


class BatchedClinchThenCommit(SearchThenCommitSSGAlg):
    # top-level params, set at initialization and unchanged
    is_simplex: bool
    epsilon: float # internal accuracy term corresponding to lambda in pseudo-code for Alg 1
    conservative_pull_back: float # corresponds to C*eps in pseudo-code for Alg 1
    search_accuracy: float

    search_stage: Literal["clinch","conserve mass"] = "clinch"

    # Clinch state, updated throughout "clinch" stage
    active_search_targets: Set[int]
    x_lb: np.ndarray

    # conserve mass state, updated at end of "initial search" stage and throughout "conserve mass" stage
    current_result: np.ndarray
    conserve_mass_target: int = 0

    def __init__(self, 
                 game: RepeatedSSG,
                 minimum_width: float,
                 time_horizon: int,
                 is_simplex: bool,
                 get_response: RepeatedSSGAlg.FollowerResponseOracle,
                 batch_size: int,
                 accuracy_exponent: float = 1.0,
                 myopic = False,
                 verbose = False,
                 search_accuracy: Optional[float] = None):
        
        super().__init__(game, time_horizon, batch_size, get_response)

        n = game.ssg.n_targets
        C = game.ssg.slope_bound
        
        self.is_simplex = is_simplex
        if is_simplex:
            assert np.allclose(
                game.ssg.constraints.A,
                np.concatenate([-np.eye(n), np.ones((1, n))]),
            )
            assert np.allclose(
                game.ssg.constraints.b,
                np.concatenate([np.zeros((n, 1)), np.ones((1, 1))]),
            )
        
        if search_accuracy is None:
            search_accuracy = (batch_size * n / time_horizon)**accuracy_exponent
            assert search_accuracy is not None
        self.search_accuracy = search_accuracy
        self.minimum_width = minimum_width

        super().__init__(game, time_horizon, batch_size, get_response)

        delta = minimum_width * search_accuracy / (6.0 * C ** 2) # required l_infty bound on output from Clinch
        self.epsilon = delta / (4 * C ** 2)

        if myopic:
            self.conservative_pull_back = 0
        else:
            oracle_accuracy = minimum_width * search_accuracy / (200.0 * C ** 5)
            # the delay from batching should induces best responses of this accuracy
            # if so, then this is a safe amount to pull back
            # otherwise, we have no guarantees
            self.conservative_pull_back = 2*C*oracle_accuracy

        self.active_search_targets = set(range(n))
        self.x_lb = np.zeros((n, 1))
        self.current_result = np.zeros(self.x_lb.shape)

        if verbose:
            print(f'batch size: {batch_size}')
            print(f'slope bound: {C}')
            print(f'search accuracy: {search_accuracy}')
            print(f'epsilon: {self.epsilon}')
            print(f'oracle accuracy: {oracle_accuracy}')
            print(f'conservative pull back: {self.conservative_pull_back}')

    def __str__(self) -> str:
        s = ""
        if self.search_result is None:
            s += "Search in progress\n"
            match self.search_stage:
                case "clinch":
                    s += f"Active targets: {self.active_search_targets}\n"
                    s += f"x_lb: {self.x_lb.flatten()}\n"
                case "conserve mass":
                    s += f"x_lb: {self.x_lb.flatten()}\n"
                    s += f"Current target: {self.conserve_mass_target}\n"
                    s += f"Current result: {self.current_result.flatten()}\n"
        return s + super().__str__()
    
    def copy(self, get_response: RepeatedSSGAlg.FollowerResponseOracle, reset_follower_utility=False) -> Self:
        obj = super().copy(get_response, reset_follower_utility)
        obj.active_search_targets = self.active_search_targets.copy()
        obj.x_lb = self.x_lb.copy()
        obj.current_result = self.current_result.copy()
        return obj
    
    def _round_search_result(self, x: np.ndarray) -> np.ndarray:
        candidate_targets = [y for y in range(len(x)) if x[y] > self.minimum_width / 2]
        U = self.game.ssg.leader_payoffs
        candidate_utilities = [x[y] * U[1][y] + (1.0 - x[y]) * U[0][y] for y in candidate_targets]
        best_target = candidate_targets[np.argmax(candidate_utilities)]
        rounded = x.copy()
        rounded[best_target] -= self.minimum_width * self.search_accuracy / 2
        return rounded

    # NOTE: each step function calls query_follower at most once

    def clinch_step(self):
        ssg = self.game.ssg
        x_lb = self.x_lb
        active_targets = self.active_search_targets

        if self.is_simplex:
            remainder = 1 - np.sum(x_lb)
            x = x_lb + remainder * np.ones(x_lb.shape) / ssg.n_targets
        else:
            x = sample_approximate_centroid(ssg.constraints, x_lb, active_targets, self.epsilon)

        y = self.query_follower(x)
        x_lb[y] = x[y] - self.conservative_pull_back

        for i in active_targets.copy():
            e_i = np.zeros(x_lb.shape)
            e_i[i] = 1.0
            if not ssg.constraints.contains(x_lb + self.epsilon * e_i):
                active_targets.remove(i)

        if y not in active_targets:
            self.search_stage = "conserve mass"
            self.current_result = x
    
    def conserve_mass_step(self):
        t = self.conserve_mass_target
        if self.current_result[t] - self.x_lb[t] > self.epsilon:
            query = self.current_result.copy()
            m = (self.current_result[t] + self.x_lb[t]) / 2
            query[t] = m
            y = self.query_follower(query)
            if y == t:
                self.x_lb[t] = m
            else:
                self.current_result[t] = m
        else:
            self.current_result[t] = self.x_lb[t]
            self.conserve_mass_target = t+1
            if self.conserve_mass_target >= self.game.ssg.n_targets:
                self.end_search(self._round_search_result(self.current_result))

    def search_step(self):
        match self.search_stage:
            case "clinch":
                self.clinch_step()
            case "conserve mass":
                self.conserve_mass_step()