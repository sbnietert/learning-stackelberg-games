from typing import Optional, Literal, Callable, Self

import numpy as np
from scipy import optimize, stats

from security_games.security_search_utils import find_boundary_point, find_hyperplane
from security_games.utils import SSG, RepeatedSSG, Polytope, SearchThenCommitSSGAlg, RepeatedSSGAlg


def refine_polytope(
    game: SSG, i: int, P: Polytope, p: np.ndarray, x: np.ndarray, iters=100
):
    oracle = lambda x: game.get_best_response(x) == i
    y = find_boundary_point(oracle, p, x, iters=iters)
    v = find_hyperplane(oracle, y)
    return P.intersect(Polytope(v.T, v.T @ y))


def security_search(game: SSG, epsilon: float = 1e-8):
    game.call_count = 0
    # upper bound on best response polytope for each target
    upper_bounds = [game.constraints for _ in range(game.n_targets)]
    # true if upper bound polytope is tight, for each target
    completed = [False for _ in range(game.n_targets)]
    # None if search not started for target, otherwise contains some point within best response region for that target
    points: list[Optional[np.ndarray]] = [None for _ in range(game.n_targets)]

    # query arbitrary initial point and add to points array
    init_point = game.constraints.sample_interior_point()
    assert init_point is not None
    init_i = game.get_best_response(init_point)
    points[init_i] = init_point

    z = None # current result
    # while each search is complete or was never started
    while not all([completed[i] or (points[i] is None) for i in range(game.n_targets)]):

        # intersect polytopes
        done = False
        while not done:
            done = True
            for i in range(game.n_targets):
                P, p = upper_bounds[i], points[i]
                if p is None:
                    continue
                for j in range(i + 1, game.n_targets):
                    Q, q = upper_bounds[j], points[j]
                    if q is None:
                        continue
                    x = P.intersect(Q).sample_interior_point()
                    if x is not None:
                        done = False
                        k = game.get_best_response(x)

                        if k == i:
                            upper_bounds[j] = refine_polytope(game, j, Q, q, x)
                        if k == j:
                            upper_bounds[i] = refine_polytope(game, i, P, p, x)

                        if points[k] is None:
                            print(f"Found new best response {k}")
                            points[k] = x

        # security search
        for i in range(game.n_targets):
            if points[i] is None or completed[i]:
                continue

            e_i = np.zeros((game.n_targets, 1))
            e_i[i] = 1
            A_norms = np.linalg.norm(upper_bounds[i].A, axis=1).reshape(-1, 1)
            result = optimize.linprog(
                c=-e_i,
                A_ub=upper_bounds[i].A,
                b_ub=upper_bounds[i].b - epsilon * A_norms,
                bounds=(None, None),
                method="simplex",
            )
            assert result.success, A_norms
            p = result.x.reshape(-1, 1)
            for j in range(game.n_targets):
                if points[j] is None:
                    p[j] = 0
            k = game.get_best_response(p, boost_index=i)
            if k == i:
                completed[i] = True
                z = p
            else:
                assert points[k] is None, (i, k)
                print(f"Found new best response {k}")
                points[k] = p
                break

    print(f"Result: {z}")
    print(f"Queries: {game.call_count}")

    return z


def _get_simplex_vertices(dimension: int):
    alpha = 1.0 / dimension * (1 - 1.0 / np.sqrt(dimension + 1))
    beta = 1.0 / np.sqrt(dimension + 1)

    simplex_vertices = [-beta * np.ones((dimension, 1))]
    for i in range(dimension):
        e_i = np.zeros((dimension, 1))
        e_i[i] = 1
        simplex_vertices.append(e_i - alpha * np.ones((dimension, 1)))

    return simplex_vertices


SecuritySearchStage = Literal[
    "initial_query",
    "main_loop",
    "intersect_polytopes",
    "refine_polytope",
    "find_boundary_point",
    "find_hyperplane",
    "security_search"
]

# search policy for potential non-myopic simulations. currently not used
class SecuritySearchThenCommit(SearchThenCommitSSGAlg):
    # top-level params, set at initialization and unchanged
    epsilon: float
    fallback_result: np.ndarray # return this if un-defined behavior reached

    # search state, shared by all search stages
    search_stage: SecuritySearchStage = "initial_query"
    stage_map: dict[SecuritySearchStage, Callable]
    upper_bounds: list[Polytope]
    completed: list[bool]
    points: list[Optional[np.ndarray]]
    current_result: Optional[np.ndarray] = None

    # intersect polytopes state
    ip_updated: bool = True # has polytope been updated
    ip_i: int = 0 # outer index
    ip_j: int = 1 # inner index

    # refine polytope state
    rp_i: int = -1
    rp_P: Optional[Polytope] = None
    rp_p: Optional[np.ndarray] = None
    rp_x: Optional[np.ndarray] = None
    rp_y: Optional[np.ndarray] = None
    rp_v: Optional[np.ndarray] = None

    # find boundary point state
    fb_next_stage: Optional[SecuritySearchStage] = None
    fb_return_variable: str = ""
    fb_target: int = -1
    fb_p: Optional[np.ndarray] = None
    fb_x: Optional[np.ndarray] = None
    fb_iters: int = 100
    fb_midpoint: Optional[np.ndarray] = None
    fb_i: int = -1

    # find hyperplane state
    fh_target: int = -1
    fh_p: Optional[np.ndarray] = None
    fh_iters: int = 100
    fh_epsilon: float = 1e-8
    fh_sentinel_points: Optional[list[np.ndarray]] = None
    fh_sentinel_labels: Optional[list[bool]] = None
    fh_i: int = -1
    fh_j: int = -1
    fh_true_point: Optional[np.ndarray] = None
    fh_false_point: Optional[np.ndarray] = None
    fh_curr_boundary_point: Optional[np.ndarray] = None
    fh_boundary_points: list[np.ndarray] = []

    # security search state
    ss_i: int = -1

    def __init__(self, game: RepeatedSSG, time_horizon: int, get_response: RepeatedSSGAlg.FollowerResponseOracle):
        batch_size = 1
        super().__init__(game, time_horizon, batch_size, get_response)
        n = game.ssg.n_targets
        self.epsilon = 1.0/time_horizon
        self.fallback_result = np.ones(n)/n # if assertion fails, fall back to this solution (in simplex case, this is optimal)

        self.upper_bounds = [game.ssg.constraints] * n
        self.completed = [False] * n
        self.points = [None] * n # points[i] is None if we have never encountered a point with follower response i

    def __str__(self) -> str:
        s = ""
        if self.search_result is None:
            s += f'Search in progress: {self.search_stage}\n'
        
        return s + super().__str__()

    def copy(self, get_response: RepeatedSSGAlg.FollowerResponseOracle, reset_follower_utility=False) -> Self:
        obj = super().copy(get_response, reset_follower_utility)
        obj.upper_bounds = [p.copy() for p in obj.upper_bounds]
        obj.completed = obj.completed.copy()
        obj.points = [(None if p is None else p.copy()) for p in obj.points]
        obj.current_result = None if obj.current_result is None else obj.current_result.copy()

        obj.rp_P = None if obj.rp_P is None else obj.rp_P.copy()
        obj.rp_p = None if obj.rp_p is None else obj.rp_p.copy()
        obj.rp_x = None if obj.rp_x is None else obj.rp_x.copy()
        obj.rp_y = None if obj.rp_y is None else obj.rp_y.copy()
        obj.rp_v = None if obj.rp_v is None else obj.rp_v.copy()

        obj.fb_p = None if obj.fb_p is None else obj.fb_p.copy()
        obj.fb_x = None if obj.fb_x is None else obj.fb_x.copy()
        obj.fb_midpoint = None if obj.fb_midpoint is None else obj.fb_midpoint.copy()

        obj.fh_p = None if obj.fh_p is None else obj.fh_p.copy()
        obj.fh_sentinel_points = None if obj.fh_sentinel_points is None else [p.copy() for p in obj.fh_sentinel_points]
        obj.fh_sentinel_labels = None if obj.fh_sentinel_labels is None else obj.fh_sentinel_labels.copy()


        obj.fh_true_point = None if obj.fh_true_point is None else obj.fh_true_point.copy()
        obj.fh_false_point = None if obj.fh_false_point is None else obj.fh_false_point.copy()
        obj.fh_curr_boundary_point = None if obj.fh_curr_boundary_point is None else obj.fh_curr_boundary_point.copy()
        obj.fh_boundary_points = [p.copy() for p in obj.fh_boundary_points]

        return obj

    def initial_query_step(self):
        init_point = self.game.ssg.constraints.sample_interior_point()
        assert init_point is not None
        init_i = self.query_follower(init_point)
        self.points[init_i] = init_point
        self.search_stage = "main_loop"

    def main_loop_step(self):
        if all([self.completed[i] or (self.points[i] is None) for i in range(self.game.ssg.n_targets)]):
            # for each target i, its search is complete or was never started
            if self.current_result is None:
                self.end_search(self.fallback_result)
            else:
                self.end_search(self.current_result)
        else:
            return self.intersect_polytopes_init()

    def intersect_polytopes_init(self):
        self.ip_i = 0
        self.ip_j = 0
        self.search_stage = "intersect_polytopes"

    def intersect_polytopes_step(self):
        n = self.game.ssg.n_targets

        # increment polytope pair
        self.ip_j += 1
        if self.ip_j == n:
            if self.ip_i < n - 2:
                self.ip_i += 1
                self.ip_j = self.ip_i + 1
            elif self.ip_updated:
                self.ip_i = 0
                self.ip_j = 1
                self.ip_updated = False
            else:
                return self.security_search_init()

        i, j = self.ip_i, self.ip_j
        assert i >= 0 and i < n
        assert j > i and j < n

        P, p = self.upper_bounds[i], self.points[i]
        Q, q = self.upper_bounds[j], self.points[j]
        if p is None or q is None: return
        
        x = P.intersect(Q).sample_interior_point()
        if x is None: return

        self.ip_updated = True
        k = self.query_follower(x)
        if k == i:
            return self.refine_polytope_init(j, Q, q, x)
        elif k == j:
            return self.refine_polytope_init(i, P, p, x)
        elif self.points[k] is None:
            print(f"Found new best response {k}")
            self.points[k] = x

    def find_boundary_point_init(self, target: int, p: np.ndarray, x: np.ndarray, next_stage: SecuritySearchStage, return_variable: str, iters=100):
        self.fb_target = target
        self.fb_p = p
        self.fb_x = x
        self.fb_iters = iters
        self.fb_i = 0
        self.fb_midpoint = (self.fb_p + self.fb_x) / 2
        self.fb_next_stage = next_stage
        self.fb_return_variable = return_variable
        self.search_stage = "find_boundary_point"        
            
    def find_boundary_point_step(self):
        assert self.fb_midpoint is not None and self.fb_next_stage is not None
        assert self.fb_p is not None and self.fb_x is not None
        if self.query_follower(self.fb_midpoint) == self.fb_target:
            self.fb_p = self.fb_midpoint
        else:
            self.fb_x = self.fb_midpoint
        self.fb_midpoint = (self.fb_p + self.fb_x) / 2
        self.fb_i += 1
        if self.fb_i == self.fb_iters:
            setattr(self, self.fb_return_variable, self.fb_x)
            self.search_stage = self.fb_next_stage

    def find_hyperplane_init(self, target: int, p: np.ndarray):
        self.fh_target = target
        self.fh_p = p
        self.fh_i = self.fh_j = 0
        self.fh_true_point = None
        self.fh_false_point = None
        self.fh_curr_boundary_point = None
        self.fh_boundary_points = []

        random_rotation = stats.special_ortho_group.rvs(p.shape[0])
        simplex_vertices = _get_simplex_vertices(p.shape[0])
        rotated_simplex_vertices = [random_rotation @ v for v in simplex_vertices]
        self.fh_sentinel_points = [p + v * self.fh_epsilon for v in rotated_simplex_vertices]
        self.fh_sentinel_labels = [False] * len(self.fh_sentinel_points)

        self.search_stage = "find_hyperplane"

    def find_hyperplane_step(self):
        if self.fh_curr_boundary_point is not None:
            self.fh_boundary_points.append(self.fh_curr_boundary_point)
            self.fh_curr_boundary_point = None
            self.fh_j += 1

        assert self.fh_sentinel_points is not None and self.fh_sentinel_labels is not None
        if self.fh_i < len(self.fh_sentinel_points):
            x = self.fh_sentinel_points[self.fh_i]
            t = self.query_follower(x)
            self.fh_sentinel_labels[self.fh_i] = (t == self.fh_target)
            self.fh_i += 1
            return
        
        if self.fh_true_point is None:
            try:
                self.fh_true_point = self.fh_sentinel_points[self.fh_sentinel_labels.index(True)]
            except: 
                self.end_search(self.fallback_result)
                return
        if self.fh_false_point is None:
            try:
                self.fh_false_point = self.fh_sentinel_points[self.fh_sentinel_labels.index(False)]
            except:
                self.end_search(self.fallback_result)
                return

        if self.fh_j < len(self.fh_sentinel_points):
            x = self.fh_sentinel_points[self.fh_j]
            x_label = self.fh_sentinel_labels[self.fh_j]
            if x_label:
                self.find_boundary_point_init(
                    self.fh_target, x, self.fh_false_point,
                    "find_hyperplane", "fh_curr_boundary_point", iters=self.fh_iters
                )
            else:
                self.find_boundary_point_init(
                    self.fh_target, self.fh_true_point, x,
                    "find_hyperplane", "fh_curr_boundary_point", iters=self.fh_iters
                )
            return

        assert self.fh_p is not None
        boundary_points = np.hstack(self.fh_boundary_points).T
        boundary_points -= self.fh_p.T
        boundary_points /= np.linalg.norm(boundary_points, axis=1)[:, np.newaxis]

        _, S, V = np.linalg.svd(boundary_points)
        hyperplane = V[-1, :]

        if S[-1] / S[0] > 1e-2:
            print(S, self.fh_p)
            self.end_search(self.fallback_result)
            return

        hyperplane = V[-1, :][:, np.newaxis]
        if self.query_follower(self.fh_p + hyperplane * self.fh_epsilon) == self.fh_target:
            hyperplane = -hyperplane
        self.rp_v = hyperplane
        self.search_stage = "refine_polytope"

    def refine_polytope_init(self, i: int, P: Polytope, p: np.ndarray, x: np.ndarray):
        self.rp_i = i
        self.rp_P = P
        self.rp_p = p
        self.rp_x = x
        self.rp_y = self.rp_v = None
        self.search_stage = "refine_polytope"

    def refine_polytope_step(self):
        assert self.rp_p is not None and self.rp_x is not None and self.rp_P is not None
        if self.rp_y is None:
            self.find_boundary_point_init(self.rp_i, self.rp_p, self.rp_x, "refine_polytope", "rp_y")
        elif self.rp_v is None:
            self.find_hyperplane_init(self.rp_i, self.rp_y)
        else:
            v = self.rp_v
            self.upper_bounds[self.rp_i] = self.rp_P.intersect(Polytope(v.T, v.T @ self.rp_y))
            self.search_stage = "intersect_polytopes"

    def security_search_init(self):
        self.ss_i = 0
        self.search_stage = "security_search"

    def security_search_step(self):
        n = self.game.ssg.n_targets

        if self.ss_i >= n:
            self.search_stage = "main_loop"
            return
        
        i = self.ss_i
        self.ss_i += 1
        if self.points[i] is None or self.completed[i]:
            return

        e_i = np.zeros((n, 1))
        e_i[i] = 1
        A_norms = np.linalg.norm(self.upper_bounds[i].A, axis=1).reshape(-1, 1)
        result = optimize.linprog(
            c=-e_i,
            A_ub=self.upper_bounds[i].A,
            b_ub=self.upper_bounds[i].b - self.epsilon * A_norms,
            bounds=(None, None),
            method="simplex",
        )
        if not result.success:
            print(A_norms)
            self.end_search(self.fallback_result)
            return
        
        p = result.x.reshape(-1, 1)
        for j in range(n):
            if self.points[j] is None:
                p[j] = 0

        k = self.query_follower(p)
        if k == i:
            self.completed[i] = True
            self.current_result = p
        else:
            if self.points[k] is not None:
                print(i,k)
                self.end_search(self.fallback_result)
                return
            print(f"Found new best response {k}")
            self.points[k] = p
            self.search_stage = "main_loop"
            return

    def search_step(self):
        method_name = self.search_stage + "_step"
        method = getattr(self, method_name, None)
        assert method is not None
        method()
