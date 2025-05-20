import numpy as np
import pickle
from datetime import datetime
import argparse
import sys
import json

from security_games.clinch import BatchedClinchThenCommit
from security_games.utils import RepeatedSSG, gen_non_myopic_with_bounded_lookahead_oracle, gen_simplex_SSG_2
from security_games.multi_threaded_clinch import MultiThreadedClinch

from typing import NamedTuple, Literal

class SimulationConfig(NamedTuple):
    id: str
    prg_seed: int
    n_games: int
    n_targets: int
    v_low: float
    v_high: float
    time_horizons: list[int]
    discounting_type: Literal["geometric", "hyperbolic"]
    discount_factors: list[float]
    agent_lookahead: int
    agent_cutoff: int
    batch_sizes_by_game: list[list[int]]
    W: float
    benchmark_precision: float
    verbose: bool    

def gen_games_and_benchmarks(config: SimulationConfig):
    ssgs = []
    benchmark_solutions = []
    benchmark_payoffs = []
    games = []

    for i in range(config.n_games):
        ssg = gen_simplex_SSG_2(n_targets=config.n_targets, v_low=config.v_low, v_high=config.v_high)
        ssgs.append(ssg)

        game = RepeatedSSG(ssg, 0.5, discounting_type=config.discounting_type)
        # choice of discount factor unimportant, not used below and will be overwritten
        games.append(game)

        get_response = lambda x,_: ssg.get_best_response(x)
        sol = BatchedClinchThenCommit(
            game, 
            minimum_width=config.W, 
            time_horizon=config.time_horizons[-1], # increase if search result not found
            is_simplex=True,
            get_response=get_response,
            batch_size=1,
            myopic=True,
            search_accuracy=config.benchmark_precision
        ).run().search_result
        assert sol is not None

        benchmark_solutions.append(sol)
        benchmark = ssg.get_leader_payoff(sol)
        benchmark_payoffs.append(benchmark)

    return (games, benchmark_payoffs)

# returns multi-threaded and single threaded utilities
def run_simulations(
    config: SimulationConfig,
    games,
    benchmark_payoffs # used for printing status
):
    get_response = gen_non_myopic_with_bounded_lookahead_oracle(config.agent_lookahead, config.agent_cutoff)

    # multi-threaded utilities
    mt_utilities = {} # indexed by (game #, gamma, T) tuples
    # single-threaded utilities
    st_utilities = {} # indexed by (game #, gamma, T, batch_size) tuples
    for i in range(len(games)):
        print(f'game:{i}', file=sys.stderr)
        print(f'null payoff:{games[i].ssg.get_leader_null_payoff()}', file=sys.stderr)
        for gamma in config.discount_factors:
            print(f'gamma:{gamma}', file=sys.stderr)
            for T in config.time_horizons:
                print(f'T:{T}', file=sys.stderr)
                res = MultiThreadedClinch(games[i].reset(discount_factor=gamma),
                                        minimum_width=config.W,
                                        time_horizon=T,
                                        is_simplex=True,
                                        get_response=get_response,
                                        verbose=False).run()
                mt_utilities[(i,gamma,T)] = res.game.leader_utility
                completed_threads = [2**(j+1) for j in range(len(res.threads)) if res.threads[j].search_result is not None]
                if config.verbose:
                    if res.highest_completed_thread >= 0:
                        best_result = res.threads[res.highest_completed_thread].search_result
                        best_result_payoff = benchmark_payoffs[i] - games[i].ssg.get_leader_payoff(best_result)
                    else:
                        best_result_payoff = "n/a"
                    print(f'MT-U:{res.game.leader_utility}, completed thread delays: {completed_threads}, result payoff: {best_result_payoff}', file=sys.stderr)
        
                for batch_size in config.batch_sizes_by_game[i]:
                    res = BatchedClinchThenCommit(games[i].reset(discount_factor=gamma),
                                                minimum_width=config.W,
                                                time_horizon=T,
                                                is_simplex=True,
                                                get_response=get_response,
                                                batch_size=batch_size,
                                                verbose=False).run()
                    st_utilities[(i,gamma,T,batch_size)] = res.game.leader_utility
                    if config.verbose:
                        if res.search_result is None:
                            result_payoff = "n/a"
                        else:
                            result_payoff = benchmark_payoffs[i] - games[i].ssg.get_leader_payoff(res.search_result)
                        print(f'B{batch_size}-U:{res.game.leader_utility}, search {"in progress" if res.search_result is None else "complete"}, result payoff: {result_payoff}', file=sys.stderr)
    return (mt_utilities, st_utilities)

def compute_regret(
    config: SimulationConfig,
    mt_utilities,
    st_utilities,
    benchmark_payoffs
):
    mt_regret = {} # indexed by (game #, gamma, T) tuples
    st_regret = {} # indexed by (game #, gamma, T, batch_size) tuples

    for i in range(config.n_games):
        benchmark = benchmark_payoffs[i]
        for gamma in config.discount_factors:
            for T in config.time_horizons:
                mt_regret[(i,gamma,T)] = T*benchmark - mt_utilities[(i,gamma,T)]
                for batch_size in config.batch_sizes_by_game[i]:
                    st_regret[(i,gamma,T,batch_size)] = T*benchmark - st_utilities[(i,gamma,T,batch_size)]
    
    return (mt_regret, st_regret)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve SSGs against simulated non-myopic agents")
    parser.add_argument("--config", type=str, required=True, help="Relative path to JSON configuration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    prg_seed = config.get("prg_seed", 1234) # random seed, for reproducibility
    n_games = config["n_games"] # number of SSGs to generate
    n_targets = config["n_targets"] # number of SSG targets
    v_low = config.get("v_low", 0.25) # agent utility LB for random SSG generation
    v_high = config.get("v_high", 0.75) # agent utility UB for random SSG generation
    max_time_horizon = config["max_time_horizon"]
    n_time_horizons = config.get("n_time_horizons", 20) # number of time horizons (between 0 and max_time_horizon) to be used
    discounting_type = config["discounting_type"]
    assert discounting_type in ['geometric', 'hyperbolic']
    discount_factors = config["discount_factors"]
    agent_lookahead = config.get("agent_lookahead", 1) # number of lookahead steps for non-myopic agent to explore with brute force search
    agent_cutoff = config.get("agent_cutoff", 999999) # length which non-myopic simulations are run into future
    benchmark_precision = config.get("benchmark_precision", 1e-4) # benchmark precision for regret computations
    verbose = config.get("verbose", False)

    # Handle mutually exclusive batch_size options
    batch_sizes = config.get("batch_sizes")
    batch_sizes_by_game = config.get("batch_sizes_by_game")
    if (batch_sizes is None) == (batch_sizes_by_game is None):
        raise ValueError("You must specify exactly one of 'batch_sizes' or 'batch_sizes_by_game'.")
    if batch_sizes_by_game is None:
        batch_sizes_by_game = [batch_sizes]*n_games

    W = v_low / (v_low + (n_targets - 1)*v_high) # regularity parameter for Clinch, just a heuristic
    time_horizon_skip = max_time_horizon // n_time_horizons

    config = SimulationConfig(
        id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), # unique per run
        prg_seed = prg_seed,
        n_games = n_games,
        n_targets = n_targets,
        v_low = v_low,
        v_high = v_high,
        time_horizons = np.arange(time_horizon_skip, max_time_horizon+1, time_horizon_skip).tolist(),
        discounting_type = discounting_type,
        discount_factors = discount_factors,
        agent_lookahead = agent_lookahead,
        agent_cutoff = agent_cutoff,
        batch_sizes_by_game = batch_sizes_by_game,
        W = W,
        benchmark_precision = benchmark_precision, 
        verbose = verbose
    )

    results_path = f'results/sec4/{config.discounting_type}_T{config.time_horizons[-1]}_n{config.n_targets}_{config.id}.pkl'
    np.random.seed(config.prg_seed)

    games, benchmark_payoffs = gen_games_and_benchmarks(config)
    mt_utilities, st_utilities = run_simulations(
        config,
        games,
        benchmark_payoffs
    )
    mt_regret, st_regret = compute_regret(
        config,
        mt_utilities,
        st_utilities,
        benchmark_payoffs
    )
    with open(results_path, 'wb') as f: pickle.dump((mt_regret, st_regret, games, benchmark_payoffs, config), f)
    print(results_path)