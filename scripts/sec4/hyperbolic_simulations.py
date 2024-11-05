import numpy as np

from security_games.clinch import BatchedClinchThenCommit
from security_games.utils import RepeatedSSG, gen_non_myopic_with_bounded_lookahead_oracle, gen_simplex_SSG_2
from security_games.multi_threaded_clinch import MultiThreadedClinch

# returns multi-threaded and single threaded utilities
def run_simulations(
        games,
        batch_sizes,
        discount_factors,
        time_horizons,
        W,
        verbose,
        null_payoffs,
        benchmark_payoffs
):
    # multi-threaded utilities
    mt_utilities = {} # indexed by (game #, gamma, T) tuples
    # single-threaded utilities
    st_utilities = {} # indexed by (game #, gamma, T, batch_size) tuples
    for i in range(len(games)):
        print(f'game:{i}')
        print(f'null payoff:{null_payoffs[i]}')
        for gamma in discount_factors:
            print(f'gamma:{gamma}')
            for T in time_horizons:
                print(f'T:{T}')
                res = MultiThreadedClinch(games[i].reset(discount_factor=gamma),
                                        minimum_width=W,
                                        time_horizon=T,
                                        is_simplex=True,
                                        get_response=get_response,
                                        verbose=False).run()
                mt_utilities[(i,gamma,T)] = res.game.leader_utility
                completed_threads = [2**(j+1) for j in range(len(res.threads)) if res.threads[j].search_result is not None]
                if verbose:
                    if res.highest_completed_thread >= 0:
                        best_result = res.threads[res.highest_completed_thread].search_result
                        best_result_payoff = benchmark_payoffs[i] - ssgs[i].get_leader_payoff(best_result)
                    else:
                        best_result_payoff = "n/a"
                    print(f'MT-U:{res.game.leader_utility}, completed thread delays: {completed_threads}, result payoff: {best_result_payoff}')
        
                for batch_size in batch_sizes:
                    res = BatchedClinchThenCommit(games[i].reset(discount_factor=gamma),
                                                minimum_width=W,
                                                time_horizon=T,
                                                is_simplex=True,
                                                get_response=get_response,
                                                batch_size=batch_size,
                                                verbose=False).run()
                    st_utilities[(i,gamma,T,batch_size)] = res.game.leader_utility
                    if verbose:
                        if res.search_result is None:
                            result_payoff = "n/a"
                        else:
                            result_payoff = benchmark_payoffs[i] - ssgs[i].get_leader_payoff(res.search_result)
                        print(f'B{batch_size}-U:{res.game.leader_utility}, search {"in progress" if res.search_result is None else "complete"}, result payoff: {result_payoff}')
    return (mt_utilities, st_utilities)

def compute_regret(
        mt_utilities,
        st_utilities,
        batch_sizes,
        benchmark_payoffs,
        discount_factors,
        time_horizons
):
    mt_regret = {} # indexed by (game #, gamma, T) tuples
    st_regret = {} # indexed by (game #, gamma, T, batch_size) tuples

    for i in range(len(benchmark_payoffs)):
        benchmark = benchmark_payoffs[i]
        for gamma in discount_factors:
            for T in time_horizons:
                mt_regret[(i,gamma,T)] = T*benchmark - mt_utilities[(i,gamma,T)]
                for batch_size in batch_sizes:
                    st_regret[(i,gamma,T,batch_size)] = T*benchmark - st_utilities[(i,gamma,T,batch_size)]
    
    return (mt_regret, st_regret)

if __name__ == "__main__":
    ## CONFIG: ensure consistency with simulations
    # params for generating SSGs and benchmark solutions
    np.random.seed(1234)
    n_games = 5
    n_targets = 3
    v_low = 0.25
    v_high = 0.75
    W = v_low / (v_low + (n_targets - 1)*v_high)
    benchmark_precision = 1e-4
    # params for non-myopic simulations
    time_horizons = np.arange(20, 501, 20)
    agent_lookahead = 1 # only one non-myopic lookahead step needed for interesting behavior
    agent_cutoff = 999999 # controls how far simulations are run into the future, could be cut down to improve performance
    get_response = gen_non_myopic_with_bounded_lookahead_oracle(agent_lookahead, agent_cutoff)
    discount_factors = [0.5, 0.75, 0.85] # should be of length 3 for batch size selection to work
    batch_sizes = [1, 2, 4, 8, 12]  
    verbose = False
    # path to save regret
    regret_path = 'results/sec4/hyperbolic_results_T500_B-1-2-4-8-12.npy'
    ## END CONFIG


    ssgs = []
    benchmark_solutions = []
    benchmark_payoffs = []
    null_payoffs = []
    games = []

    for i in range(n_games):
        ssg = gen_simplex_SSG_2(n_targets=n_targets, v_low=v_low, v_high=v_high)
        ssgs.append(ssg)

        game = RepeatedSSG(ssg, 0.5, discounting_type="hyperbolic")
        games.append(game)

        get_response = lambda x,_: ssg.get_best_response(x)
        sol = BatchedClinchThenCommit(game, 
                                    minimum_width=W, 
                                    time_horizon=1000, # increase if search result not found
                                    is_simplex=True,
                                    get_response=get_response,
                                    batch_size=1,
                                    myopic=True,
                                    search_accuracy=benchmark_precision).run().search_result

        benchmark_solutions.append(sol)
        benchmark = ssg.get_leader_payoff(sol)
        benchmark_payoffs.append(benchmark)

        null_payoffs.append(ssg.get_leader_payoff(np.ones(n_targets)/n_targets))


    (mt_utilities, st_utilities) = run_simulations(
        games,
        batch_sizes,
        discount_factors,
        time_horizons,
        W,
        verbose,
        null_payoffs,
        benchmark_payoffs
    )

    (mt_regret, st_regret) = compute_regret(
        mt_utilities,
        st_utilities,
        batch_sizes,
        benchmark_payoffs,
        discount_factors,
        time_horizons
    )

    np.save(regret_path, (mt_regret, st_regret, benchmark_payoffs, ssgs))