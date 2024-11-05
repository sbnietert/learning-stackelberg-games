import numpy as np
import pickle

from security_games.clinch import BatchedClinchThenCommit
from security_games.utils import RepeatedSSG, gen_non_myopic_with_bounded_lookahead_oracle, RepeatedSSGAlg, gen_simplex_SSG_2


def find_best_batch_size(game: RepeatedSSG,
                         minimum_width: float,
                         time_horizon: int,
                         get_response: RepeatedSSGAlg.FollowerResponseOracle,
                         min_size: int,
                         max_size: int,
                         num_divisions: int,
                         branching_factor: int): # explore top branching_factor # divisions
    test_sizes = np.arange(min_size, max_size+1, np.maximum((max_size - min_size)//num_divisions,1))
    utilities = np.zeros(len(test_sizes))
    utilities_by_size = {}
    for (i,size) in enumerate(test_sizes):
        res = BatchedClinchThenCommit(game,
                                      minimum_width=minimum_width,
                                      time_horizon=time_horizon,
                                      is_simplex=True,
                                      get_response=get_response,
                                      batch_size=size,
                                      verbose=False).run()
        utilities[i] = utilities_by_size[size] = res.game.leader_utility

    top_size_indices = utilities.argsort()[-branching_factor:][::-1]
    if len(test_sizes) == max_size - min_size + 1: # all sizes tested
        i_star = top_size_indices[0]
        return (test_sizes[i_star], utilities[i_star])
    else:
        top_size_indices = sorted(top_size_indices) # sort the indices so that we scan in order
    
    for i in top_size_indices:
        if i-1 not in top_size_indices and i > 0: # should test interval before
            (B_l, u_l) = find_best_batch_size(game,
                                                     minimum_width,
                                                     time_horizon,
                                                     get_response, 
                                                     test_sizes[i-1]+1,
                                                     test_sizes[i]-1,
                                                     num_divisions,
                                                     branching_factor)
            utilities_by_size[B_l] = u_l
        if i < len(test_sizes) - 1:
            (B_r, u_r) = find_best_batch_size(game,
                                                     minimum_width,
                                                     time_horizon,
                                                     get_response, 
                                                     test_sizes[i]+1,
                                                     test_sizes[i+1]-1,
                                                     num_divisions,
                                                     branching_factor)
            utilities_by_size[B_r] = u_r
            
    max_utility = 0
    best_size = 0
    for size in utilities_by_size.keys():
        if utilities_by_size[size] > max_utility:
            max_utility = utilities_by_size[size]
            best_size = size

    assert best_size > 0
    return (best_size, max_utility)


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
    T = 500 # time horizon
    agent_lookahead = 1 # only one non-myopic lookahead step needed for interesting behavior
    agent_cutoff = 999999 # controls how far simulations are run into the future, could be cut down to improve performance
    get_response = gen_non_myopic_with_bounded_lookahead_oracle(agent_lookahead, agent_cutoff)
    discount_factors = [0.5, 0.75, 0.85] # should be of length 3 for batch size selection to work
    # path to save precomputed batch sizes
    batch_sizes_path = f'results/sec4/geometric_batch_sizes_T{T}.pkl'
    ## END CONFIG

    ssgs = []
    benchmark_solutions = []
    benchmark_payoffs = []
    null_payoffs = []
    games = []

    for i in range(n_games):
        ssg = gen_simplex_SSG_2(n_targets=n_targets, v_low=v_low, v_high=v_high)
        ssgs.append(ssg)
        game = RepeatedSSG(ssg, 0.5, discounting_type="geometric")
        games.append(game)    

    # find the best batch sizes at time horizon T for each discount factor
    B_star_table = {}
    for i in range(n_games):
        for gamma in discount_factors:
            print(f'game:{i}, gamma:{gamma}')
            (B_star, _) = find_best_batch_size(games[i].reset(discount_factor=gamma),W,T,get_response,1,20,5,2)
            print(f'B_star:{B_star}')
            B_star_table[(i,gamma)] = B_star

    # use these to produces an informative set of batch sizes for each plot
    batch_sizes_by_game = {}
    for i in range(n_games):
        B1 = B_star_table[(i,0.5)]
        B2 = B_star_table[(i,0.75)]
        B4 = B_star_table[(i,0.85)]
        B3 = B2 + (B4 - B2)//2
        B5 = 2*B4 - B3
        batch_sizes_by_game[i] = [B1,B2,B3,B4,B5]
        print((i,batch_sizes_by_game[i]))

    with open(batch_sizes_path, 'wb') as f: pickle.dump(batch_sizes_by_game, f)