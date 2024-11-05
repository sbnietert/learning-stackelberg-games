import numpy as np
import matplotlib.pyplot as plt


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
    discount_factors = [0.5, 0.75, 0.85] # should be of length 3 for batch size selection to work
    # path to precomputed batch sizes
    batch_sizes_path = 'results/sec4/geometric_batch_sizes_T500.npy'
    # path to regret
    regret_path = 'results/sec4/geometric_results_T500_B-precomputed.npy'
    # plot options
    y_lim_list = [70, 35, 20, 80, 60]
    n_games = 5
    main_figure_path = 'results/sec4/geometric_main.pdf'
    extras_figure_path = 'results/sec4/geometric_extras.pdf'
    ## END CONFIG

    batch_sizes_by_game = np.load(batch_sizes_path, allow_pickle=True).tolist()
    B_star_by_discount_factor = {
        discount_factors[0]: batch_sizes_by_game[0][0],
        discount_factors[1]: batch_sizes_by_game[0][1],
        discount_factors[2]: batch_sizes_by_game[0][3],
    }

    (mt_regret, st_regret, benchmark_payoffs, ssgs) = np.load(regret_path, allow_pickle=True)


    # main plot (first game)
    fig, axes = plt.subplots(1,len(discount_factors)+1, figsize=(14/1.25,3.5/1.25))
    
    for (j,gamma) in enumerate(discount_factors): 
        x = time_horizons
        y_mt = [mt_regret[(0,gamma,T)] for T in time_horizons]
        y_null = [T*benchmark_payoffs[0] - T*ssgs[0].get_leader_payoff(np.ones(n_targets)/n_targets) for T in time_horizons]
        y_st_table = {}
    
        batch_sizes = sorted(list(set([key[3] for key in st_regret.keys() if key[1] == gamma])))
        for batch_size in batch_sizes_by_game[0]:
            y_st_table[batch_size] = [st_regret[(0,gamma,T,batch_size)] for T in time_horizons]
        
        axes[j].plot(x,y_mt,"-*",label="Multi-Threaded Clinch")
        
        for (k,batch_size) in enumerate(batch_sizes_by_game[0]):
            axes[j].plot(x,y_st_table[batch_size],label=f"Batched Clinch (B={batch_size})")
        axes[j].plot(x,y_null,"--",label="Null strategy (B=T)")
        axes[j].set_xlabel("Time horizon (T)")
        if j == 0:
            axes[j].set_ylabel("Regret")
        axes[j].set_ylim([0,y_lim_list[0]])
        axes[j].set_title(f"γ = {gamma}, B* = {B_star_by_discount_factor[gamma]}")
        if j == 0:
            handles, labels = axes[j].get_legend_handles_labels()
    
    axes[-1].set_axis_off()
    axes[-1].legend(handles, labels, loc='right')

    plt.suptitle('Regret of Multi-Threaded vs. Batched Clinch', y=1)
    plt.tight_layout()
    fig.savefig(main_figure_path)

    # extras plot

    fig, axes = plt.subplots(n_games-1,len(discount_factors)+1, figsize=(14/1.25,(n_games-1)*3.5/1.25))

    for i in range(1,n_games):
        B_star_by_discount_factor = {
            0.5: batch_sizes_by_game[i][0],
            0.75: batch_sizes_by_game[i][1],
            0.85: batch_sizes_by_game[i][3],
        }
        
        for (j,gamma) in enumerate(discount_factors): 
            x = time_horizons
            y_mt = [mt_regret[(i,gamma,T)] for T in time_horizons]
            y_null = [T*benchmark_payoffs[i] - T*ssgs[i].get_leader_payoff(np.ones(n_targets)/n_targets) for T in time_horizons]
            y_st_table = {}
        
            batch_sizes = sorted(list(set([key[3] for key in st_regret.keys() if key[1] == gamma])))
            for batch_size in batch_sizes_by_game[i]:
                y_st_table[batch_size] = [st_regret[(i,gamma,T,batch_size)] for T in time_horizons]
            
            axes[i-1][j].plot(x,y_mt,"-*",label="Multi-Threaded Clinch")
            
            for (k,batch_size) in enumerate(batch_sizes_by_game[i]):
                axes[i-1][j].plot(x,y_st_table[batch_size],label=f"Batched Clinch (B={batch_size})")
            axes[i-1][j].plot(x,y_null,"--",label="Null strategy (B=T)")
            axes[i-1][j].set_xlabel("Time horizon (T)")
            if j == 0:
                axes[i-1][j].set_ylabel("Regret")
            axes[i-1][j].set_ylim([0,y_lim_list[i]])
            axes[i-1][j].set_title(f"γ = {gamma}, B* = {B_star_by_discount_factor[gamma]}")
            if j == 0:
                handles, labels = axes[i-1][j].get_legend_handles_labels()
        
        axes[i-1][-1].set_axis_off()
        axes[i-1][-1].legend(handles, labels, loc='right')

    plt.suptitle('Regret of Multi-Threaded vs. Batched Clinch', y=1)
    plt.tight_layout()

    fig.savefig(extras_figure_path)