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
    batch_sizes = [1, 2, 4, 8, 12]  
    # path to  regret
    regret_path = 'results/sec4/hyperbolic_results_T500_B-1-2-4-8-12.npy'
    # plot options
    y_lim_list = [70, 35, 20, 80, 60]
    n_games = 5
    figure_path = 'results/sec4/hyperbolic.pdf'
    ## END CONFIG

    (mt_regret, st_regret, benchmark_payoffs, ssgs) = np.load(regret_path, allow_pickle=True).tolist()


    y_lim_list = [50, 50, 50, 50, 50]
    fig, axes = plt.subplots(n_games,len(discount_factors)+1, figsize=(14/1.25,n_games*3.5/1.25))

    for i in range(n_games):
        for (j,gamma) in enumerate(discount_factors): 
            x = time_horizons
            y_mt = [mt_regret[(i,gamma,T)] for T in time_horizons]
            y_null = [T*benchmark_payoffs[i] - T*ssgs[i].get_leader_payoff(np.ones(n_targets)/n_targets) for T in time_horizons]
            y_st_table = {}
        
            batch_sizes = sorted(list(set([key[3] for key in st_regret.keys() if key[1] == gamma])))
            for batch_size in batch_sizes:
                y_st_table[batch_size] = [st_regret[(i,gamma,T,batch_size)] for T in time_horizons]
            
            axes[i][j].plot(x,y_mt,"-*",label="Multi-Threaded Clinch")
            
            for (k,batch_size) in enumerate(batch_sizes):
                axes[i][j].plot(x,y_st_table[batch_size],label=f"Batched Clinch (B={batch_size})")
            axes[i][j].plot(x,y_null,"--",label="Null strategy (B=T)")
            axes[i][j].set_xlabel("Time horizon (T)")
            if j == 0:
                axes[i][j].set_ylabel("Regret")
            axes[i][j].set_ylim([0,y_lim_list[i]])
            axes[i][j].set_title(f"k = {gamma}")
            if j == 0:
                handles, labels = axes[i][j].get_legend_handles_labels()
        
        axes[i][-1].set_axis_off()
        axes[i][-1].legend(handles, labels, loc='right')

    plt.suptitle('Regret of Multi-Threaded vs. Batched Clinch w/ Hyperbolic Discounting')
    plt.tight_layout()

    fig.savefig(figure_path)