import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from nonmyopic_simulations import SimulationConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plots results from non-myopic simulations")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--results", required=True, type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_base = json.load(f)

    mt_regret, st_regret, games, benchmark_payoffs, config = np.load(args.results, allow_pickle=True)
    figure_path = f'results/sec4/{config.discounting_type}_T{config.time_horizons[-1]}_n{config.n_targets}_{config.id}'

    # Handle mutually exclusive ylim options
    ylim = config_base.get("plot_ylim")
    ylim_by_game = config_base.get("plot_ylim_by_game")
    if (ylim is None) == (ylim_by_game is None):
        raise ValueError("You must specify exactly one of 'plot_ylim' or 'plot_ylim_by_game'.")
    if ylim_by_game is None:
        ylim_by_game = [ylim_by_game]*config.n_games

    game_blocks = []
    plot_split_index = config_base.get("plot_split_index")
    if plot_split_index is None:
        game_blocks.append(range(config.n_games))
    else:
        game_blocks.append(range(plot_split_index))
        game_blocks.append(range(plot_split_index, config.n_games))

    for block in game_blocks:
        fig, axes = plt.subplots(
            len(block),
            len(config.discount_factors)+1,
            figsize=(14/1.25, len(block)*3.5/1.25)
        )

        for plot_idx, game_idx in enumerate(block):
            for (j,gamma) in enumerate(config.discount_factors): 
                if len(block) > 1:
                    ax = axes[plot_idx][j]
                else:
                    ax = axes[j]
                x = config.time_horizons
                y_mt = [mt_regret[(game_idx,gamma,T)] for T in config.time_horizons]
                y_null = [T*benchmark_payoffs[game_idx] - T*games[game_idx].ssg.get_leader_null_payoff() for T in config.time_horizons]
                y_st_table = {}
            
                ax.plot(x,y_mt,"-*",label="Multi-Threaded Clinch")
                for batch_size in config.batch_sizes_by_game[game_idx]:
                    y_st_table[batch_size] = [st_regret[(game_idx,gamma,T,batch_size)] for T in config.time_horizons]
                    ax.plot(x,y_st_table[batch_size],label=f"Batched Clinch (B={batch_size})")
                ax.plot(x,y_null,"--",label="Null strategy (B=T)")
                ax.set_xlabel("Time horizon (T)")
                if j == 0:
                    ax.set_ylabel("Regret")
                ax.set_ylim([0,ylim_by_game[game_idx]])
                ax.set_title(f"{'gamma' if config.discounting_type == 'geometric' else 'k'} = {gamma}")
                if j == 0:
                    handles, labels = ax.get_legend_handles_labels()
            
            if len(block) > 1:
                ax = axes[plot_idx][-1]
            else:
                ax = axes[-1]
            ax.set_axis_off()
            ax.legend(handles, labels, loc='right')

        plt.suptitle(f'Regret of Multi-Threaded vs. Batched Clinch w/ {'Geometric' if config.discounting_type == 'geometric' else 'Hyperbolic'} Discounting')
        plt.tight_layout()

        if len(game_blocks) == 1:
            fig.savefig(figure_path + '.pdf')
        else:
            fig.savefig(figure_path + f'_plot{plot_idx}.pdf')