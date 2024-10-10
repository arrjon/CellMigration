import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.patches import Patch

from synth_data_params_bayesflow.summary_stats import reduced_coordinates_to_sumstat, compute_mean_summary_stats, \
    compute_summary_stats, compute_autocorrelation


def plot_violin(ax, data, label, ylabel, alpha=0.05):
    colors = ['#1f77b4', '#ff7f0e']  # Blue for non-significant, orange for significant
    n_sim_plots = len(data) - 2

    plot = ax.violinplot(data, showmedians=True, showextrema=False)
    p_values = []
    for i, pc in enumerate(plot['bodies'][1:], start=1):
        if data[i][0] == 0 and data[i][1] == 0:
            # just a dummy value
            p_values.append(0)
            color = colors[1]
        else:
            a_test = stats.anderson_ksamp((data[0], data[i]), method=stats.PermutationMethod())
            p_values.append(a_test.pvalue)
            color = colors[1] if a_test.pvalue < alpha else colors[0]
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)

    ax.set_xticks(np.arange(n_sim_plots + 2) + 1, ['Data', 'Median-Simulation'] + ['Simulation'] * n_sim_plots,
                  rotation=90)
    ax.set_ylabel(ylabel)
    if np.sum(np.array(p_values) < alpha) > (len(data) - 1) // 2:
        ax.set_title(f'{label}\n(Statistically Different)')
    else:
        ax.set_title(f'{label}\n')


def plot_compare_summary_stats(test_sim: list, posterior_sim: list, path: str = None, compare_n: int = 5,
                               seed: int = 42):
    """
    Compare the summary statistics of the test simulation and the posterior simulations using the Anderson-Darling
    k-sample test and violin plots. Assumes that the first simulation of the posterior is the median/map.
    """
    if len(test_sim) != 1:
        raise ValueError("The test simulation should only have one population")

    np.random.seed(seed)
    factor = 1.31  # to scale to real world coordinates, assumes that data is Morpheus coordinates

    # compute the summary statistics
    synthetic_summary_stats_list = [reduced_coordinates_to_sumstat(t * factor) for t in
                                    test_sim]  # should be only one population
    simulation_synth_summary_stats_list = [reduced_coordinates_to_sumstat(pop_sim * factor) for pop_sim in posterior_sim]

    # compute the mean of the summary statistics
    (ad_mean_synth, ad_mean_synth_averg,
     MSD_mean_synth, MSD_mean_synth_averg,
     TA_mean_synth, TA_mean_synth_averg,
     VEL_mean_synth, VEL_mean_synth_averg,
     WT_mean_synth, WT_mean_synth_averg) = compute_mean_summary_stats(synthetic_summary_stats_list)
    (ad_mean_synth_sim, ad_mean_synth_sim_averg,
     MSD_mean_synth_sim, MSD_mean_synth_sim_averg,
     TA_mean_synth_sim, TA_mean_synth_sim_averg,
     VEL_mean_synth_sim, VEL_mean_synth_sim_averg,
     WT_mean_synth_sim, WT_mean_synth_sim_averg) = compute_mean_summary_stats(simulation_synth_summary_stats_list)

    fig, ax = plt.subplots(nrows=1, ncols=5, tight_layout=True, figsize=(12, 5))

    # always include the median (index 0)
    if len(ad_mean_synth_sim) == 1:
        random_index = [0]
    else:
        random_index = np.random.choice(range(1, len(ad_mean_synth_sim)),
                                        min(compare_n, len(ad_mean_synth_sim)) - 1, replace=False)
        random_index = np.append(0, random_index)

    # Perform the Anderson-Darling k-sample test and plot for each statistic
    # it might happen that lists are empty, so we need to check for that and add a dummy value
    # Angle Degree
    plot_violin(ax[0], [ad_mean_synth[0]] + [ad_mean_synth_sim[i] if len(ad_mean_synth_sim[i]) > 0 else [0, 0] for i in random_index],
                'Angle Degree', 'Angle Degree (degrees)\nMean per Cell')

    # Mean Squared Displacement (MSD)
    plot_violin(ax[1], [MSD_mean_synth[0]] + [MSD_mean_synth_sim[i] if len(MSD_mean_synth_sim[i]) > 0 else [0, 0]  for i in random_index],
                'Mean Squared Displacement', 'MSD\nMean per Cell')

    # Turning Angle
    plot_violin(ax[2], [TA_mean_synth[0]] + [TA_mean_synth_sim[i] if len(TA_mean_synth_sim[i]) > 0 else [0, 0]  for i in random_index],
                'Turning Angle', 'Turning Angle (radians)\nMean per Cell')

    # Velocity
    plot_violin(ax[3], [VEL_mean_synth[0]] + [VEL_mean_synth_sim[i] if len(VEL_mean_synth_sim[i]) > 0 else [0, 0]  for i in random_index],
                'Velocity', 'Velocity\nMean per Cell')

    # Waiting Time
    plot_violin(ax[4], [WT_mean_synth[0]] + [WT_mean_synth_sim[i] if len(WT_mean_synth_sim[i]) > 0 else [0, 0]  for i in random_index],
                'Waiting Time', 'Waiting Time (sec)\nMean per Cell')

    if path is not None:
        plt.savefig(f'{path}.png')
    plt.show()

    # Wasserstein distance
    wasserstein_distance = stats.wasserstein_distance(ad_mean_synth_sim[0], ad_mean_synth[0])
    wasserstein_distance += stats.wasserstein_distance(MSD_mean_synth_sim[0], MSD_mean_synth[0])
    wasserstein_distance += stats.wasserstein_distance(TA_mean_synth_sim[0], TA_mean_synth[0])
    wasserstein_distance += stats.wasserstein_distance(VEL_mean_synth_sim[0], VEL_mean_synth[0])
    wasserstein_distance += stats.wasserstein_distance(WT_mean_synth_sim[0], WT_mean_synth[0])
    print(f"Wasserstein distance: {wasserstein_distance}")

    return wasserstein_distance


def plot_trajectory(test_sim: np.ndarray, posterior_sim: np.ndarray,
                    path: str = None, label_true: str = None,
                    two_plots: bool = False, show_image: bool = False, show_umap: bool = False):
    """
    Plot the trajectories of the test simulation and one posterior simulation.
    """
    cells_in_population = test_sim.shape[0]

    if two_plots:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, tight_layout=True, figsize=(10, 5))
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, tight_layout=True, figsize=(5, 5))
        ax = [ax, ax]
    tiff_im = plt.imread('Cell_migration_grid_v3_final2_invers.tiff')

    # plot the simulations or the MAP
    if show_image and two_plots:
        ax[1].imshow(tiff_im)
    ax[1].plot(posterior_sim[0, :, 0], posterior_sim[0, :, 1], 'b', label='Simulated Trajectories', alpha=1)
    for cell_id in range(1, cells_in_population):
        ax[1].plot(posterior_sim[cell_id, :, 0], posterior_sim[cell_id, :, 1], 'b')

    # plot the synthetic data
    if show_image:
        ax[0].imshow(tiff_im)
    if label_true is None:
        label_true = 'Synthetic Trajectories'
    ax[0].plot(test_sim[0, :, 0], test_sim[0, :, 1], 'r', label=label_true, alpha=1)
    for cell_id in range(1, cells_in_population):
        ax[0].plot(test_sim[cell_id, :, 0], test_sim[cell_id, :, 1], 'r')

    ax[0].set_ylabel('y')
    for a in ax:
        a.legend()
        a.set_xlabel('x')
    if path is not None:
        plt.savefig(f'{path}.png')
    plt.show()

    if show_umap:
        import umap

        trajectories = np.concatenate([test_sim, posterior_sim])
        # wide format
        trajectories = np.concatenate([trajectories[..., 0], trajectories[..., 1]], axis=1)
        trajectories[np.isnan(trajectories)] = -1

        color_code = np.concatenate([np.zeros(test_sim.shape[0]), np.ones(posterior_sim.shape[0])])
        colors = ['#1f77b4', '#ff7f0e']

        reducer = umap.UMAP(random_state=42, n_jobs=1)
        embedding = reducer.fit_transform(trajectories)

        plt.figure(tight_layout=True, figsize=(5, 5))
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=[colors[int(i)] for i in color_code],
            alpha=0.5,
        )
        plt.gca().set_aspect('equal', 'datalim')
        patches = [Patch(color=colors[i], label=f'{[label_true, "Simulated"][i]}') for i in range(2)]
        plt.legend(handles=patches)
        plt.title('UMAP Projection')
        plt.savefig(f'{path}-umap.png')
        plt.show()

    return


def plot_autocorrelation(cell_population: np.ndarray, cell_population_2: np.ndarray = None, path: str = None):
    """
    Plot the autocorrelation of the summary statistics of the cell population for different statistics.
    """

    msd_list, ta_list, v_list, ad_list, wt_list = compute_summary_stats(cell_population)
    stats = [[msd_list, ta_list, v_list, ad_list]]
    if cell_population_2 is not None:
        msd_list_2, ta_list_2, v_list_2, ad_list_2, wt_list_2 = compute_summary_stats(cell_population_2)
        stats.append([msd_list_2, ta_list_2, v_list_2, ad_list_2])
        fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, tight_layout=True, figsize=(12, 4))
    else:
        fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, tight_layout=True, figsize=(12, 4))
        ax = [ax]

    for j, a in enumerate(ax):
        # plot auto-correlation
        for i, (statistic, label) in enumerate(zip(stats[j],
                                                   ['MSD', 'Turning Angle', 'Velocity', 'Angle Degree'])):
            autocorr_results = compute_autocorrelation(statistic)

            time_lag = np.arange(1, len(autocorr_results) + 2)
            # Plot each cell's autocorrelation
            for idx, autocorr in enumerate(autocorr_results):
                a[i].plot(time_lag, autocorr, alpha=0.6, color='teal')

            a[i].set_xlabel('Lag')
            a[i].set_title(label)
        a[0].set_ylabel('Autocorrelation')
    if path is not None:
        plt.savefig(f'{path}.png')
    plt.show()
    return

#
    # fig, ax = plt.subplots(nrows=1, ncols=5, tight_layout=True, figsize=(12, 5))
    # alpha = 0.05
    # colors = ['#1f77b4', '#ff7f0e']  # Blue for non-significant, orange for significant
    # np.random.seed(42)
    # random_index = np.random.choice(range(len(ad_mean_synth_sim)), 5, replace=False)
    #
    # def plot_violin(ax, data, label, ylabel):
    #     n_sim_plots = len(data) - 2
    #
    #     plot = ax.violinplot(data, showmedians=True, showextrema=False)
    #     p_values = []
    #     for i, pc in enumerate(plot['bodies'][1:], start=1):
    #         a_test = stats.anderson_ksamp((data[0], data[i]), method=stats.PermutationMethod())
    #         p_values.append(a_test.pvalue)
    #         color = colors[1] if a_test.pvalue < alpha else colors[0]
    #         pc.set_facecolor(color)
    #         pc.set_edgecolor('black')
    #         pc.set_alpha(0.8)
    #
    #     ax.set_xticks(np.arange(n_sim_plots + 2) + 1, ['Data', 'MAP-Simulation'] + ['Simulation'] * n_sim_plots,
    #                   rotation=90)
    #     ax.set_ylabel(ylabel)
    #     if np.sum(np.array(p_values) < alpha) > (len(data) - 1) // 2:
    #         ax.set_title(f'{label}\n(Statistically Different)')
    #     else:
    #         ax.set_title(f'{label}\n')
    #
    # # Perform the Anderson-Darling k-sample test and plot for each statistic
    # # Angle Degree
    # plot_violin(ax[0],
    #             [ad_mean_synth[0], ad_mean_synth_sim[map_idx_sim]] + [ad_mean_synth_sim[i] for i in random_index],
    #             'Angle Degree', 'Angle Degree (degrees)\nMean per Cell')
    #
    # # Mean Squared Displacement (MSD)
    # plot_violin(ax[1],
    #             [MSD_mean_synth[0], MSD_mean_synth_sim[map_idx_sim]] + [MSD_mean_synth_sim[i] for i in random_index],
    #             'Mean Squared Displacement', 'MSD\nMean per Cell')
    #
    # # Turning Angle
    # plot_violin(ax[2],
    #             [TA_mean_synth[0], TA_mean_synth_sim[map_idx_sim]] + [TA_mean_synth_sim[i] for i in random_index],
    #             'Turning Angle', 'Turning Angle (radians)\nMean per Cell')
    #
    # # Velocity
    # plot_violin(ax[3],
    #             [VEL_mean_synth[0], VEL_mean_synth_sim[map_idx_sim]] + [VEL_mean_synth_sim[i] for i in random_index],
    #             'Velocity', 'Velocity\nMean per Cell')
    #
    # # Waiting Time
    # plot_violin(ax[4],
    #             [WT_mean_synth[0], WT_mean_synth_sim[map_idx_sim]] + [WT_mean_synth_sim[i] for i in random_index],
    #             'Waiting Time', 'Waiting Time (sec)\nMean per Cell')
    #
    # plt.savefig(f'{checkpoint_path}/Summary Stats.png')
    # plt.show()
    #
    # # Wasserstein distance
    # wasserstein_distance = stats.wasserstein_distance(ad_mean_synth_sim[map_idx_sim], ad_mean_synth[0])
    # wasserstein_distance += stats.wasserstein_distance(MSD_mean_synth_sim[map_idx_sim], MSD_mean_synth[0])
    # wasserstein_distance += stats.wasserstein_distance(TA_mean_synth_sim[map_idx_sim], TA_mean_synth[0])
    # wasserstein_distance += stats.wasserstein_distance(VEL_mean_synth_sim[map_idx_sim], VEL_mean_synth[0])
    # wasserstein_distance += stats.wasserstein_distance(WT_mean_synth_sim[map_idx_sim], WT_mean_synth[0])
    # print(f"Wasserstein distance: {wasserstein_distance}")
    # # %% raw
    # # plot the summary statistics
    # fig, ax = plt.subplots(nrows=3, ncols=5, sharey='col', tight_layout=True, figsize=(12, 8))
    # n_sim_plots = 3
    #
    # # Perform the Kolmogorov-Smirnov test
    # # ks_statistic, p_value = stats.ks_2samp(ad_mean_synth_sim[map_idx_sim], ad_mean_synth[0])
    # a_test = stats.anderson_ksamp((ad_mean_synth_sim[map_idx_sim], ad_mean_synth[0]), method=stats.PermutationMethod())
    # ks_statistic, p_value = a_test.statistic, a_test.pvalue
    # print(f"Angle Degree KS Statistic: {ks_statistic}")
    # print(f"Angle Degree P-value: {p_value}, {p_value < 0.05}: different distributions")
    # ax[0, 0].violinplot([ad_mean_synth_sim[map_idx_sim], ad_mean_synth[0]], showmeans=True)
    # if p_value < 0.05:
    #     ax[0, 0].set_title(f'Angle Degree\n(Different)')
    # else:
    #     ax[0, 0].set_title(f'Angle Degree\n(Same)')
    # ax[0, 0].set_ylabel(f'Angle Degree (degrees)\nMean per Cell')
    # ax[1, 0].set_xticks([1, 2], ['Simulation', 'Synthetic'])
    # ax[1, 0].violinplot([ad_mean_synth_sim_averg, ad_mean_synth_averg], showmeans=True)
    # ax[1, 0].set_ylabel(f'Angle Degree (degrees)\nPopulation Mean')
    # ax[0, 0].set_xticks([1, 2], ['Simulation', 'Synthetic'])
    # ax[2, 0].violinplot([ad_mean_synth_sim[map_idx_sim]] + [ad_mean_synth_sim[i] for i in range(n_sim_plots)],
    #                     showmeans=True)
    # ax[2, 0].set_xticks(np.arange(n_sim_plots + 1) + 1, ['Map'] + ['Simulation'] * n_sim_plots, rotation=60)
    #
    # # ks_statistic, p_value = stats.ks_2samp(MSD_mean_synth_sim[map_idx_sim], MSD_mean_synth[0])
    # a_test = stats.anderson_ksamp((MSD_mean_synth_sim[map_idx_sim], MSD_mean_synth[0]),
    #                               method=stats.PermutationMethod())
    # ks_statistic, p_value = a_test.statistic, a_test.pvalue
    # print(f"MSD KS Statistic: {ks_statistic}")
    # print(f"MSD P-value: {p_value}, {p_value < 0.05}: different distributions")
    # ax[0, 1].violinplot([MSD_mean_synth_sim[map_idx_sim], MSD_mean_synth[0]], showmeans=True)
    # if p_value < 0.05:
    #     ax[0, 1].set_title(f'Mean Squared Displacement\n(Different)')
    # else:
    #     ax[0, 1].set_title(f'Mean Squared Displacement\n(Same)')
    # ax[0, 1].set_ylabel(f'MSD\nMean per Cell')
    # ax[0, 1].set_xticks([1, 2], ['Simulation', 'Synthetic'])
    # ax[1, 1].violinplot([MSD_mean_synth_sim_averg, MSD_mean_synth_averg], showmeans=True)
    # ax[1, 1].set_ylabel(f'MSD\nPopulation Mean')
    # ax[1, 1].set_xticks([1, 2], ['Simulation', 'Synthetic'])
    # ax[2, 1].violinplot([MSD_mean_synth_sim[map_idx_sim]] + [MSD_mean_synth_sim[i] for i in range(n_sim_plots)],
    #                     showmeans=True)
    # ax[2, 1].set_xticks(np.arange(n_sim_plots + 1) + 1, ['Map'] + ['Simulation'] * n_sim_plots, rotation=60)
    #
    # # ks_statistic, p_value = stats.ks_2samp(TA_mean_synth_sim[map_idx_sim], TA_mean_synth[0])
    # a_test = stats.anderson_ksamp((TA_mean_synth_sim[map_idx_sim], TA_mean_synth[0]), method=stats.PermutationMethod())
    # ks_statistic, p_value = a_test.statistic, a_test.pvalue
    # print(f"Turning Angle KS Statistic: {ks_statistic}")
    # print(f"Turning Angle P-value: {p_value}, {p_value < 0.05}: different distributions")
    # ax[0, 2].violinplot([TA_mean_synth_sim[map_idx_sim], TA_mean_synth[0]], showmeans=True)
    # if p_value < 0.05:
    #     ax[0, 2].set_title(f'Turning Angle\n(Different)')
    # else:
    #     ax[0, 2].set_title(f'Turning Angle\n(Same)')
    # ax[0, 2].set_ylabel(f'Turning Angle (radians)\nMean per Cell')
    # ax[0, 2].set_xticks([1, 2], ['Simulation', 'Synthetic'])
    # ax[1, 2].violinplot([TA_mean_synth_sim_averg, TA_mean_synth_averg], showmeans=True)
    # ax[1, 2].set_ylabel(f'Turning Angle (radians)\nPopulation Mean')
    # ax[1, 2].set_xticks([1, 2], ['Simulation', 'Synthetic'])
    # ax[2, 2].violinplot([TA_mean_synth_sim[map_idx_sim]] + [TA_mean_synth_sim[i] for i in range(n_sim_plots)],
    #                     showmeans=True)
    # ax[2, 2].set_xticks(np.arange(n_sim_plots + 1) + 1, ['Map'] + ['Simulation'] * n_sim_plots, rotation=60)
    #
    # # ks_statistic, p_value = stats.ks_2samp(VEL_mean_synth_sim[map_idx_sim], VEL_mean_synth[0])
    # a_test = stats.anderson_ksamp((VEL_mean_synth_sim[map_idx_sim], VEL_mean_synth[0]),
    #                               method=stats.PermutationMethod())
    # ks_statistic, p_value = a_test.statistic, a_test.pvalue
    # print(f"Velocity KS Statistic: {ks_statistic}")
    # print(f"Velocity P-value: {p_value}, {p_value < 0.05}: different distributions")
    # ax[0, 3].violinplot([VEL_mean_synth_sim[map_idx_sim], VEL_mean_synth[0]], showmeans=True)
    # if p_value < 0.05:
    #     ax[0, 3].set_title(f'Velocity\n(Different)')
    # else:
    #     ax[0, 3].set_title(f'Velocity\n(Same)')
    # ax[0, 3].set_ylabel(f'Velocity\nMean per Cell')
    # ax[0, 3].set_xticks([1, 2], ['Simulation', 'Synthetic'])
    # ax[1, 3].violinplot([VEL_mean_synth_sim_averg, VEL_mean_synth_averg], showmeans=True)
    # ax[1, 3].set_ylabel(f'Velocity\nPopulation Mean')
    # ax[1, 3].set_xticks([1, 2], ['Simulation', 'Synthetic'])
    # ax[2, 3].violinplot([VEL_mean_synth_sim[map_idx_sim]] + [VEL_mean_synth_sim[i] for i in range(n_sim_plots)],
    #                     showmeans=True)
    # ax[2, 3].set_xticks(np.arange(n_sim_plots + 1) + 1, ['Map'] + ['Simulation'] * n_sim_plots, rotation=60)
    #
    # # ks_statistic, p_value = stats.ks_2samp(WT_mean_synth_sim[map_idx_sim], WT_mean_synth[0])
    # a_test = stats.anderson_ksamp((WT_mean_synth_sim[map_idx_sim], WT_mean_synth[0]), method=stats.PermutationMethod())
    # ks_statistic, p_value = a_test.statistic, a_test.pvalue
    # print(f"Waiting Time KS Statistic: {ks_statistic}")
    # print(f"Waiting Time P-value: {p_value}, {p_value < 0.05}: different distributions")
    # ax[0, 4].violinplot([WT_mean_synth_sim[map_idx_sim], WT_mean_synth[0]], showmeans=True)
    # if p_value < 0.05:
    #     ax[0, 4].set_title(f'Waiting Time\n(Different)')
    # else:
    #     ax[0, 4].set_title(f'Waiting Time\n(Same)')
    # ax[0, 4].set_ylabel(f'Waiting Time (sec)\nMean per Cell')
    # ax[0, 4].set_xticks([1, 2], ['Simulation', 'Synthetic'])
    # ax[1, 4].violinplot([WT_mean_synth_sim_averg, WT_mean_synth_averg], showmeans=True)
    # ax[1, 4].set_ylabel(f'Waiting Time (sec)\nPopulation Mean')
    # ax[1, 4].set_xticks([1, 2], ['Simulation', 'Synthetic'])
    # ax[2, 4].violinplot([WT_mean_synth_sim[map_idx_sim]] + [WT_mean_synth_sim[i] for i in range(n_sim_plots)],
    #                     showmeans=True)
    # ax[2, 4].set_xticks(np.arange(n_sim_plots + 1) + 1, ['Map'] + ['Simulation'] * n_sim_plots, rotation=60)
    # plt.savefig(f'{checkpoint_path}/Summary Stats.png')
    # plt.show()
