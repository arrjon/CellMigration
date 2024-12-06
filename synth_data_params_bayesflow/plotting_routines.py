import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from summary_stats import reduced_coordinates_to_sumstat, compute_mean_summary_stats, \
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


def plot_sumstats_distance_hist(obj_func_wass: callable, test_sim_dict: dict, sumstats_list: list[list],
                                weights: list = None,
                                labels: list[str] = None, colors: list[str] = None, path: str = None):
    """
    Plot the distribution of the wasserstein distance between the summary statistics of the test simulation and the
    simulations.
    """
    if labels is not None:
        if len(labels) != len(sumstats_list):
            raise ValueError("The number of labels should be equal to the number of summary statistics lists")

    marginal_distances_list = []
    for sumstats in sumstats_list:
        # compute the distance for each hand-crafted summary statistics
        marginal_distances = np.zeros((len(sumstats), len(test_sim_dict.keys())))
        for i, st in enumerate(sumstats):
            marginal_distances[i] = obj_func_wass(test_sim_dict, st, return_marginal=True, normalize=True)

        marginal_distances_list.append(marginal_distances)

    fig, ax = plt.subplots(1, marginal_distances_list[0].shape[1], figsize=(10, 3), tight_layout=True)
    name_plots = ['Angle Degree\n', 'Mean Squared\nDisplacement', 'Turning Angle\n', 'Velocity\n', 'Waiting Time\n']

    for i in range(marginal_distances_list[0].shape[1]):
        for j, marginal_distances in enumerate(marginal_distances_list):
            ax[i].hist(marginal_distances[:, i], bins=15, weights=weights, alpha=0.5, density=True,
                       label=labels[j] if labels is not None and i == 0 else None,
                       color=colors[j] if colors is not None else None)
        ax[i].set_title(name_plots[i])
        ax[i].set_xlabel('Normalized\nWasserstein Distance')
    ax[0].set_ylabel('Density')
    if labels is not None:
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15),
                   ncol=len(labels) // 2 if len(labels) > 3 else len(labels))
    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    return


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

    # plot the simulations or the MAP
    if show_image and two_plots:
        tiff_im = plt.imread('Cell_migration_grid_v3_final2_invers.tiff')
        ax[1].imshow(tiff_im)
    ax[1].plot(posterior_sim[0, :, 0], posterior_sim[0, :, 1], 'b', label='Simulated Trajectories', alpha=1)
    for cell_id in range(1, cells_in_population):
        ax[1].plot(posterior_sim[cell_id, :, 0], posterior_sim[cell_id, :, 1], 'b')
        ax[1].scatter(posterior_sim[cell_id, :, 0], posterior_sim[cell_id, :, 1], s=10, color='blue')

    # plot the synthetic data
    if show_image:
        tiff_im = plt.imread('Cell_migration_grid_v3_final2_invers.tiff')
        ax[0].imshow(tiff_im)
    if label_true is None:
        label_true = 'Synthetic Trajectories'
    ax[0].plot(test_sim[0, :, 0], test_sim[0, :, 1], 'r', label=label_true, alpha=1)
    for cell_id in range(1, cells_in_population):
        ax[0].plot(test_sim[cell_id, :, 0], test_sim[cell_id, :, 1], 'r')
        ax[0].scatter(test_sim[cell_id, :, 0], test_sim[cell_id, :, 1], s=10, color='red')

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


def calculate_ci(
    values: np.ndarray,
    ci_level: float,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate confidence/credibility levels using percentiles.

    Parameters
    ----------
    values:
        The values used to calculate percentiles.
    ci_level:
        Lower tail probability.
    kwargs:
        Additional keyword arguments are passed to the `numpy.percentile` call.

    Returns
    -------
    lb, ub:
        Bounds of the confidence/credibility interval.
    """
    # Percentile values corresponding to the CI level
    percentiles = 100 * np.array([(1 - ci_level) / 2, 1 - (1 - ci_level) / 2])
    # Upper and lower bounds
    lb, ub = np.percentile(values, percentiles, **kwargs)
    return lb, ub


def sampling_parameter_cis(
    posterior_samples: np.ndarray,
    true_param: np.ndarray = None,
    param_names: list[str] = None,
    alpha: list[int] = None,
    step: float = 0.05,
    show_median: bool = True,
    title: str = None,
    size: tuple[float, float] = None,
    ax: matplotlib.axes.Axes = None,
    legend_bbox_to_anchor: tuple[float, float] = (1, 1),
) -> matplotlib.axes.Axes:
    """
    Plot MCMC-based parameter credibility intervals. Function adapted from pyPESTO.
    """
    if alpha is None:
        alpha = [95]

    # automatically sort values in decreasing order
    alpha_sorted = sorted(alpha, reverse=True)
    # define colormap
    evenly_spaced_interval = np.linspace(0, 1, len(alpha_sorted))
    colors = [plt.cm.tab20c_r(x) for x in evenly_spaced_interval]
    # number of sampled parameters
    n_pars = posterior_samples.shape[-1]

    # set axes and figure
    if ax is None:
        _, ax = plt.subplots(figsize=size, tight_layout=True)

    # loop over parameters
    for npar in range(n_pars):
        # initialize height of boxes
        _step = step
        # loop over confidence levels
        for n, level in enumerate(alpha_sorted):
            # extract percentile-based confidence intervals
            lb, ub = calculate_ci(posterior_samples, ci_level=level / 100, axis=0)

            # assemble boxes for projectile plot
            x1 = [lb[npar], ub[npar]]
            y1 = [npar + _step, npar + _step]
            y2 = [npar - _step, npar - _step]
            # Plot boxes
            ax.fill(
                np.append(x1, x1[::-1]),
                np.append(y1, y2[::-1]),
                color=colors[n],
                label=str(level) + "% CI",
            )

            if show_median:
                if n == len(alpha_sorted) - 1:
                    _median = np.median(posterior_samples[:, npar])
                    ax.plot(
                        [_median, _median],
                        [npar - _step, npar + _step],
                        "k-",
                        label="Median",
                    )
            if true_param is not None:
                if n == len(alpha_sorted) - 1:
                    ax.plot(
                        [true_param[npar], true_param[npar]],
                        [npar - _step, npar + _step],
                        linestyle="--",
                        color="black",
                        label="True Parameter",
                    )

            # increment height of boxes
            _step += step

    ax.set_yticks(range(n_pars))
    if param_names is not None:
        ax.set_yticklabels(param_names)
    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Parameter name")

    if title:
        ax.set_title(title)

    # handle legend
    plt.gca().invert_yaxis()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=legend_bbox_to_anchor)

    return ax


def plot_posterior_2d(
    posterior_draws,
    prior=None,
    prior_draws=None,
    param_names=None,
    true_params=None,
    height=3,
    label_fontsize=14,
    legend_fontsize=16,
    tick_fontsize=12,
    post_color="#8f2727",
    prior_color="gray",
    post_alpha=0.9,
    prior_alpha=0.7,
):
    """Generates a bivariate pairplot given posterior draws and optional prior or prior draws.
    Function adapted from BayesFlow.

    posterior_draws   : np.ndarray of shape (n_post_draws, n_params)
        The posterior draws obtained for a SINGLE observed data set.
    prior             : bayesflow.forward_inference.Prior instance or None, optional, default: None
        The optional prior object having an input-output signature as given by bayesflow.forward_inference.Prior
    prior_draws       : np.ndarray of shape (n_prior_draws, n_params) or None, optional (default: None)
        The optional prior draws obtained from the prior. If both prior and prior_draws are provided, prior_draws
        will be used.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    true_params: np.ndarray of shape (n_params,) or None, optional, default: None
    height            : float, optional, default: 3
        The height of the pairplot
    label_fontsize    : int, optional, default: 14
        The font size of the x and y-label texts (parameter names)
    legend_fontsize   : int, optional, default: 16
        The font size of the legend text
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    post_color        : str, optional, default: '#8f2727'
        The color for the posterior histograms and KDEs
    priors_color      : str, optional, default: gray
        The color for the optional prior histograms and KDEs
    post_alpha        : float in [0, 1], optonal, default: 0.9
        The opacity of the posterior plots
    prior_alpha       : float in [0, 1], optonal, default: 0.7
        The opacity of the prior plots

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    AssertionError
        If the shape of posterior_draws is not 2-dimensional.
    """

    # Ensure correct shape
    assert (
        len(posterior_draws.shape)
    ) == 2, "Shape of `posterior_samples` for a single data set should be 2 dimensional!"

    # Obtain n_draws and n_params
    n_draws, n_params = posterior_draws.shape

    # If prior object is given and no draws, obtain draws
    if prior is not None and prior_draws is None:
        draws = prior(n_draws)
        if type(draws) is dict:
            prior_draws = draws["prior_draws"]
        else:
            prior_draws = draws
    # Otherwise, keep as is (prior_draws either filled or None)
    else:
        pass

    # Attempt to determine parameter names
    if param_names is None:
        if hasattr(prior, "param_names"):
            if prior.param_names is not None:
                param_names = prior.param_names
            else:
                param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]
        else:
            param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]

    # Pack posterior draws into a dataframe
    posterior_draws_df = pd.DataFrame(posterior_draws, columns=param_names)

    # Add posterior
    g = sns.PairGrid(posterior_draws_df, height=height)
    g.map_diag(sns.histplot, fill=True, color=post_color, alpha=post_alpha, kde=True)
    g.map_lower(sns.kdeplot, fill=True, color=post_color, alpha=post_alpha)

    # Add prior, if given
    if prior_draws is not None:
        prior_draws_df = pd.DataFrame(prior_draws, columns=param_names)
        g.data = prior_draws_df
        g.map_diag(sns.histplot, fill=True, color=prior_color, alpha=prior_alpha, kde=True, zorder=-1)
        g.map_lower(sns.kdeplot, fill=True, color=prior_color, alpha=prior_alpha, zorder=-1)

    # Custom function to plot true_params on the diagonal
    if true_params is not None:
        def plot_true_params(x, **kwargs):
            param = x.iloc[0]  # Get the single true value for the diagonal
            plt.axvline(param, color="black", linestyle="--")  # Add vertical line

        # only plot on the diagonal a vertical line for the true parameter
        g.data = pd.DataFrame(true_params[np.newaxis], columns=param_names)
        g.map_diag(plot_true_params)

    # Add legend, if prior also given
    if prior_draws is not None or prior is not None:
        handles = [
            Line2D(xdata=[], ydata=[], color=post_color, lw=3, alpha=post_alpha),
            Line2D(xdata=[], ydata=[], color=prior_color, lw=3, alpha=prior_alpha),
        ]
        labels = ["Posterior", "Prior"]
        if true_params is not None:
            handles.append(Line2D(xdata=[], ydata=[], color="black", linestyle="--"))
            labels.append("True Parameter")
        g.fig.legend(handles, labels, fontsize=legend_fontsize, loc="center right")

    # Remove upper axis
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].axis("off")

    # Modify tick sizes
    for i, j in zip(*np.tril_indices_from(g.axes, 1)):
        g.axes[i, j].tick_params(axis="both", which="major", labelsize=tick_fontsize)
        g.axes[i, j].tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Add nice labels
    for i, param_name in enumerate(param_names):
        g.axes[i, 0].set_ylabel(param_name, fontsize=label_fontsize)
        g.axes[len(param_names) - 1, i].set_xlabel(param_name, fontsize=label_fontsize)

    # Add grids
    for i in range(n_params):
        for j in range(n_params):
            g.axes[i, j].grid(alpha=0.5)

    g.tight_layout()
    return g.fig