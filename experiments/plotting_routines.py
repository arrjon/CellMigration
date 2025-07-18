import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from summary_stats import compute_summary_stats, compute_autocorrelation


def plot_sumstats_distance_stats(obj_func_comparison: callable,
                                 test_sim_dict: dict,
                                 sumstats_list: list[list],
                                 labels: list[str] = None,
                                 colors: list[str] = None,
                                 ylog_scale: bool = False,
                                 title: str = 'Weighted\nWasserstein Distance',
                                 path: str = None):
    """
    Plot boxplots of the distance between each simulation method and the test simulation,
    separately for each summary‐stat category.
    """
    # 1) compute marginal distances for each summary‐stat group
    marginal_distances_list = []
    for sumstats in sumstats_list:
        # shape = (n_methods, n_sim, n_stats_in_group)
        md = np.zeros((len(sumstats), len(test_sim_dict)))
        for i, st in enumerate(sumstats):
            md[i, :] = obj_func_comparison(test_sim_dict, st, return_marginal=True)
        marginal_distances_list.append(md)
    marginal_distances_list = np.array(marginal_distances_list)

    n_stats = len(test_sim_dict)
    n_methods  = len(sumstats_list)
    name_plots = ['Displacement', 'Turning Angle', 'Velocity', 'Angle Degree', 'NPE Summary', 'UMAP']
    order     = [0, 2, 3, 1, 4, 5]  # reorder

    # 2) make one subplot per summary‐stat category
    fig, axes = plt.subplots(1, n_stats, figsize=(10, 2), layout="constrained")
    if n_stats == 1:
        axes = [axes]

    for ax, grp_idx in zip(axes, order):
        data = marginal_distances_list[:, :, grp_idx]   # (n_methods, n_sim, n_stats)
        # boxplot: one array per method
        b = ax.boxplot(data.T, patch_artist=True)

        # color each box
        for patch, col in zip(b['boxes'], colors):
            patch.set_facecolor(col)

        ax.set_xlabel(name_plots[grp_idx], fontsize=12)
        ax.set_xticks(np.arange(1, n_methods+1))
        ax.set_xticklabels([], rotation=45, ha='right')
        if ylog_scale and not grp_idx == 5:
            ax.set_yscale('log')
        if grp_idx < 4:
            ax.set_ylim(0.0001, 20)
            if grp_idx != 0:
                ax.set_yticks([])
        if grp_idx == 4:
            ax.set_ylim(0.1, 20)

    axes[0].set_ylabel(title, fontsize=12)
    axes[-2].set_ylabel("L1 Distance", fontsize=12)
    axes[-1].set_ylabel("Cosine Similarity", fontsize=12)
    for a in axes:
        a.tick_params(axis="x", labelsize=10)
        a.tick_params(axis="y", labelsize=10)

    # 3) shared legend
    handles = [Patch(facecolor=c, label=l) for c, l in zip(colors, labels)]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.33), fontsize=12)

    # 4) save & show
    if path:
        plt.savefig(path, bbox_inches='tight')
    plt.show()


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

    msd_list, ta_list, v_list, ad_list = compute_summary_stats(cell_population)
    stats = [[msd_list, ta_list, v_list, ad_list]]
    if cell_population_2 is not None:
        msd_list_2, ta_list_2, v_list_2, ad_list_2 = compute_summary_stats(cell_population_2)
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
    prior_bounds: dict = None,
    alpha: list[int] = None,
    color_list: list[str] = None,
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
        _, ax = plt.subplots(figsize=size, layout='constrained')

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
                color=colors[n] if color_list is None else color_list[npar],
                label=str(level) + "% CI",
                alpha=None if color_list is None else 1 / (len(alpha_sorted) - n),
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
                        linestyle="-",
                        color="black",
                        label="True Parameter",
                    )

            # increment height of boxes
            _step += step

    if prior_bounds is not None:
        for i, bounds in enumerate(prior_bounds):
            ax.vlines(x=bounds[0], ymin=0 + (i * 4), ymax=3 + (i * 4), color='gray', linestyle='--', linewidth=0.8)
            ax.vlines(x=bounds[1], ymin=0 + (i * 4), ymax=3 + (i * 4), color='gray', linestyle='--', linewidth=0.8,
                      label='Uniform Prior' if i == 0 else None)

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
    if color_list is not None:
        handles_new = []
        labels_new = labels[:len(alpha_sorted)]
        for i in range(len(alpha_sorted)):
            handles_new.append(Patch(color='grey', alpha=1 / (len(alpha_sorted) - i), label=labels_new[i]))

        show_last = 0
        if true_param is not None or show_median:
            show_last += 1
            handles_new.append(handles[-show_last])
            labels_new.append(labels[-show_last])
        if prior_bounds is not None:
            show_last += 1
            handles_new.append(handles[-show_last])
            labels_new.append(labels[-show_last])
        ax.legend(handles=handles_new, labels=labels_new, bbox_to_anchor=legend_bbox_to_anchor)
    else:
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=legend_bbox_to_anchor)

    return ax


def plot_posterior_2d(
    posterior_draws,
    prior=None,
    prior_draws=None,
    param_names=None,
    true_params=None,
    reference_params=None,
    height=2,
    label_fontsize=16,
    legend_fontsize=18,
    tick_fontsize=12,
    bins="auto",
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
    reference_params: np.ndarray of shape (n_params,) or None, optional, default: None
    height            : float, optional, default: 3
        The height of the pairplot
    label_fontsize    : int, optional, default: 12
        The font size of the x and y-label texts (parameter names)
    legend_fontsize   : int, optional, default: 16
        The font size of the legend text
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    bins              : int or 'auto' or None, optional, default: 'auto'
        The number of bins for the histograms. If None, the default from seaborn is used
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
    g.map_diag(sns.histplot, fill=True, color=post_color, alpha=post_alpha, bins=bins, kde=True)
    g.map_lower(sns.kdeplot, fill=True, color=post_color, alpha=post_alpha)

    # Add prior, if given
    if prior_draws is not None:
        prior_draws_df = pd.DataFrame(prior_draws, columns=param_names)
        g.data = prior_draws_df
        g.map_diag(sns.histplot, fill=True, color=prior_color, alpha=prior_alpha, bins=bins, kde=True, zorder=-1)
        g.map_lower(sns.kdeplot, fill=True, color=prior_color, alpha=prior_alpha, zorder=-1)

    # Custom function to plot true_params on the diagonal
    if true_params is not None:
        def plot_true_params(x, **kwargs):
            param = x.iloc[0]  # Get the single true value for the diagonal
            plt.axvline(param, color="red", linestyle="--")  # Add vertical line

        # only plot on the diagonal a vertical line for the true parameter
        g.data = pd.DataFrame(true_params[np.newaxis], columns=param_names)
        g.map_diag(plot_true_params)

    if reference_params is not None:
        def plot_ref_params(x, **kwargs):
            param = x.iloc[0]  # Get the single true value for the diagonal
            plt.axvline(param, color="black", linestyle="--")  # Add vertical line

        # only plot on the diagonal a vertical line for the true parameter
        g.data = pd.DataFrame(reference_params[np.newaxis], columns=param_names)
        g.map_diag(plot_ref_params)

    # Add legend, if prior also given
    if prior_draws is not None or prior is not None:
        handles = [
            Line2D(xdata=[], ydata=[], color=post_color, lw=3, alpha=post_alpha),
            Line2D(xdata=[], ydata=[], color=prior_color, lw=3, alpha=prior_alpha),
        ]
        labels = ["Posterior", "Prior"]
        if true_params is not None:
            handles.append(Line2D(xdata=[], ydata=[], color="red", linestyle="--"))
            labels.append("True Parameter")
        if reference_params is not None:
            handles.append(Line2D(xdata=[], ydata=[], color="black", linestyle="--"))
            labels.append("Summary Network")
        g.fig.legend(handles, labels, fontsize=legend_fontsize, loc="lower center", bbox_to_anchor=(0.5, -0.1))

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


def plot_posterior_1d(
    posterior_samples,
    prior_draws,
    log_param_names,
    test_sim, test_params,
    labels_colors,
    make_sumstat_dict_nn=None,
    height=1.,
    save_path=None,
):
    """
    Generates horizontal 1D marginal density and histogram plots for each parameter from posterior draws,
    with optional prior, true, and reference parameters, and places the legend to the right.
    """

    # -- your existing helper to get reference_params for abc_mean, etc. --
    def get_reference(name, test_sim):
        if make_sumstat_dict_nn is None:
            return None
        if name == 'abc_mean':
            return make_sumstat_dict_nn(test_sim, use_npe_summaries=False)['summary_net']
        elif name == 'npe':
            return make_sumstat_dict_nn(test_sim, use_npe_summaries=True, return_reduced=True)['summary_pred']
        else:
            return None

    # collect all names with non-None samples
    plot_items = [(name, ps) for name, ps in posterior_samples.items() if ps is not None]
    n_rows = len(plot_items)
    # assume all ps have same number of params
    n_params = plot_items[0][1].shape[1]

    # plotting grid
    fig, axes = plt.subplots(
        n_rows, n_params,
        figsize=(10, height * n_rows),
        sharex='col',
        sharey=True,  # 'col'
        layout='constrained'
    )

    handles = []
    for i, (name, ps) in enumerate(plot_items):
        reference_params = get_reference(name, test_sim)
        post_df = pd.DataFrame(ps, columns=log_param_names)
        prior_df = pd.DataFrame(prior_draws, columns=log_param_names)

        for j, pname in enumerate(log_param_names):
            ax = axes[i, j]

            # posterior
            sns.histplot(
                post_df[pname], ax=ax,
                bins='auto', kde=True, stat='probability',
                color=labels_colors[name][1], alpha=1, fill=True
            )

            # prior
            sns.histplot(
                prior_df[pname], ax=ax,
                bins=15, kde=False, stat='probability',
                color="gray", alpha=0.7, fill=True, zorder=-1
            )

            # true & reference lines
            if test_params is not None:
                ax.axvline(test_params[j], color="red", linestyle="--")
            if reference_params is not None:
                ax.axvline(reference_params[j], color="black", linestyle="--")

            # styling
            ax.set_xlabel(pname, fontsize=12)
            if j == 0:
                ax.set_ylabel("Density", fontsize=12)
            else:
                ax.set_ylabel("")
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)
            ax.grid(alpha=0.5)
            ax.set_xlim(prior_df[pname].values.min(), prior_df[pname].values.max())

        handles.append(
            Line2D([], [], color=labels_colors[name][1], lw=3, alpha=1)
        )

    # shared legend below all rows
    if test_params is not None:
        handles = [
                      Line2D([], [], color="red", linestyle="--"),
                      Line2D([], [], color="black", linestyle="--"),
                      Line2D([], [], color="gray", lw=3, alpha=0.7),
                  ] + handles
        labels = ["True Parameter", "Summary Net Prediction", "Prior"] + [val[0] for val in labels_colors.values()]
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=12,
                   bbox_to_anchor=(0.5, -0.23), ncols=3)
    else:
        handles += [Line2D([], [], color="gray", lw=3, alpha=0.7)]
        labels = [val[0] for val in labels_colors.values()] + ["Prior"]
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=12,
                   bbox_to_anchor=(0.5, -0.18), ncols=3)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    return fig
