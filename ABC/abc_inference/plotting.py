"""
Publication-style corner plot of the ABC posterior samples in physical units,
matching the style of the NPE posterior plots (SBI plotting_temporal_cl.py).

Diagonal panels show a boundary-reflected 1D KDE with the 68% interval shaded
and the median ± 1σ in the title; lower-triangle panels show filled credible
regions of a smoothed 2D histogram. Mass-loss rates are shown as log10 on a
linear axis.
"""

# general
import warnings

# numerics
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# local
from abc_inference.units import LAST_MASS_LOSS_RATE_INDEX, convert_to_phys


# -------------------------------------------------------------
# =================== ↓ Visual style ↓ ========================
# -------------------------------------------------------------

POSTERIOR_COLOR = "#2E6FB0"
DARK_ACCENT_COLOR = "#16324F"  # inner contours and median line
TRUTH_COLOR = "#D1495B"
SPINE_COLOR = "#C9D3DE"
TEXT_COLOR = "#2B2B2B"

POSTERIOR_COLORMAP = LinearSegmentedColormap.from_list(
    "post_blue",
    ["#ffffff", "#dcebf8", "#9cc2e3", POSTERIOR_COLOR, DARK_ACCENT_COLOR],
)

# Enclosed probability mass of the drawn 2D credible regions, corresponding to
# 0.5, 1, 1.5 and 2 sigma of a 2D Gaussian.
CREDIBLE_REGION_MASSES = [0.118, 0.393, 0.675, 0.864]

MEDIAN_LINESTYLE = (0, (4, 3))

PARAMETER_PLOT_LABELS = [
    r"$\log \dot M_1$",
    r"$\log \dot M_2$",
    r"$v_{\inf,1}$",
    r"$v_{\inf,2}$",
    r"$e$",
    r"$\cos (i)$",
    r"$\eta$",
]

# The turbulence strength η (index 6) is left out of the posterior plots by
# default, as in the NPE posterior plots.
DEFAULT_PLOTTED_PARAMETER_INDICES = [0, 1, 2, 3, 4, 5]

PLOT_RC_PARAMS = {
    "axes.labelsize": 17,
    "axes.titlesize": 17,
    "legend.fontsize": 15,
}

# -------------------------------------------------------------
# =================== ↑ Visual style ↑ ========================
# -------------------------------------------------------------


def _density_levels(histogram, enclosed_masses):
    """
    Iso-density levels of a (smoothed) 2D histogram that enclose the given
    probability masses.

    Args:
        histogram: The 2D histogram.
        enclosed_masses: Probability masses the levels should enclose.

    Returns:
        Sorted, unique, positive levels strictly below the histogram maximum
        (so that every level yields a non-empty filled contour).
    """
    sorted_densities = np.sort(histogram.ravel())[::-1]
    cumulative_mass = np.cumsum(sorted_densities)
    if cumulative_mass[-1] <= 0:
        return []
    cumulative_mass /= cumulative_mass[-1]

    levels = []
    for mass in enclosed_masses:
        index = min(np.searchsorted(cumulative_mass, mass), len(sorted_densities) - 1)
        levels.append(sorted_densities[index])

    levels = sorted(set(float(level) for level in levels if level > 0))
    maximum_density = float(histogram.max())
    return [level for level in levels if level < maximum_density]


def _kde_1d(values, lower, upper, num_grid_points=400):
    """
    Boundary-reflected 1D KDE evaluated on [lower, upper].

    Reflecting the samples at both prior boundaries avoids the artificial
    drop-off a plain KDE shows at hard edges. The bandwidth is rescaled so it
    corresponds to the original sample size rather than the tripled one.

    Args:
        values: The 1D samples.
        lower: Lower boundary of the evaluation interval.
        upper: Upper boundary of the evaluation interval.
        num_grid_points: Number of evaluation points.

    Returns:
        (grid, density), or (None, None) if there are too few distinct samples.
    """
    values = values[np.isfinite(values)]
    if values.size < 5 or np.ptp(values) == 0:
        return None, None

    reflected = np.concatenate([values, 2 * lower - values, 2 * upper - values])
    kde = gaussian_kde(reflected)
    kde.set_bandwidth(kde.factor * (reflected.size / values.size) ** (1.0 / 5.0))

    grid = np.linspace(lower, upper, num_grid_points)
    return grid, kde(grid)


def _format_value(value):
    """Format a value with precision adapted to its magnitude."""
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _to_display_units(values, parameter_index):
    """Physical units, with mass-loss rates as log10(M_sun / yr)."""
    values = np.asarray(convert_to_phys(values, parameter_index), dtype=float)
    if parameter_index <= LAST_MASS_LOSS_RATE_INDEX:
        values = np.log10(np.clip(values, 1e-300, None))
    return values


def _plot_1d_marginal(ax, values, limits, truth, label):
    """Diagonal panel: KDE, shaded 68% interval, median and truth."""
    grid, density = _kde_1d(values, *limits)
    if grid is not None:
        ax.fill_between(grid, density, color=POSTERIOR_COLOR, alpha=0.12, lw=0)

        quantile_16, median, quantile_84 = np.percentile(values, [16, 50, 84])
        in_68_interval = (grid >= quantile_16) & (grid <= quantile_84)
        ax.fill_between(
            grid[in_68_interval],
            density[in_68_interval],
            color=POSTERIOR_COLOR,
            alpha=0.30,
            lw=0,
        )
        ax.plot(grid, density, color=POSTERIOR_COLOR, lw=2.0)
        ax.set_ylim(0, density.max() * 1.18)

        upper_error = _format_value(quantile_84 - median)
        lower_error = _format_value(median - quantile_16)
        ax.set_title(
            rf"{label} $= {_format_value(median)}^{{+{upper_error}}}_{{-{lower_error}}}$",
            fontsize=14,
            color=TEXT_COLOR,
            pad=6,
        )
        ax.axvline(median, color=DARK_ACCENT_COLOR, ls=MEDIAN_LINESTYLE, lw=1.3, alpha=0.9)

    if truth is not None:
        ax.axvline(truth, color=TRUTH_COLOR, lw=1.8)

    ax.set_yticks([])
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(SPINE_COLOR)


def _plot_2d_marginal(ax, x_values, y_values, x_limits, y_limits, x_truth, y_truth,
                      bins, fill_colors, color_option):
    """Lower-triangle panel: filled credible regions and the truth marker."""
    try:
        histogram, x_edges, y_edges = np.histogram2d(
            x_values,
            y_values,
            bins=bins,
            range=[x_limits, y_limits],
        )
        histogram = gaussian_filter(histogram, 1.25)
        levels = _density_levels(histogram, CREDIBLE_REGION_MASSES)

        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        mesh_x, mesh_y = np.meshgrid(x_centers, y_centers)

        if color_option:
            ax.pcolormesh(
                x_edges,
                y_edges,
                histogram.T,
                cmap=POSTERIOR_COLORMAP,
                shading="auto",
                alpha=0.85,
            )
        if levels:
            # Close the innermost region slightly above the peak.
            filled_levels = levels + [histogram.max() * 1.0001]
            ax.contourf(
                mesh_x,
                mesh_y,
                histogram.T,
                levels=filled_levels,
                colors=fill_colors[-(len(filled_levels) - 1):],
                antialiased=True,
            )
            ax.contour(
                mesh_x,
                mesh_y,
                histogram.T,
                levels=levels,
                colors=[DARK_ACCENT_COLOR],
                linewidths=0.6,
                alpha=0.55,
            )
    except Exception as exception:
        warnings.warn(f"2D panel fell back to a scatter plot: {exception}")
        ax.scatter(x_values, y_values, s=2, alpha=0.15, color=POSTERIOR_COLOR, lw=0)

    if x_truth is not None and y_truth is not None:
        ax.axvline(x_truth, color=TRUTH_COLOR, lw=1.0, alpha=0.65)
        ax.axhline(y_truth, color=TRUTH_COLOR, lw=1.0, alpha=0.65)
        ax.plot(
            x_truth,
            y_truth,
            marker="s",
            ms=6,
            mfc=TRUTH_COLOR,
            mec="white",
            mew=0.8,
            zorder=5,
        )

    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)


def plot_marginals(
    samples,
    true_values_phys,
    labels=None,
    savefile=None,
    bins=80,
    prior_ranges=None,
    parameter_indices=None,
    figsize=(15, 15),
    color_option=False,
):
    """
    Corner plot of the posterior samples in physical units.

    Args:
        samples: Posterior samples of shape [N, D] in code units.
        true_values_phys: True parameter values in physical units (mass-loss
            rates in M_sun / yr, not log), one per parameter in theta, or None.
        labels: Axis labels, one per plotted parameter. Defaults to the
            entries of PARAMETER_PLOT_LABELS for the plotted parameters.
        savefile: If given, the figure is saved to this path (dpi 300).
        bins: Resolution of the 2D histograms.
        prior_ranges: Optional list of (low, high) per parameter in code units,
            used as axis limits. Without it the sample range is used.
        parameter_indices: Indices into theta of the parameters to plot.
            Defaults to DEFAULT_PLOTTED_PARAMETER_INDICES (all but η).
        figsize: Figure size in inches.
        color_option: If True, also paint a faint continuous density behind the
            filled credible regions.

    Returns:
        (fig, axes).
    """
    if parameter_indices is None:
        parameter_indices = DEFAULT_PLOTTED_PARAMETER_INDICES
    num_plotted = len(parameter_indices)
    if labels is None:
        labels = [PARAMETER_PLOT_LABELS[index] for index in parameter_indices]

    # -------------------------------------------------------------
    # ============ ↓ Conversion to display units ↓ ================
    # -------------------------------------------------------------

    display_samples = [
        _to_display_units(samples[:, index], index)
        for index in parameter_indices
    ]

    truths = [None] * num_plotted
    if true_values_phys is not None:
        truths = []
        for index in parameter_indices:
            truth = float(true_values_phys[index])
            if index <= LAST_MASS_LOSS_RATE_INDEX:
                truth = float(np.log10(max(truth, 1e-300)))
            truths.append(truth)

    if prior_ranges is not None:
        limits = [
            [float(_to_display_units(bound, index)) for bound in prior_ranges[index]]
            for index in parameter_indices
        ]
    else:
        limits = [
            [float(values.min()), float(values.max())]
            for values in display_samples
        ]

    # -------------------------------------------------------------
    # ============ ↑ Conversion to display units ↑ ================
    # -------------------------------------------------------------

    fill_colors = [
        POSTERIOR_COLORMAP(value)
        for value in np.linspace(0.20, 0.95, len(CREDIBLE_REGION_MASSES))
    ]

    with plt.rc_context(PLOT_RC_PARAMS):
        fig, axes = plt.subplots(num_plotted, num_plotted, figsize=figsize, squeeze=False)
        fig.patch.set_facecolor("white")
        plt.subplots_adjust(wspace=0.06, hspace=0.06)

        for row in range(num_plotted):
            for column in range(num_plotted):
                ax = axes[row, column]
                ax.set_facecolor("white")

                if row == column:
                    _plot_1d_marginal(
                        ax,
                        display_samples[row],
                        limits[row],
                        truths[row],
                        labels[row],
                    )
                elif row > column:
                    _plot_2d_marginal(
                        ax,
                        display_samples[column],
                        display_samples[row],
                        limits[column],
                        limits[row],
                        truths[column],
                        truths[row],
                        bins,
                        fill_colors,
                        color_option,
                    )
                else:
                    ax.axis("off")

                # Shared limits and sparse linear ticks on the visible panels.
                if row >= column:
                    ax.set_xlim(limits[column])
                    ax.xaxis.set_major_locator(MaxNLocator(4, prune="both"))
                    if row > column:
                        ax.set_ylim(limits[row])
                        ax.yaxis.set_major_locator(MaxNLocator(4, prune="both"))
                    ax.tick_params(
                        direction="in",
                        length=3.5,
                        width=0.8,
                        colors=TEXT_COLOR,
                        labelsize=11.5,
                        top=False,
                        right=False,
                    )

                # Tick labels and axis labels only on the outer edges.
                if row < num_plotted - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(labels[column])
                    for tick_label in ax.get_xticklabels():
                        tick_label.set_rotation(30)
                        tick_label.set_ha("right")
                if column > 0 or row == 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel(labels[row])

        # The legend goes into the empty upper-right cell.
        if num_plotted >= 2:
            legend_handles = [
                Patch(facecolor=POSTERIOR_COLOR, alpha=0.55, edgecolor=POSTERIOR_COLOR, label="Posterior"),
                Line2D([0], [0], color=TRUTH_COLOR, lw=2, marker="s", mfc=TRUTH_COLOR,
                       mec="white", ms=7, label="True value"),
                Line2D([0], [0], color=DARK_ACCENT_COLOR, lw=1.3, ls=MEDIAN_LINESTYLE, label="Median"),
            ]
            legend_ax = axes[0, num_plotted - 1]
            legend_ax.axis("off")
            legend_ax.legend(
                handles=legend_handles,
                loc="center",
                frameon=False,
                fontsize=13,
                handlelength=1.6,
                borderaxespad=0.2,
            )

        fig.align_labels()
        if savefile:
            fig.savefig(savefile, bbox_inches="tight", dpi=300)

    return fig, axes
