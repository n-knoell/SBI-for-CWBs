"""
Generic image plots of photon-count maps: a grid of images and a posterior
predictive grid of simulations from posterior samples.
"""

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def image_grid(images, ncols=6, figsize=None, cmap='viridis', vmin=None, vmax=None, titles=None):
    """
    Show 2D images in a grid with logarithmic color scale and one colorbar.

    Args:
        images: A (H, W) image or an (N, H, W) array / list of images.
        ncols: Maximum number of columns.
        figsize: Figure size; defaults to 2.2 inches per panel.
        cmap: Colormap.
        vmin, vmax: Color limits.
        titles: Optional panel titles.

    Returns:
        (fig, flattened axes).
    """
    images = np.asarray(images)
    if images.ndim == 2:
        images = images[None, ...]
    num_images = images.shape[0]
    ncols = min(ncols, num_images)
    nrows = int(np.ceil(num_images / ncols))
    if figsize is None:
        figsize = (2.2 * ncols, 2.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for index, ax in enumerate(axes):
        ax.axis('off')
        if index < num_images:
            image = ax.imshow(images[index], origin='lower', cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
            if titles is not None:
                ax.set_title(titles[index])

    fig.subplots_adjust(right=0.85)
    colorbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    plt.colorbar(image, cax=colorbar_ax)
    return fig, axes


def posterior_predictive_grid(posterior_samples, simulator_fn, nplot=9, ncols=3, random_seed=None):
    """
    Posterior predictive check: simulate a few random posterior samples and
    show the resulting maps.

    Args:
        posterior_samples: Samples of shape (N, D).
        simulator_fn: Function theta -> 2D image.
        nplot: Number of samples to simulate.
        ncols: Number of grid columns.
        random_seed: Seed for choosing the samples.

    Returns:
        (fig, flattened axes).
    """
    rng = np.random.default_rng(random_seed)
    indices = rng.choice(len(posterior_samples), size=min(nplot, len(posterior_samples)), replace=False)
    images = [np.asarray(simulator_fn(theta)) for theta in posterior_samples[indices]]
    return image_grid(images, ncols=ncols)
