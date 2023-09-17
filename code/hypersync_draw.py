"""
Functions to visualise sync in coupled oscillators
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

__all__ = [
    "plot_series",
    "plot_order_param",
    "plot_phases",
    "plot_sync",
]


def plot_series(thetas, times, ax=None, n=None):
    """
    Plot time series of sin(theta) for the given phases thetas.

    Parameters
    ----------
    thetas : ndarray
        The values of the phases.
    times : ndarray
        The corresponding times.
    ax : Matplotlib axis, optional
        The Matplotlib axis to plot on, by default None (creates a new axis).
    n : int, optional
        The number of thetas to plot, by default None (plots all thetas).

    Returns
    -------
    ax : Matplotlib axis
        The Matplotlib axis the plot was drawn on.
    """

    if ax is None:
        ax = plt.gca()

    # plot time series
    for theta in thetas[:n]:
        ax.plot(times, np.sin(theta), c="grey", alpha=0.1)

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\sin(\theta)$")

    return ax


def plot_order_param(thetas, times, ax=None, order=1, color="r", ls="-"):
    """
    Plot the order parameter for the given phases thetas.

    Parameters
    ----------
    thetas : ndarray
        The values of the phases over time.
    times : ndarray
        The corresponding times.
    ax : Matplotlib axis, optional
        The Matplotlib axis to plot on, by default None (creates a new axis).
    order : int, optional
        The order of the order parameter, by default 1.
    color : str, optional
        The color of the plot, by default "r".
    ls : str, optional
        The line style of the plot, by default "-".

    Returns
    -------
    ax : Matplotlib axis
        The Matplotlib axis the plot was drawn on.
    """

    if ax is None:
        ax = plt.gca()

    N = len(thetas)
    R = np.sum(np.exp(1j * order * thetas), axis=0) / N
    ax.plot(times, np.abs(R), c=color, ls=ls, label=f"$R_{order}$")

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$R$")
    ax.set_ylim([-0.01, 1.01])

    return ax


def plot_phases(thetas, it, ax=None, color="b", ms=2):
    """
    Plot the phase plot of oscillators at time `it` on a circle.

    Parameters
    ----------
    thetas : np.ndarray
        The phase of each oscillator over time. Shape is (N, T).
    it : int
        The time index to plot the phase plot for.
    ax : plt.Axes, optional
        The axes to plot on, by default None.
    color : str, optional
        The color of the phase plot, by default "b".

    Returns
    -------
    plt.Axes
        The plot's axes.
    """
    if ax is None:
        ax = plt.gca()

    ax.plot(
        np.sin(thetas[:, it]), np.cos(thetas[:, it]), "o", c=color, ms=ms, alpha=0.3
    )

    circle = np.linspace(0, 2 * np.pi, num=100, endpoint=False)
    ax.plot(np.sin(circle), np.cos(circle), "-", c="lightgrey", zorder=-2)
    sb.despine(ax=ax, left=True, bottom=True)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_aspect("equal")

    return ax


def plot_sync(thetas, times, n=None):
    """
    Plot the time series of oscillators, their phase plots, and the order parameter.

    Parameters
    ----------
    thetas : np.ndarray
        The phase of each oscillator over time. Shape is (N, T).
    times : np.ndarray
        The time stamps for the `thetas` data.
    n : int, optional
        Number of time series to plot, by default None.

    Returns
    -------
    tuple
        (`fig`, `axs`) where `fig` is a `plt.Figure` and `axs` is a numpy ndarray of `plt.Axes`.
    """

    fig, axs = plt.subplots(2, 2, figsize=(4, 2), width_ratios=[3, 1], sharex="col")

    plot_series(thetas, times, ax=axs[0, 0], n=n)

    plot_order_param(thetas, times, ax=axs[1, 0], order=1)
    plot_order_param(thetas, times, ax=axs[1, 0], order=2, ls="--")

    plot_phases(thetas, 0, ax=axs[0, 1], color="b")

    plot_phases(thetas, -1, ax=axs[1, 1], color="b")

    return fig, axs
