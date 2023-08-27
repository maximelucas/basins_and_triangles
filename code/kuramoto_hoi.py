"""
Functions to simulate and visualise the synchronisation 
of oscillators with group interactions
"""

from math import sin, exp, cos

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import networkx as nx
from numpy.linalg import norm

import xgi

__all__ = [
    "generate_q_twisted_state",
    "generate_k_clusters",
    "generate_state",
    "identify_state",
    "identify_winding_number",
    "order_parameter",
    "rhs_pairwise",
    "rhs_pairwise_micro",
    "rhs_triplet_sym_micro",
    "rhs_pairwise_all",
    "rhs_pairwise_all_harmonic",
    "rhs_triplet_all_sym",
    "rhs_triplet_all_asym",
    "rhs_all_noloop",
    "rhs_adhikari",
    "rhs_lucas",
    "simulate_kuramoto",
    "plot_series",
    "plot_order_param",
    "plot_phases",
    "plot_sync",
]


def generate_q_twisted_state(N, q, noise=1e-2, seed=None):
    """
    Generate a q-twisted state for N phase oscillators.

    Parameters:
    -----------
    N : int
        The number of oscillators.
    q : int
        The number of twists.
    noise : float, optional
        The magnitude of gaussian noise added to the phases. Default is 1e-2.
    seed : int or None, optional
        Seed for the random number generator. Default is None.

    Returns:
    --------
    psi_init : array-like, shape (N,)
        An array of generated phase angles representing the initial state.
    """
    if seed is not None:
        np.random.seed(seed)

    perturbation = noise * np.random.normal(size=N)
    rand_phase = np.random.random() * 2 * np.pi

    psi_init = 2 * np.pi * q * np.arange(1, N + 1) / N
    psi_init += perturbation

    return psi_init


def generate_k_clusters(N, k, ps, noise=1e-2, seed=None):
    """
    Generate a k-cluster state for N phase oscillators.

    Parameters:
    -----------
    N : int
        The number of oscillators.
    k : int
        The number of clusters to generate.
    ps : array-like
        The probabilities of each cluster. The number of elements in `ps` must be equal to `k`,
        and the sum of probabilities must be approximately 1.
    noise : float, optional
        The magnitude of gaussian noise added to the phases. Default is 1e-2.
    seed : int or None, optional
        Seed for the random number generator. Default is None.

    Returns:
    --------
    psi_init : array-like, shape (N,)
        An array of generated phase angles representing the initial state.
    """

    if seed is not None:
        np.random.seed(seed)

    if len(ps) != k:
        raise ValueError(
            "The number of elements in ps must be equal to the number of clusters k."
        )

    if not np.isclose(sum(ps), 1):
        raise ValueError("The ps must sum to one.")

    perturbation = noise * np.random.normal(size=N)
    rand_phase = np.random.random() * 2 * np.pi

    choices = rand_phase + np.linspace(0, 2 * np.pi, num=k, endpoint=False)
    psi_init = np.random.choice(choices, size=N, p=ps)

    psi_init += perturbation

    return psi_init


def generate_state(N, kind="random", noise=1e-2, seed=None, **kwargs):
    """
    Generate initial conditions for a system of N oscillators.

    Parameters
    ----------
    N : int
        Number of oscillators in the system.
    kind : str, optional
        Kind of initial conditions to generate, by default "random".
        "sync": synchronized state, "randsync": homogeneous random,
        "random": completely random, "splay": splay state,
        "rand2clust": random two-cluster state.
    noise : float, optional
        Level of noise to add to the initial conditions, by default 1e-2.
    p2 : float, optional
        Probability of choosing the second cluster in "rand2clust" option, by default None.

    Returns
    -------
    numpy.ndarray
        Array of shape (N,) with the initial conditions for each oscillator.
    """

    if seed is not None:
        np.random.seed(seed)

    perturbation = noise * np.random.normal(size=N)
    rand_phase = np.random.random() * 2 * np.pi

    if kind == "sync":
        psi_init = rand_phase * np.ones(N)
    elif kind == "random":  # random
        psi_init = np.random.random(N) * 2 * np.pi
    elif kind == "splay":  # splay state
        psi_init = np.linspace(0, 2 * np.pi, num=N, endpoint=False)
    elif kind == "k-cluster":
        psi_init = generate_k_clusters(N, **kwargs, noise=noise, seed=seed)
    elif kind == "q-twisted":
        psi_init = generate_q_twisted_state(N, **kwargs, noise=noise, seed=seed)

    if kind in ["sync", "splay"]:
        psi_init += perturbation

    return psi_init


def identify_state(thetas, t=-1, atol=1e-3):
    """
    Identify the synchronization state.

    Parameters:
    -----------
    thetas : array-like, shape (n_phases, n_time_steps)
        Phases of the oscillators over time
    t : int, optional
        The index of the time step to analyze. Default is -1, indicating the last time step.
    atol : float, optional
        The absolute tolerance used for comparison with synchronization parameters. Default is 1e-3.

    Returns:
    --------
    state : str
        A string representing the identified synchronization state:
        - "sync" for complete synchronization.
        - "2-cluster" for 2-cluster synchronization.
        - "3-cluster" for 3-cluster synchronization.
        - "q-twisted" for q-twisted synchronization, where q is the winding number.
        - "splay" for splay synchronization.
        - "other" for other or unsynchronized states.
    """
    N = thetas.shape[0]

    R1 = order_parameter(thetas, order=1)
    R2 = order_parameter(thetas, order=2)
    R3 = order_parameter(thetas, order=3)
    diff = np.diff(thetas[:,t], append=thetas[0, t]) % (2*np.pi)
    is_diff_zero = np.isclose(diff, 0, atol=atol) + np.isclose(diff, 2*np.pi, atol=atol)
    
    q, is_twisted = identify_winding_number(thetas, t=-1)
    sorted_thetas = np.sort(thetas, axis=0)  # sort along node axis
    q_sorted, is_splay = identify_winding_number(sorted_thetas, t=-1)

    if np.isclose(R1[t], 1, atol=atol) and np.all(is_diff_zero):
        return "sync"
    elif np.isclose(R2[t], 1, atol=atol):
        return "2-cluster"
    elif np.isclose(R3[t], 1, atol=atol):
        return "3-cluster"
    elif is_twisted:
        return f"{q}-twisted"
    elif is_splay and q_sorted == 1:
        return "splay"
    else:
        return "other"


def identify_winding_number(thetas, t):
    """
    Check if twisted state and identify its winding number.

    The winding number indicates how many times the phase angles wind around the unit circle.

    Parameters:
    -----------
    thetas : array-like, shape (n_phases, n_time_steps)
        Phases of the oscillators over time
    t : int
        The index of the time step for which the winding number and twisted state are calculated.

    Returns:
    --------
    w_no : int
        The winding number, indicating how many times the phase angles wind around the unit circle.
    is_twisted_state : bool
        True if the phase differences are close to their mean, indicating a twisted state;
        False otherwise.
    """
    thetas = thetas % (2 * np.pi)  # ensure it's mod 2 pi
    N = len(thetas)

    diff = np.diff(thetas[:, t], prepend=thetas[-1, t])

    # ensure phase diffs are in [-pi, pi]
    diff = np.where(diff > np.pi, diff - 2 * np.pi, diff)
    diff = np.where(diff < -np.pi, diff + 2 * np.pi, diff)

    q = np.sum(diff)
    w_no = round(q / (2 * np.pi))
    is_twisted_state = norm(diff - np.mean(diff)) < 1e-1

    return w_no, is_twisted_state


def order_parameter(thetas, order=1, complex=False, axis=0):
    """
    Calculate the generalised Daido order parameter of order `order`.

    The order parameter is a measure of how the phase angles are aligned.
    It can be used to quantify the level of synchronization or coherence in a system.

    Parameters:
    -----------
    thetas : array-like, shape (n_oscillators, n_times)
        Phases of the oscillators over time
    order : int, optional
        The order of the order parameter. Default is 1.
    complex : bool, optional
        If True, return the complex order parameter. If False (default), return its magnitude.
    axis : int, optional
        The axis of length n_oscillators, along which the sum of the phases is taken. Default is 0.

    Returns:
    --------
    result : array-like
        If `complex` is True, the complex order parameter.
        If `complex` is False, the magnitude of the order parameter.
    """

    N = len(thetas)
    Z = np.sum(np.exp(1j * order * thetas), axis=axis) / N

    if complex:
        return Z
    else:
        return np.abs(Z)


def rhs_triplet_sym_micro(t, psi, omega, k1, k2, triangles):
    """Right-hand side of the ODE.
    Only triplets.
    Coupling function: sin(oj + ok - 2oi)

    Parameters
    ----------
    t: float
        Time
    psi: array of float
        Phases to integrate
    """
    N = len(psi)
    triplet = np.zeros(N)

    for i, j, k in triangles:
        # sin(oj - ok - 2 oi)
        oi = psi[i]
        oj = psi[j]
        ok = psi[k]
        triplet[i] += 2 * sin(oj + ok - 2 * oi)
        triplet[j] += 2 * sin(oi + ok - 2 * oj)
        triplet[k] += 2 * sin(oj + oi - 2 * ok)

    return omega + (k2 / N**2) * triplet


def rhs_pairwise_micro(t, psi, omega, k1, k2, links):
    """Right-hand side of the ODE.
    Only pairwise.
    Coupling function: sin(oj - oi)

    Parameters
    ----------
    t: float
        Time
    psi: array of float
        Phases to integrate
    """

    N = len(psi)
    pairwise = np.zeros(N)

    for i, j in links:
        # sin(oj - oi)
        oi = psi[i]
        oj = psi[j]
        pairwise[i] += sin(oj - oi)
        pairwise[j] += sin(oi - oj)

    return omega + (k1 / N) * pairwise


def rhs_pairwise(t, psi, omega, k1, k2, adj1):
    """Right-hand side of the ODE.
    All-to-all, only pairwise.
    Coupling function: sin(oj - oi)

    Parameters
    ----------
    t: float
        Time
    psi: array of float
        Phases to integrate
    """

    N = len(psi)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    pairwise = adj1.dot(sin_psi) * cos_psi - adj1.dot(cos_psi) * sin_psi

    return omega + (k1 / N) * pairwise


def rhs_pairwise_all(t, psi, omega, k1, k2):
    """Right-hand side of the ODE.
    All-to-all, only pairwise.
    Coupling function: sin(oj - oi)

    Parameters
    ----------
    t: float
        Time
    psi: array of float
        Phases to integrate
    """

    N = len(psi)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    sum_cos_psi = np.sum(cos_psi)
    sum_sin_psi = np.sum(sin_psi)

    # oj - oi
    pairwise = -sum_cos_psi * sin_psi + sum_sin_psi * cos_psi

    return omega + (k1 / N) * pairwise


def rhs_pairwise_all_harmonic(t, psi, omega, k1, k2):
    """Right-hand side of the ODE.
    All-to-all, only pairwise.
    Coupling function: sin(2oj - 2oi)

    Parameters
    ----------
    t: float
        Time
    psi: array of float
        Phases to integrate
    """

    N = len(psi)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    sum_cos_psi_sq = np.sum(cos_psi**2)
    sum_sin_psi_sq = np.sum(sin_psi**2)

    # 2oj - 2oi
    pairwise = 2 * (
        -cos_psi * sin_psi * sum_cos_psi_sq
        + cos_psi**2 * np.sum(cos_psi * sin_psi)
        - sin_psi**2 * np.sum(cos_psi * sin_psi)
        + cos_psi * sin_psi * sum_sin_psi_sq
    )

    return omega + (k1 / N) * pairwise


def rhs_triplet_all_sym(t, psi, omega, k1, k2):
    """Right-hand side of the ODE.
    All-to-all, only triplets.
    Coupling function: sin(oj + ok - 2oi)

    Parameters
    ----------
    t: float
        Time
    psi: array of float
        Phases to integrate
    """

    N = len(psi)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    sum_cos_psi = np.sum(np.cos(psi))
    sum_sin_psi = np.sum(np.sin(psi))

    # oj + ok - 2oi
    triplet = (
        -2 * sum_cos_psi**2 * sin_psi * cos_psi
        + cos_psi**2 * sum_cos_psi * sum_sin_psi
        - sum_cos_psi * sin_psi**2 * sum_sin_psi
        + sum_cos_psi * cos_psi**2 * sum_sin_psi
        - sum_cos_psi * sin_psi**2 * sum_sin_psi
        + 2 * cos_psi * sin_psi * sum_sin_psi**2
    )

    return omega + (k2 / N**2) * triplet


def rhs_triplet_all_asym(t, psi, omega, k1, k2):
    """Right-hand side of the ODE.
    All-to-all, only triplets.
    Coupling function: sin(2 oj - ok - oi)

    Parameters
    ----------
    t: float
        Time
    psi: array of float
        Phases to integrate
    """

    N = len(psi)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    sum_cos_psi = np.sum(np.cos(psi))
    sum_sin_psi = np.sum(np.sin(psi))

    sum_cos_psi_sq = np.sum(np.cos(psi) ** 2)
    sum_sin_psi_sq = np.sum(np.sin(psi) ** 2)

    # 2 oj - ok - oi
    triplet = (
        -sum_cos_psi_sq * sum_cos_psi * sin_psi
        + 2 * cos_psi * np.sum(cos_psi * sin_psi) * sum_cos_psi
        + sum_cos_psi * sin_psi * sum_sin_psi_sq
        - cos_psi * sum_cos_psi_sq * sum_sin_psi
        - 2 * np.sum(cos_psi * sin_psi) * sin_psi * sum_sin_psi
        + cos_psi * sum_sin_psi_sq * sum_sin_psi
    )

    return omega + (k2 / N**2) * triplet


def rhs_all_noloop(t, psi):
    """Right-hand side of the differential equation"""

    links = H.edges.filterby("size", 2).members()
    triangles = H.edges.filterby("size", 3).members()

    pairwise = np.zeros(N)
    triplet = np.zeros(N)

    sum_cos_psi = np.sum(np.cos(psi))
    sum_sin_psi = np.sum(np.sin(psi))

    pairwise = sum_sin_psi * np.cos(psi) - sum_cos_psi * np.sin(psi)

    # oj + ok - 2oi
    triplet = (
        sum_cos_psi * np.cos(psi) ** 2 * sum_sin_psi
        + 2 * np.cos(psi) * sum_sin_psi * np.sin(psi) * sum_sin_psi
        - sum_cos_psi * np.sin(psi) ** 2 * sum_sin_psi
        + sum_cos_psi * np.cos(psi) ** 2 * sum_sin_psi
        + sum_cos_psi * (-2) * sum_cos_psi * np.cos(psi) * np.sin(psi)
        + sum_cos_psi * (-1) * sum_sin_psi * np.sin(psi) ** 2
    )

    return omega + k1 * pairwise + k2 * triplet


def rhs_lucas(t, psi, omega, k1, k2, k1_avg, k2_avg, links, triangles):
    """Right-hand side of the differential equation

    Pairwise coupling function: sin(oj - oi)
    Triplet coupling function: sin(oj + ok - 2oi)

    ref: Eq 1 of https://www.nature.com/articles/s41467-023-37190-9

    Parameters
    ----------
    t : float
        Time
    psi : array of float
        State of the N phases at time t
    k1 : float
        Pairwise coupling strength
    k2 : float
        Triplet coupling strength
    **args
        Other arguments to feed this function
    """

    N = len(psi)
    pairwise = np.zeros(N)
    triplet = np.zeros(N)

    for i, j in links:
        # sin(oj - oi)
        oi = psi[i]
        oj = psi[j]
        pairwise[i] += sin(oj - oi)
        pairwise[j] += sin(oi - oj)

    for i, j, k in triangles:
        # sin(2 oj - ok - oi)
        oi = psi[i]
        oj = psi[j]
        ok = psi[k]
        triplet[i] += 2 * sin(oj + ok - 2 * oi)
        triplet[j] += 2 * sin(oi + ok - 2 * oj)
        triplet[k] += 2 * sin(oj + oi - 2 * ok)

    g1 = k1 / k1_avg if k1_avg != 0 else 0
    g2 = (k2 / (2 * k2_avg)) if k2_avg != 0 else 0

    return omega + g1 * pairwise + g2 * (triplet / 2)


def rhs_adhikari(t, psi):
        """Right-hand side of the differential equation"""

        pairwise = np.zeros(N)
        triplet = np.zeros(N)

        for i, j in links:
            # sin(oj - oi)
            oi = psi[i]
            oj = psi[j]
            pairwise[i] += sin(oj - oi)
            pairwise[j] += sin(oi - oj)

        for i, j, k in triangles:
            # sin(2 oj - ok - oi)
            oi = psi[i]
            oj = psi[j]
            ok = psi[k]
            triplet[i] += sin(2 * oj - ok - oi) + sin(2 * ok - oj - oi)
            triplet[j] += sin(2 * oi - ok - oj) + sin(2 * ok - oi - oj)
            triplet[k] += sin(2 * oj - oi - ok) + sin(2 * oi - oj - ok)

        return omega + k1 * pairwise + k2 * triplet


def simulate_kuramoto(
    S,
    k1,
    k2,
    omega=None,
    theta_0=None,
    t_end=100,
    dt=0.01,
    rhs=None,
    integrator="explicit_euler",
    **kwargs,
):
    """
    Simulate the Kuramoto model on a hypergraph with links and triangles.

    Parameters
    ----------
    S : Network object
        Network object representing the graph.
    omega : array-like, optional
        Natural frequencies of each node. If not given, a random normal distribution with mean 0 and
        standard deviation 1 is used.
    theta_0 : array-like, optional
        Initial phases of each node. If not given, a random phase is used.
    k1 : float, optional
        Coupling strength for pairwise interactions.
    k2 : float, optional
        Coupling strength for triplet interactions.
    t_end : float, optional
        End time of the integration
    dt : float, optional
        Time step for the simulation.
    integrator : str, optional
        Integration method to use. Either "explicit_euler" or any method supported by
        scipy.integrate.solve_ivp.

    Returns
    -------
    thetas : array-like
        Phases of each node at each time step.
    times : array-like
        Time points for each time step.
    """

    H = xgi.convert_labels_to_integers(S, "label")
    N = H.num_nodes

    if omega is None:
        omega = np.random.normal(0, 1, N)

    if theta_0 is None:
        theta_0 = np.random.random(N) * 2 * np.pi

    times = np.arange(0, t_end + dt / 2, dt)
    n_t = len(times)

    links = H.edges.filterby("size", 2).members()
    triangles = H.edges.filterby("size", 3).members()

    k1_avg = H.nodes.degree(order=1).mean()
    k2_avg = H.nodes.degree(order=2).mean()

    if rhs is None:

        def rhs(t, psi, k1, k2, **kwargs):
            """Right-hand side of the differential equation"""

            pairwise = np.zeros(N)
            triplet = np.zeros(N)

            for i, j in links:
                # sin(oj - oi)
                oi = psi[i]
                oj = psi[j]
                pairwise[i] += sin(oj - oi)
                pairwise[j] += sin(oi - oj)

            for i, j, k in triangles:
                # sin(2 oj - ok - oi)
                oi = psi[i]
                oj = psi[j]
                ok = psi[k]
                triplet[i] += 2 * sin(oj + ok - 2 * oi)
                triplet[j] += 2 * sin(oi + ok - 2 * oj)
                triplet[k] += 2 * sin(oj + oi - 2 * ok)

            g1 = k1 / k1_avg if k1_avg != 0 else 0
            g2 = (k2 / (2 * k2_avg)) if k2_avg != 0 else 0

            return omega + g1 * pairwise + g2 * (triplet / 2)

    thetas = np.zeros((N, n_t))
    thetas[:, 0] = theta_0

    if integrator == "explicit_euler":
        for it in range(1, n_t):
            thetas[:, it] = thetas[:, it - 1] + dt * rhs(
                0, thetas[:, it - 1], omega, k1, k2, **kwargs
            )
    else:
        thetas = solve_ivp(
            rhs,
            [times[0], times[-1]],
            theta_0,
            t_eval=times,
            method=integrator,
            rtol=1.0e-8,
            atol=1.0e-8,
        ).y

    return thetas, times


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
