"""
Functions to identify states in coupled oscillators
"""

import numpy as np
from numpy.linalg import norm

__all__ = [
    "identify_state",
    "identify_k_clusters",
    "identify_winding_number",
    "order_parameter",
]


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

    R1 = order_parameter(thetas, order=1)
    # R2 = order_parameter(thetas, order=2)
    # R3 = order_parameter(thetas, order=3)
    diff = np.diff(thetas[:, t], append=thetas[0, t]) % (2 * np.pi)
    is_diff_zero = np.isclose(diff, 0, atol=atol) + np.isclose(
        diff, 2 * np.pi, atol=atol
    )

    q, is_twisted = identify_winding_number(thetas, t=-1)
    sorted_thetas = np.sort(thetas, axis=0)  # sort along node axis
    q_sorted, is_splay = identify_winding_number(sorted_thetas, t=-1)

    try:
        is_2clust, sizes2 = identify_k_clusters(thetas, k=2, t=-1, atol=1e-2)
    except Exception as err:
        is_2clust = False
        sizes2 = []
        print(err)

    try:
        is_3clust, sizes3 = identify_k_clusters(thetas, k=3, t=-1, atol=1e-2)
    except Exception as err:
        is_3clust = False
        sizes3 = []
        print(err)

    if is_twisted:
        return f"{q}-twisted"
    elif is_splay and q_sorted == 1:
        return "splay"
    elif np.isclose(R1[t], 1, atol=atol) and np.all(is_diff_zero):
        return "sync"
    elif is_2clust:
        return "2-cluster"
    elif is_3clust:
        return "3-cluster"
    else:
        return "other"


def identify_k_clusters(thetas, k, t, atol=1e-2):
    """
    Check if k-cluster state.

    A k-cluster state has k evenly spaced clusters on the circle.

    Parameters:
    -----------
    thetas : array-like, shape (n_phases, n_time_steps)
        Phases of the oscillators over time
    k : int
        Number of clusters
    t : int
        The index of the time step for which the winding number and twisted state are calculated.

    Returns:
    --------
    is_k_clusters : bool
        True if the state is a k-cluster. False otherwise.
    sizes : tuple of float
        The relative sizes of each cluster.
    """

    n_clust = k
    dist = 2 * np.pi / n_clust
    N = len(thetas)

    psi = thetas[:, t] % (2 * np.pi)
    psi = np.sort(psi)

    diff = np.diff(psi)
    idcs = np.where(diff > dist / 2)[0]

    clusters = []
    n_changes = len(idcs)
    for i in range(n_changes + 1):
        start = idcs[i - 1] + 1 if i > 0 else None
        end = idcs[i] + 1 if i < n_changes else None
        clusters.append(psi[start:end])

    if len(clusters) < k:
        return False, []

    is_k_clusters = True  # changed below if False
    sizes = [0] * k

    for i in range(n_changes + 1):
        if np.mean(np.diff(clusters[i])) > atol:  # cluster is not compact
            is_k_clusters = False

    for i in range(n_changes):
        dist_ij = abs(np.mean(clusters[i]) - np.mean(clusters[i + 1]))
        if abs(dist_ij - dist) > atol:
            is_k_clusters = False  # clusters have wrong distance between them

    if n_clust == len(clusters):
        sizes = [len(cluster) / N for cluster in clusters]
    elif n_clust == len(clusters) - 1:
        sizes = [len(cluster) / N for cluster in clusters[:-1]]
        sizes[0] += len(clusters[-1])  # 0th and last clusters are the same
    else:
        raise ValueError("k must be equal to or one unit below len(cluster)")

    return is_k_clusters, sizes


def identify_winding_number(thetas, t, atol=1e-1):
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

    diff = np.diff(thetas[:, t], prepend=thetas[-1, t])

    # ensure phase diffs are in [-pi, pi]
    diff = np.where(diff > np.pi, diff - 2 * np.pi, diff)
    diff = np.where(diff < -np.pi, diff + 2 * np.pi, diff)

    q = np.sum(diff)
    w_no = round(q / (2 * np.pi))
    is_twisted_state = norm(diff - np.mean(diff)) < atol

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
