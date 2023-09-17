"""
Functions to generate states in coupling phase oscillators
"""

import numpy as np

__all__ = [
    "generate_q_twisted_state",
    "generate_k_clusters",
    "generate_state",
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
    psi_init += rand_phase
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
        Kind of state to generate, by default "random".
        * "sync": full sync, all identical phases,
        * "random": uniform random on [0, 2pi[,
        * "splay": splay state, evenly spaced on [0, 2pi[,
        * "k-cluster": random k-cluster state,
        * "q-twisted": q-twisted state
    noise : float, optional
        Level of noise to add to the initial conditions, by default 1e-2.
    seed : int or None (default)
        Seed for the random number generator.
    **kwargs
        Keyword arguments to be passed to `generate_k_clusters()` or
        `generate_q_twisted_state()`.


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
    else:
        raise ValueError("Unknown kind.")

    if kind in ["sync", "splay"]:
        psi_init += perturbation

    return psi_init
