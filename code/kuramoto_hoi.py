"""
Functions to simulate the synchronisation 
of oscillators with group interactions
"""

from math import cos, exp, sin

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sb
import xgi
from numpy.linalg import norm

__all__ = [
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
]


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


