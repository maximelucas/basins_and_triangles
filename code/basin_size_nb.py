#!/usr/bin/env python

"""
Call with 
./basins_max.py --num_threads 8 -n 1000 -t 300
"""

# imports
import argparse
import multiprocessing
import shutil
from datetime import datetime
from itertools import combinations
from math import sin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import xgi
from hypersync_draw import *
from hypersync_generate import *
from hypersync_identify import *
from hypersync_integrate import *
from numba import jit

sb.set_theme(style="ticks", context="notebook")

results_dir = "../results/"
data_dir = "../data/"

Path(results_dir).mkdir(parents=True, exist_ok=True)
Path(data_dir).mkdir(parents=True, exist_ok=True)


@jit(nopython=True)
def rhs_oneloop_nb(t, theta, omega, k1, k2, r1, r2):
    """
    RHS

    Parameters
    ----------
    sigma : float
        Triplet coupling strength
    K1, K2 : int
        Pairwise and triplet nearest neighbour ranges
    """

    N = len(theta)

    pairwise = np.zeros(N)
    triplets = np.zeros(N)

    # triadic coupling
    idx_2 = list(range(-r2, 0)) + list(range(1, r2 + 1))
    idx_1 = range(-r1, r1 + 1)

    for ii in range(N):
        for jj in idx_1:  # pairwise
            jjj = (ii + jj) % N
            pairwise[ii] += sin(theta[jjj] - theta[ii])

        for jj in idx_2:  # triplet
            for kk in idx_2:
                if jj < kk:  # because coupling function is symmetric in j and k
                    jjj = (ii + jj) % N
                    kkk = (ii + kk) % N
                    # x2 to count triangles in both directions
                    triplets[ii] += 2 * sin(theta[kkk] + theta[jjj] - 2 * theta[ii])

    return (k1 / r1) * pairwise + k2 / (r2 * (2 * r2 - 1)) * triplets


@jit(nopython=True)
def rhs_oneloop_nb_quadruplet(t, theta, omega, k1, k2, r1, r2):
    """
    RHS

    Parameters
    ----------
    sigma : float
        Triplet coupling strength
    K1, K2 : int
        Pairwise and triplet nearest neighbour ranges
    """

    N = len(theta)

    pairwise = np.zeros(N)
    triplets = np.zeros(N)

    # triadic coupling
    idx_2 = list(range(-r2, 0)) + list(range(1, r2 + 1))
    idx_1 = range(-r1, r1 + 1)

    for ii in range(N):
        for jj in idx_1:  # pairwise
            jjj = (ii + jj) % N
            pairwise[ii] += sin(theta[jjj] - theta[ii])

        for jj in idx_2:  # triplet
            for kk in idx_2:
                for ll in idx_2:
                    if jj != kk and jj != ll and kk != ll:
                        jjj = (ii + jj) % N
                        kkk = (ii + kk) % N
                        lll = (ll + jj) % N
                        # x2 to count triangles in both directions
                        triplets[ii] += sin(
                            theta[lll] + theta[kkk] + theta[jjj] - 3 * theta[ii]
                        )

    g2 = (1 / 3) * r2 * (2 * r2 - 2) * (2 * r2 - 1)  # (2 * r2) choose 3
    return (k1 / r1) * pairwise + k2 / g2 * triplets


@jit(nopython=True)
def rhs_oneloop_nb_asym(t, theta, omega, k1, k2, r1, r2):
    """
    RHS

    Parameters
    ----------
    sigma : float
        Triplet coupling strength
    K1, K2 : int
        Pairwise and triplet nearest neighbour ranges
    """

    N = len(theta)

    pairwise = np.zeros(N)
    triplets = np.zeros(N)

    # triadic coupling
    idx_2 = list(range(-r2, 0)) + list(range(1, r2 + 1))
    idx_1 = range(-r1, r1 + 1)

    for ii in range(N):
        for jj in idx_1:  # pairwise
            jjj = (ii + jj) % N
            pairwise[ii] += sin(theta[jjj] - theta[ii])

        for jj in idx_2:  # triplet
            for kk in idx_2:
                if jj != kk:
                    jjj = (ii + jj) % N
                    kkk = (ii + kk) % N
                    # x2 to count triangles in both directions
                    triplets[ii] += sin(2 * theta[kkk] - theta[jjj] - theta[ii])

    return (k1 / r1) * pairwise + k2 / (r2 * (2 * r2 - 1)) * triplets


@jit(nopython=True)
def rhs_oneloop_SC_nb(t, theta, omega, k1, k2, r1, r2):
    """
    RHS

    Parameters
    ----------
    sigma : float
        Triplet coupling strength
    K1, K2 : int
        Pairwise and triplet nearest neighbour ranges
    """

    N = len(theta)

    pairwise = np.zeros(N)
    triplets = np.zeros(N)

    # triadic coupling
    idx_2 = list(range(-r2, 0)) + list(range(1, r2 + 1))
    idx_1 = range(-r1, r1 + 1)

    for ii in range(N):
        for jj in idx_1:  # pairwise
            jjj = (ii + jj) % N
            pairwise[ii] += sin(theta[jjj] - theta[ii])

        for jj in idx_2:  # triplet
            for kk in idx_2:
                if (jj < kk) and (
                    kk - jj
                ) <= r2:  # because coupling function is symmetric in j and k
                    jjj = (ii + jj) % N
                    kkk = (ii + kk) % N
                    # x2 to count triangles in both directions
                    triplets[ii] += 2 * sin(theta[kkk] + theta[jjj] - 2 * theta[ii])

    g2 = (r2 * (2 * r2 - 1)) / 2  # divide by to remove undirected
    return (k1 / r1) * pairwise + k2 / g2 * triplets


def di_nearest_neigbors(N, d, r):
    """
    Create a d-uniform hypergraph representing nearest neighbor relationships.

    Parameters
    ----------
    N : int
        The total number of nodes.

    d : int
        Size of hyperedges

    r : int
        The range of neighbors to consider. Neighbors within the range [-r, r]
        (excluding the node itself) will be connected.

    Returns
    -------
    xgi.Hypergraph
        A hypergraph object representing the nearest neighbor relationships.
    """

    DH = xgi.DiHypergraph()
    nodes = np.arange(N)

    edges = []
    neighbor_rel_ids = np.concatenate((np.arange(-r, 0), np.arange(1, r + 1)))

    for i in nodes:
        neighbor_ids = i + neighbor_rel_ids
        edge_neighbors_i = combinations(neighbor_ids, d - 1)
        edges_i = [[list(np.mod(comb, N)), [i]] for comb in edge_neighbors_i]
        edges = edges + edges_i

    # edges = np.mod(edges, N)

    DH.add_nodes_from(nodes)
    DH.add_edges_from(edges)
    DH.cleanup()  # remove duplicate

    return DH


def ring_dihypergraph(N, r1, r2):
    H2 = di_nearest_neigbors(N, d=3, r=r2)
    H1 = di_nearest_neigbors(N, d=2, r=r1)

    DH = xgi.DiHypergraph()
    DH.add_nodes_from(H1.nodes)
    DH.add_edges_from(H1.edges.dimembers())
    DH.add_edges_from(H2.edges.dimembers())

    return DH


def rhs_diHG(t, psi, omega, k1, k2, r1, r2, dilinks, ditriangles):
    """
    RHS

    Parameters
    ----------
    k1, k2 : floats
        Pairwise and triplet coupling strengths
    r1, r2 : int
        Pairwise and triplet nearest neighbour ranges
    adj1 : ndarray, shape (N, N)
        Adjacency matrix of order 1
    triangles: list of sets
        List of unique triangles

    """

    N = len(psi)

    pairwise = np.zeros(N)

    for senders, receiver in dilinks:
        # sin(oj - oi)
        senders = list(senders)
        receiver = list(receiver)
        i = receiver
        j = senders
        oi = psi[i]
        oj = psi[j]
        pairwise[i] += sin(oj - oi)

    triplet = np.zeros(N)

    # print(len(triangles))
    for senders, receiver in ditriangles:
        # sin(oj + ok - 2 oi)
        senders = list(senders)
        receiver = list(receiver)
        i = receiver
        j = senders[0]
        k = senders[1]
        oi = psi[i]
        oj = psi[j]
        ok = psi[k]
        triplet[i] += 2 * sin(oj + ok - 2 * oi)

    g1 = r1
    g2 = r2 * (2 * r2 - 1)

    return omega + (k1 / g1) * pairwise + (k2 / g2) * triplet


def simulate_iteration(
    i,
    H,
    k1,
    k2,
    omega,
    t_end,
    dt,
    ic,
    noise,
    rhs,
    integrator,
    args,
    t_eval,
    n_reps,
    run_dir="",
    **options,
):
    N = len(H)
    times = np.arange(0, t_end + dt / 2, dt)

    nrep_thetas = np.zeros((n_reps, N))  # , len(times)))

    for j in range(n_reps):
        psi_init = generate_state(N, kind=ic, noise=noise)

        thetas, times = simulate_kuramoto(
            H,
            k1=k1,
            k2=k2,
            omega=omega,
            theta_0=psi_init,
            t_end=t_end,
            dt=dt,
            rhs=rhs,
            integrator=integrator,
            args=args,
            t_eval=False,
        )

        nrep_thetas[j] = thetas[:, -1]

        if (j <= 5) or ("cluster" in identify_state(thetas)):
            fig, axs = plot_sync(thetas, times)

            axs[0, 1].set_title(f"$t={times[0]}$s", fontsize="x-small")
            axs[1, 1].set_title(f"$t={times[-1]}$s", fontsize="x-small")

            axs[0, 0].set_xlabel("")
            axs[1, 0].legend(loc="best", fontsize="x-small")

            plt.subplots_adjust(hspace=0.5, top=0.8)
            fig.suptitle(identify_state(thetas))
            tag_params = f"ring_k1_{k1}_k2_{k2}_{j}"
            fig_name = f"sync_{tag_params}"  # _{k2s[j]}_{i}"
            plt.savefig(f"{run_dir}{fig_name}.png", dpi=300, bbox_inches="tight")
            plt.close()

    return i, nrep_thetas


if __name__ == "__main__":
    # Record the start time
    start_time = datetime.now()

    parser = argparse.ArgumentParser(description="Run Kuramoto simulation")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads for parallelization",
    )
    parser.add_argument(
        "-n", "--n_reps", type=int, default=100, help="Number of repetitions"
    )
    parser.add_argument("-t", "--t_end", type=float, default=300, help="End time")

    parser.add_argument(
        "-i", "--integrator", type=str, default="RK45", help="ODE integrator"
    )

    args = parser.parse_args()

    n_reps = args.n_reps  # number of random realisations

    # generate structure
    N = 100
    # H = xgi.complete_hypergraph(N, max_order=2)

    r1 = 2
    r2 = 2

    suffix = "di_asym"  # "SC"

    # H = xgi.trivial_hypergraph(N)
    H = ring_dihypergraph(N, r1, r2)

    dilinks = H.edges.filterby("size", 2).dimembers()
    ditriangles = H.edges.filterby("size", 3).dimembers()

    # define parameters

    # dynamical
    k1 = 1  # pairwise coupling strength
    k2s = np.arange(0, 9.5, 0.5)  # triplet coupling strength
    omega = 0  # 1 * np.ones(N)  # np.random.normal(size=N) #1 * np.ones(N)

    ic = "random"  # initial condition type, see below
    noise = 1e-1  # noise strength

    # integration
    t_end = args.t_end  # default = 200
    dt = 0.01
    times = np.arange(0, t_end + dt / 2, dt)

    t_eval = False  # integrate at all above timepoints
    integrator = args.integrator
    options = {"atol": 1e-8, "rtol": 1e-8}

    tag_params = f"ring_k1_{k1}_k2s_ic_{ic}_tend_{t_end}_nreps_{n_reps}_{suffix}"

    # create directory for this run
    run_dir = f"{results_dir}run_{tag_params}/"
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    # copy current script
    current_script_path = Path(__file__).resolve()
    shutil.copy2(current_script_path, run_dir)

    print("============")
    print("simulating..")
    print("============")

    # Create a multiprocessing Pool
    with multiprocessing.Pool(processes=args.num_threads) as pool:
        results = []
        for i, k2 in enumerate(k2s):
            args = (r1, r2, dilinks, ditriangles)

            results.append(
                pool.apply_async(
                    simulate_iteration,
                    (
                        i,
                        H,
                        k1,
                        k2,
                        omega,
                        t_end,
                        dt,
                        ic,
                        noise,
                        rhs_diHG,  # change rhs here
                        integrator,
                        args,
                        t_eval,
                        n_reps,
                        run_dir,
                    ),
                    options,
                )
            )

        print("============")
        print("collecting parallel results..")
        print("============")

        # Collect the results
        thetas_arr = np.zeros((len(k2s), n_reps, N))  # , len(times)))
        for result in results:
            print(i)
            i, k2_thetas = result.get()
            thetas_arr[i] = k2_thetas
        print("============")
        print("done..")
        print("============")

    print("============")
    print("saving simulations..")
    print("============")

    np.save(
        f"{run_dir}thetas_arr_{tag_params}.npy", thetas_arr
    )  # Save thetas_arr to a NPY file

    print("============")
    print("identifying states..")
    print("============")

    # identify states and count their occurrences
    results = {}

    thetas_arr = thetas_arr.swapaxes(0, 1)  # get (n_rep, k2s)
    thetas_arr = thetas_arr[:, :, :, None]
    for j, k2 in enumerate(k2s):
        states = [identify_state(thetas, atol=0.05) for thetas in thetas_arr[:, j]]
        states_unique, counts = np.unique(states, return_counts=True)
        probs = counts / n_reps

        results[k2] = {}
        for state, prob in zip(states_unique, probs):
            results[k2][state] = prob

    print("============")
    print("plotting end states..")
    print("============")
    # plot end states
    wsize = 1
    fig, axs = plt.subplots(5, len(k2s), figsize=(len(k2s) * wsize, 5 * wsize))
    for i, thetas_rep in enumerate(thetas_arr[:5]):
        for j, thetas_k in enumerate(thetas_rep):
            plot_phases(thetas_k, -1, ax=axs[i, j], color="b")
            axs[i, j].text(
                0,
                0,
                f"{identify_state(thetas_k)}",
                fontsize="xx-small",
                va="center",
                ha="center",
            )

    for i, ax in enumerate(axs[0, :]):
        ax.set_title(f"k2={k2s[i]}")

    fig_name = f"end_state_{tag_params}"  # _{k2s[j]}_{i}"
    plt.savefig(f"{run_dir}{fig_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("============")
    print("plotting summary..")
    print("============")

    print(results)
    df = pd.DataFrame.from_dict(results, orient="index").reset_index(names="k2")
    df_long = df.melt(id_vars="k2", var_name="state", value_name="proba")

    df_long.to_csv(f"{run_dir}df_long_{tag_params}.csv")

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    g = sb.lineplot(
        data=df_long,
        x="k2",
        y="proba",
        hue="state",
        markers=True,
        ax=ax,
        alpha=0.7,
        style="state",
        # hue_order=["sync", "2-cluster", "other"]
    )

    g.set(yscale="log")
    sb.move_legend(g, loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_xlabel("k2, triplet coupling strength")

    title = f"ring {suffix}, {ic} ic, {n_reps} reps"
    ax.set_title(title)

    sb.despine()
    ax.set_ylim(ymax=1.1)

    fig_name = f"basin_size_ring_{suffix}_ic_{ic}_nreps_{n_reps}"

    plt.savefig(f"{run_dir}{fig_name}.png", dpi=300, bbox_inches="tight")

    print("============")
    print("done..")
    print("============")

    # Record the end time
    end_time = datetime.now()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Extract days, hours, and minutes
    days = elapsed_time.days
    seconds = elapsed_time.total_seconds()
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    # Format the elapsed time
    formatted_time = (
        f"{days} days {hours} hours {minutes} minutes {seconds:.0f} seconds"
    )

    # Print or save the elapsed time
    print(f"Execution time: {formatted_time}")

    with open(f"{run_dir}execution_report.txt", "w") as f:
        f.write(f"Execution time: {formatted_time}")
