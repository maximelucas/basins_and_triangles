#!/usr/bin/env python

"""
Call with 
./basins_max.py --num_threads 8 -n 1000 -t 300
"""

# imports
import argparse
import multiprocessing
import shutil
import sys
from itertools import combinations
from math import sin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import xgi
from kuramoto_hoi import *
from tqdm import tqdm

sb.set_theme(style="ticks", context="notebook")

results_dir = "../results/"
data_dir = "../data/"

Path(results_dir).mkdir(parents=True, exist_ok=True)
Path(data_dir).mkdir(parents=True, exist_ok=True)


def nearest_neighbors(N, d, r, kind=None):
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

    Raises:
    ------
    """

    r_min = int(np.ceil((d - 1) / 2))
    if r < r_min:
        raise ValueError(f"r must be >= ceil((d - 1) / 2) = {r_min} to have edges.")

    H = xgi.Hypergraph()

    nodes = np.arange(N)

    edges = []
    neighbor_rel_ids = np.concatenate((np.arange(-r, 0), np.arange(1, r + 1)))

    for i in nodes:
        neighbor_ids = i + neighbor_rel_ids
        edge_neighbors_i = combinations(neighbor_ids, d - 1)
        if kind == "strict":
            edge_neighbors_i = [el for el in edge_neighbors_i if max(el) - min(el) <= r]
        edges_i = [(i, *comb) for comb in edge_neighbors_i]
        edges = edges + edges_i

    edges = np.mod(edges, N)

    H.add_nodes_from(nodes)
    H.add_edges_from(edges)
    H.cleanup()  # remove duplicate edges
    return H


# dynamics
def rhs_pairwise_triplet_all_sym(t, psi, omega, k1, k2):
    out = (
        rhs_pairwise_all(t, psi, omega, k1, k2)
        + rhs_triplet_all_sym(t, psi, omega, k1, k2)
        - omega
    )

    return out


def rhs_pairwise_triplet_all_asym(t, psi, omega, k1, k2):
    out = (
        rhs_pairwise_all(t, psi, omega, k1, k2)
        + rhs_triplet_all_asym(t, psi, omega, k1, k2)
        - omega
    )

    return out


def rhs_23_nn_sym(t, psi, omega, k1, k2, r1, r2, adj1, triangles):
    N = len(psi)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    pairwise = adj1.dot(sin_psi) * cos_psi - adj1.dot(cos_psi) * sin_psi

    triplet = np.zeros(N)

    for i, j, k in triangles:
        # sin(2 oj - ok - oi)
        oi = psi[i]
        oj = psi[j]
        ok = psi[k]
        triplet[i] += 2 * sin(oj + ok - 2 * oi)
        triplet[j] += 2 * sin(oi + ok - 2 * oj)
        triplet[k] += 2 * sin(oj + oi - 2 * ok)

    # sum_cos_psi = adj2.dot(cos_psi)
    # sum_sin_psi = adj2.dot(sin_psi)

    # # oj + ok - 2oi
    # triplet = (
    #     -2 * sum_cos_psi**2 * sin_psi * cos_psi
    #     + cos_psi**2 * sum_cos_psi * sum_sin_psi
    #     - sum_cos_psi * sin_psi**2 * sum_sin_psi
    #     + sum_cos_psi * cos_psi**2 * sum_sin_psi
    #     - sum_cos_psi * sin_psi**2 * sum_sin_psi
    #     + 2 * cos_psi * sin_psi * sum_sin_psi**2
    # )

    g1 = r1
    g2 = r2 * (2 * r2 - 1)

    return omega + (k1 / g1) * pairwise + (k2 / g2) * triplet / 2


# simulate system

# simulate
# kwargs = {
#    "links": links,
#    "triangles": triangles,
#    "k1_avg": k1_avg,
#    "k2_avg": k2_avg,
# }


def simulate_iteration(i, H, k1, k2, omega, t_end, dt, ic, noise, rhs, n_reps, run_dir="", **kwargs):
    N = len(H)
    times = np.arange(0, t_end + dt / 2, dt)

    nrep_thetas = np.zeros((n_reps, N))  # , len(times)))

    for j in range(n_reps):
        psi_init = generate_state(N, kind=ic, noise=noise)

        thetas, times = simulate_kuramoto(
            H,
            k1,
            k2,
            omega=omega,
            theta_0=psi_init,
            t_end=t_end,
            dt=dt,
            rhs=rhs,  # rhs_pairwise_all  #rhs_triplet_all_asym
            **kwargs,
        )

        nrep_thetas[j] = thetas[:, -1]

        if j <= 5:
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
    parser.add_argument(
        "-t", "--t_end", type=float, default=300, help="Number of repetitions"
    )

    args = parser.parse_args()

    n_reps = args.n_reps  # number of random realisations

    # generate structure
    N = 100
    # H = xgi.complete_hypergraph(N, max_order=2)

    r1 = 2
    r2 = 2

    H2 = nearest_neighbors(N, d=3, r=r2, kind=None)
    H1 = nearest_neighbors(N, d=2, r=r1, kind=None)
    H = H1 << H2
    # define parameters

    # dynamical
    k1 = 1  # pairwise coupling strength
    k2s = np.arange(0, 4.5, 0.25)  # triplet coupling strength
    omega = 1 * np.ones(N)  # np.random.normal(size=N) #1 * np.ones(N)

    ic = "random"  # initial condition type, see below
    noise = 1e-1  # noise strength

    # integration
    t_end = args.t_end  # default = 200
    dt = 0.01
    times = np.arange(0, t_end + dt / 2, dt)

    # may be used in the simulation function
    links = H.edges.filterby("size", 2).members()
    triangles = H.edges.filterby("size", 3).members()
    adj1 = xgi.adjacency_matrix(H, order=1)
    adj2 = xgi.adjacency_matrix(H, order=2)
    k1_avg = H.nodes.degree(order=1).mean()
    k2_avg = H.nodes.degree(order=2).mean()

    tag_params = f"ring_k1_{k1}_k2s_ic_{ic}_tend_{t_end}_nreps_{n_reps}"

    # create directory for this run
    run_dir = f"{results_dir}run_{tag_params}/"
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    # copy current script 
    current_script_path = Path(__file__).resolve()
    shutil.copy2(current_script_path, run_dir)


    kwargs = {
        #    "links": links,
        "triangles": triangles,
        #    "k1_avg": k1_avg,
        #    "k2_avg": k2_avg
        "r1": r1,
        "r2": r2,
        "adj1": adj1,
        #    "adj2": adj2,
    }

    print("============")
    print("simulating..")
    print("============")

    # Create a multiprocessing Pool
    with multiprocessing.Pool(processes=args.num_threads) as pool:
        results = []
        for i, k2 in enumerate(k2s):
            results.append(
                pool.apply_async(
                    simulate_iteration,
                    (i, H, k1, k2, omega, t_end, dt, ic, noise, rhs_23_nn_sym, n_reps, run_dir),
                    kwargs,
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

    thetas_arr = thetas_arr.swapaxes(0, 1) # get (n_rep, k2s)
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

    title = f"ring, {ic} ic, {n_reps} reps \n rhs_23_nn_sym"
    ax.set_title(title)

    sb.despine()
    ax.set_ylim(ymax=1.1)

    fig_name = f"basin_size_a2a_ic_{ic}_nreps_{n_reps}_rhs_pairwise_triplet_all_asym"

    plt.savefig(f"{run_dir}{fig_name}.png", dpi=300, bbox_inches="tight")

    print("============")
    print("done..")
    print("============")
