#!/usr/bin/env python

"""
Call with 
./basin_size_nb.py --num_threads 8 -n 1000 -t 300
"""

import matplotlib as mpl

mpl.use("agg")

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

# from basin_size_nb import simulate_iteration

sb.set_theme(style="ticks", context="notebook")

results_dir = "../results/"
data_dir = "../data/"

Path(results_dir).mkdir(parents=True, exist_ok=True)
Path(data_dir).mkdir(parents=True, exist_ok=True)


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
            **options,
        )

        nrep_thetas[j] = thetas[:, -1]

        if (j <= 5) or (
            "cluster" in identify_state(thetas)
        ):  # ) or (("other" in identify_state(thetas) and j <= 500)):
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
        "-n", "--n_reps", type=int, default=100, help="Number of repetitions"
    )
    parser.add_argument("-t", "--t_end", type=float, default=300, help="End time")

    parser.add_argument(
        "-i", "--integrator", type=str, default="RK45", help="ODE integrator"
    )

    args = parser.parse_args()

    n_reps = args.n_reps  # number of random realisations

    # generate structure
    N = 42
    # H = xgi.complete_hypergraph(N, max_order=2)

    r1 = 2
    r2 = 2

    suffix = "ring_SC"  # "SC"

    # H = xgi.trivial_hypergraph(N)
    ps = 20 * np.array([1 / N, 1 / N**2])
    H = xgi.random_hypergraph(N, ps)
    # H = ring_dihypergraph(N, r1, r2)

    # links = H.edges.filterby("size", 2).members()
    # triangles = H.edges.filterby("size", 3).members()

    # define parameters

    # dynamical
    k1 = 1  # pairwise coupling strength
    k2 = 4  # s = np.arange(0, 4.5, 0.5)  # triplet coupling strength
    omega = 0  # 1 * np.ones(N)  # np.random.normal(size=N) #1 * np.ones(N)

    ic = "random"  # initial condition type, see below
    noise = 1e-1  # noise strength

    # integration
    t_end = args.t_end  # default = 200
    dt = 0.01
    times = np.arange(0, t_end + dt / 2, dt)

    t_eval = False  # whether to integrate at all above timepoints
    integrator = args.integrator
    options = {"atol": 1e-8, "rtol": 1e-8}

    tag_params = f"k1_{k1}_k2s_ic_{ic}_tend_{t_end}_nreps_{n_reps}_{suffix}"

    # create directory for this run
    run_dir = f"{results_dir}run_test_{tag_params}/"
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    # copy current script
    current_script_path = Path(__file__).resolve()
    shutil.copy2(current_script_path, run_dir)

    print("============")
    print("simulating..")
    print("============")

    # Create a multiprocessing Pool

    args = (r1, r2)

    i, thetas_out = simulate_iteration(
        0,
        H,
        k1,
        k2,
        omega,
        t_end,
        dt,
        ic,
        noise,
        rhs_ring_nb,  # change rhs here
        integrator,
        args,
        t_eval,
        n_reps,
        run_dir,
        **options,
    )

    thetas_arr = thetas_out

    print("============")
    print("saving simulations..")
    print("============")

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
