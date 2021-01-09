import os
import sys
import tqdm
import pickle
import random
import argparse
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from concurrent import futures
from sacred import Experiment
from sacred.observers import MongoObserver

from porto_taxis import (
    PortoExtras,
    PortoFeatures,
    eval_mixture,
    eval_shortest_path,
    save_eval_results,
)

from pprint import pprint
from multimodal_irl.bv_em import MaxEntEMSolver, bv_em

from mdp_extras import padding_trick, PaddedMDPWarning


# Experiment config
# @ex.config
def base_config():

    # Initialization for the EM algorithm
    initialisation = "Random"

    # Rollout min, max length
    # 500 paths seems like a reasonable maximum path length
    rollout_minmaxlen = (0, 500)

    # Number of restarts to use for non-random initializations
    num_init_restarts = 5000

    # Tolerance for Negative Log Likelihood convergence
    em_nll_tolerance = 1e-2

    # Minimum and maximum reward parameter values
    reward_range = (-10, 0)

    # Tolerance for MaxEnt feature convergence (units of km)
    maxent_feature_tolerance = 1e-3

    # Number of learned clusters
    num_clusters = 3

    # Maximum number of EM iterations
    max_iterations = None

    # Replicate ID for this experiment
    replicate = 0


def poto_taxi_forecasting_v2(
    initialisation,
    rollout_minmaxlen,
    num_init_restarts,
    reward_range,
    em_nll_tolerance,
    maxent_feature_tolerance,
    num_clusters,
    max_iterations,
    _log,
    _run,
    _seed,
):
    result_fname = f"{_seed}.results"

    _log.info(f"{_seed}: Loading...")
    xtr = PortoExtras()
    phi = PortoFeatures(xtr=xtr)
    bin_prefix = os.path.join(os.path.dirname(__file__), "..", "porto_taxis", "bin")
    with open(os.path.join(bin_prefix, "path-data.pkl"), "rb") as file:
        all_rollouts = pickle.load(file)

    # Sub-sample rollouts
    short_rollouts = [
        r
        for r in all_rollouts
        if rollout_minmaxlen[0] <= len(r) <= rollout_minmaxlen[1]
    ]
    random.shuffle(short_rollouts)
    rollouts_train = short_rollouts[0 : len(short_rollouts) // 2]
    rollouts_test = short_rollouts[len(short_rollouts) // 2 :]

    _log.info(
        f"{_seed}: Got set of {len(rollouts_train)} training rollouts, {len(rollouts_test)} testing rollouts"
    )
    _log.info(
        f"{_seed}: Max training path length is {np.max([len(r) for r in rollouts_train])}"
    )

    _log.info(f"{_seed}: Solving...")
    if initialisation == "Baseline":
        learned_nlls, learned_paths, learned_fds, learned_pdms = eval_shortest_path(
            xtr, phi, rollouts_test
        )
        save_eval_results(
            result_fname,
            learned_nlls=learned_nlls,
            learned_paths=learned_paths,
            learned_fds=learned_fds,
            learned_pdms=learned_pdms,
        )
        _run.add_artifact(result_fname)
        os.remove(result_fname)
        return np.nan

    else:
        # Run MM-IRL experiment
        assert initialisation in ("Random", "KMeans", "GMM")

        # Apply padding trick
        xtr_p, rollouts_p_train = padding_trick(xtr, rollouts_train)

        # Prep solver
        _log.info(f"{_seed}: Loading MaxEnt solver...")
        solver = MaxEntEMSolver(
            # LBFGS convergence threshold (units of km)
            minimize_kwargs=dict(tol=maxent_feature_tolerance),
            pre_it=lambda i: _log.info(f"{_seed}: Starting iteration {i}"),
            # parallel_executor=futures.ThreadPoolExecutor(num_clusters),
        )

        _log.info(f"{_seed}: Initializing Mixture...")
        if initialisation == "Random":
            # Initialize randomly
            init_mode_weights, init_rewards = solver.init_random(
                phi, num_clusters, reward_range
            )
        elif initialisation == "KMeans":
            # Initialize with KMeans
            init_mode_weights, init_rewards = solver.init_kmeans(
                xtr,
                phi,
                rollouts_train,
                num_clusters,
                reward_range,
                num_restarts=num_init_restarts,
            )
        elif initialisation == "GMM":
            # Initialize with GMM
            init_mode_weights, init_rewards = solver.init_gmm(
                xtr,
                phi,
                rollouts_train,
                num_clusters,
                reward_range,
                num_restarts=num_init_restarts,
            )
        else:
            raise ValueError()

        # Evaluate initial mixture model
        _log.info(f"{_seed}: Evaluating initial solution...")
        init_nlls, init_paths, init_fds, init_pdms = eval_mixture(
            xtr, phi, init_mode_weights, init_rewards, rollouts_test
        )

        # Run actual BV EM algorithm
        _log.info(f"{_seed}: BV EM Loop...")
        (
            iterations,
            resp_history,
            mode_weights_history,
            rewards_history,
            nll_history,
            reason,
        ) = bv_em(
            solver,
            xtr_p,
            phi,
            rollouts_p_train,
            num_clusters,
            reward_range,
            mode_weights=init_mode_weights,
            rewards=init_rewards,
            tolerance=em_nll_tolerance,
            max_iterations=max_iterations,
        )
        iterations = int(len(resp_history))
        learned_resp = resp_history[-1]
        learned_mode_weights = mode_weights_history[-1]
        learned_rewards = rewards_history[-1]
        nll = float(nll_history[-1])

        # Log training NLLs as a metric
        for _nll in nll_history:
            _run.log_scalar("training.nll", float(_nll))

        _log.info(f"{_seed}: Iterations: {iterations}")
        _log.info(f"{_seed}: Responsibility Matrix")
        _log.info(learned_resp)
        _log.info(f"{_seed}: Mode Weights: {learned_mode_weights}")
        _log.info(f"{_seed}: Rewards: {[r.theta for r in learned_rewards]}")
        _log.info(f"{_seed}: Model NLL: {nll}")

        # Evaluate trained model
        _log.info(f"{_seed}: Evaluating trained model...")
        learned_nlls, learned_paths, learned_fds, learned_pdms = eval_mixture(
            xtr, phi, init_mode_weights, init_rewards, rollouts_test
        )

        save_eval_results(
            init_nlls=init_nlls,
            init_paths=init_paths,
            init_fds=init_fds,
            init_pdms=init_pdms,
            learned_nlls=learned_nlls,
            learned_paths=learned_paths,
            learned_fds=learned_fds,
            learned_pdms=learned_pdms,
            iterations=iterations,
            learned_resp=learned_resp,
            learned_mode_weights=learned_mode_weights,
            learned_rewards=learned_rewards,
            nll=nll,
            reason=reason,
        )
        _run.add_artifact(result_fname)
        os.remove(result_fname)

    _log.info(f"{_seed}: Done")

    return float(nll)


def run(config, mongodb_url="localhost:27017"):
    """Run a single experiment with the given configuration"""

    # Dynamically bind experiment config and main function
    ex = Experiment()
    ex.config(base_config)
    ex.main(poto_taxi_forecasting_v2)

    # Attach MongoDB observer if necessary
    if not ex.observers:
        ex.observers.append(MongoObserver(url=mongodb_url))

    # Suppress warnings about padded MPDs
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=PaddedMDPWarning)

        # Run the experiment
        run = ex.run(config_updates=config)  # , options={"--loglevel": "ERROR"})

    # Return the result
    return run.result


def main():
    """Main function"""

    # Run argparse to get arguments here
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-k",
        "--num_modes",
        required=False,
        default=3,
        type=int,
        help="Number of clusters to learn",
    )

    parser.add_argument(
        "-w",
        "--num_workers",
        required=False,
        default=None,
        type=int,
        help="Number of workers to use - if not provided, will be inferred from system and workload",
    )

    parser.add_argument(
        "-i",
        "--init",
        required=False,
        type=str,
        default="Random",
        choices=("Random", "KMeans", "GMM", "Baseline"),
        help="Initialisation method to use",
    )

    parser.add_argument(
        "-m",
        "--max_iterations",
        required=False,
        type=int,
        default=None,
        help="Maximum number of EM iterations",
    )

    parser.add_argument(
        "-N",
        "--num_replicates",
        required=False,
        type=int,
        default=5,
        help="Number of replicates to perform",
    )

    args = parser.parse_args()
    print("Arguments:", args, flush=True)

    _base_config = {
        "num_clusters": args.num_modes,
        "initialisation": args.init,
        "max_iterations": args.max_iterations,
    }
    print("META: Base configuration: ")
    pprint(_base_config)

    configs = []
    for replicate in range(args.num_replicates):
        _config = _base_config.copy()
        _config.update({"replicate": replicate})
        configs.append(_config)

    # Try and determine how many CPUs we are allowed to use
    num_cpus = (
        len(os.sched_getaffinity(0))
        # Ask the (linux) OS how many CPUs wer are scheduled to use
        if "sched_getaffinity" in dir(os)
        # If we can't find our scheduled number of CPUs, just use one less than the
        # system's physical socket count - leave one for GUI, bookkeeping etc.
        else os.cpu_count() - 1
    )

    print(f"META: {num_cpus} CPUs available")
    if args.num_workers is not None:
        num_workers = min(num_cpus, len(configs), args.num_workers)
    else:
        num_workers = min(num_cpus, len(configs))

    print(
        f"META: Distributing {args.num_replicates} replicate(s) over {num_workers} workers"
    )

    # Read MongoDB URL from config file, if it exists
    mongodb_config_file = "mongodb-config.txt"
    mongodb_url = "localhost:27017"
    if os.path.exists(mongodb_config_file):
        print(f"META: Reading MongoDB config from {mongodb_config_file}")
        with open(mongodb_config_file, "r") as file:
            mongodb_url = file.readline()
    print(f"META: MongoDB Server URL: {mongodb_url}")

    # Parallel loop
    with tqdm.tqdm(total=len(configs)) as pbar:
        with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            tasks = {executor.submit(run, config, mongodb_url) for config in configs}
            for future in futures.as_completed(tasks):
                # Use arg or result here if desired
                # arg = tasks[future]
                # result = future.result()
                pbar.update(1)

    # Non-parallel loop for debugging
    # for config in tqdm.tqdm(configs):
    #     run(config, mongodb_url)

    print("META: Finished replicate sweep")


if __name__ == "__main__":
    main()
