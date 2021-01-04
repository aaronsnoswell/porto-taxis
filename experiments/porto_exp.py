import os
import sys
import tqdm
import pickle
import random
import argparse
import warnings

sys.path.append("D:\\Development\\porto-taxis")

import numpy as np

from concurrent import futures
from sacred import Experiment
from sacred.observers import MongoObserver

from porto_taxis import PortoExtras, PortoFeatures, PortoInference

from pprint import pprint
from unimodal_irl import maxent_path_logprobs
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
    em_nll_tolerance = 1e-3

    # Minimum and maximum reward parameter values
    reward_range = (-10, 0)

    # Tolerance for MaxEnt feature convergence (units of km)
    maxent_feature_tolerance = 1e-3

    # Number of learned clusters
    num_clusters = 3

    # Replicate ID for this experiment
    replicate = 0


def poto_taxi_forecasting(
    initialisation,
    rollout_minmaxlen,
    num_init_restarts,
    reward_range,
    em_nll_tolerance,
    maxent_feature_tolerance,
    num_clusters,
    _log,
    _run,
):

    _log.info("Initializing")

    # About 8 seconds to load
    xtr = PortoExtras()
    # About 35 seconds to load
    phi = PortoFeatures(xtr=xtr)

    # About 0.6 second to load
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
        f"Got set of {len(rollouts_train)} training rollouts, {len(rollouts_test)} testing rollouts"
    )
    _log.info(
        "Max training path length is {}".format(
            np.max([len(r) for r in rollouts_train])
        )
    )

    # A very small test set for debugging purposes
    # XXX REMOVE ME TEST TODO
    rollouts_test = rollouts_test[0:10]

    _log.info("Solving...")
    if initialisation == "Baseline":

        # Run shortest path baseline

        paths = []
        fds = []
        pdms = []
        for gt_path in tqdm.tqdm(rollouts_test):
            # Get start, end state
            s1 = gt_path[0][0]
            sg = gt_path[-1][0]

            # Query MDP for shortest (distance) path as a baseline
            shortest_path = xtr.shortest_path(s1, sg)
            fds.append(
                phi.feature_distance_metric(shortest_path, gt_path, gamma=xtr.gamma)
            )
            pdms.append(xtr.percent_distance_missed_metric(shortest_path, gt_path))
            paths.append(shortest_path)

        results = dict(
            # Fill a dummy array with NLL values
            nlls=[np.nan for _ in range(len(rollouts_test))],
            paths=paths,
            pdms=pdms,
            fds=fds,
            iterations=np.nan,
            resp_history=[],
            mode_weights_history=[],
            rewards_history=[],
            nll_history=[],
            reason="",
        )

    else:
        # Run MM-IRL experiment

        assert initialisation in ("Random", "KMeans", "GMM")

        # Apply padding trick
        xtr_p, rollouts_p_train = padding_trick(xtr, rollouts_train)

        # Prep solver
        print("Loading MaxEnt solver...")
        solver = MaxEntEMSolver(
            # LBFGS convergence threshold (units of km)
            minimize_kwargs=dict(tol=maxent_feature_tolerance),
            minimize_options=dict(disp=True),
        )

        print("Initializing...")
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

        print("Solving...")
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
            max_iterations=1,
        )
        iterations = len(resp_history)
        learned_resp = resp_history[-1]
        learned_mode_weights = mode_weights_history[-1]
        learned_rewards = rewards_history[-1]
        nll = nll_history[-1]

        print(f"Iterations: {iterations}")
        print("Responsibility Matrix")
        print(learned_resp)
        print(f"Mode Weights: {learned_mode_weights}")
        print(f"Rewards: {[r.theta for r in learned_rewards]}")
        print(f"Model NLL: {nll}")

        def mixture_ml_path(mode_weights, models, rewards, s1, sg):
            """Find ML path from start state to goal under a mixture model
            
            Solving the actual optimization problem for the true ML path is *hard*.
            Instead, we choose the path that has the highest likelihood under it's personal
            mixture component. See the paper for details.
            
            Args:
                mode_weights (numpy array): Weights for each mixture component
                models (list): List of PortoInference() models - one for each mixture
                    component
                rewards (list): List of Lienar() - one for each mixture component
                s1 (int): Starting state
                sg (int): Goal state
            """
            candidate_paths = []
            candidate_path_probs = []

            for rho, mdl, reward in zip(mode_weights, models, rewards):
                path = mdl.ml_path(s1, sg)
                path_prob = rho * np.exp(maxent_path_logprobs(xtr, phi, reward, [path]))
                candidate_paths.append(path)
                candidate_path_probs.append(path_prob)

            # Select the highest probability path
            path_idx = np.argmax(candidate_path_probs)
            return candidate_paths[path_idx]

        # NLL for each path is computed per-reward for efficiency reasons
        print("Evaluating NLLs...")
        mode_nlls = []
        for reward in learned_rewards:
            mode_nlls.append(
                -1.0 * maxent_path_logprobs(xtr, phi, reward, rollouts_test)
            )
        nlls = np.average(mode_nlls, axis=0, weights=learned_mode_weights)

        # Prepare mixture of inference models
        print("Preparing mixture for inference...")
        models = []
        for r in learned_rewards:
            models.append(PortoInference(xtr, phi, r.theta))

        print("Evaluating ML paths...")
        paths = []
        fds = []
        pdms = []
        for gt_path in tqdm.tqdm(rollouts_test):
            # Get start, end state
            s1 = gt_path[0][0]
            sg = gt_path[-1][0]

            # Query mixture model for ML path
            model_path = mixture_ml_path(
                learned_mode_weights, models, learned_rewards, s1, sg
            )
            fds.append(
                phi.feature_distance_metric(model_path, gt_path, gamma=xtr.gamma)
            )
            pdms.append(xtr.percent_distance_missed_metric(model_path, gt_path))
            paths.append(model_path)

        results = dict(
            nlls=nlls.tolist(),
            paths=paths,
            pdms=pdms,
            fds=fds,
            iterations=int(iterations),
            resp_history=resp_history.tolist(),
            mode_weights_history=mode_weights_history.tolist(),
            rewards_history=[
                learned_r.theta.tolist()
                for learned_reward_mixture in rewards_history
                for learned_r in learned_reward_mixture
            ],
            nll_history=nll_history.tolist(),
            reason=reason,
        )

    _log.info("Done")

    return results


def run(config, mongodb_url="localhost:27017"):
    """Run a single experiment with the given configuration"""

    # Dynamically bind experiment config and main function
    ex = Experiment()
    ex.config(base_config)
    ex.main(poto_taxi_forecasting)

    # Attach MongoDB observer if necessary
    if not ex.observers:
        ex.observers.append(MongoObserver(url=mongodb_url))

    # Suppress warnings about padded MPDs
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=PaddedMDPWarning)

        # Run the experiment
        run = ex.run(config_updates=config, options={"--loglevel": "ERROR"})

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
        "-N",
        "--num_replicates",
        required=False,
        type=int,
        default=10,
        help="Number of replicates to perform",
    )

    args = parser.parse_args()
    print("Arguments:", args, flush=True)

    _base_config = {
        "num_clusters": args.num_modes,
        "initialisation": args.init,
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

    # # Parallel loop
    # with tqdm.tqdm(total=len(configs)) as pbar:
    #     with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    #         tasks = {executor.submit(run, config, mongodb_url) for config in configs}
    #         for future in futures.as_completed(tasks):
    #             # Use arg or result here if desired
    #             # arg = tasks[future]
    #             # result = future.result()
    #             pbar.update(1)

    # Non-parallel loop for debugging
    for config in tqdm.tqdm(configs):
        run(config, mongodb_url)

    print("META: Finished replicate sweep")


if __name__ == "__main__":
    main()