import os
import sys
import tqdm
import pickle
import random
import argparse
import warnings

sys.path.append(os.path.join(__file__, ".."))

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

    # Maximum number of paths to use for testing
    # Each test path takes about 30 seconds to evaluate
    max_num_testpaths = 250

    # Tolerance for Negative Log Likelihood convergence
    em_nll_tolerance = 1e-3

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


def mixture_ml_path(xtr, phi, mode_weights, models, rewards, s1, sg):
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


def poto_taxi_forecasting(
    initialisation,
    rollout_minmaxlen,
    num_init_restarts,
    max_num_testpaths,
    reward_range,
    em_nll_tolerance,
    maxent_feature_tolerance,
    num_clusters,
    max_iterations,
    _log,
    _run,
    _seed,
):

    _log.info(f"{_seed}: Loading...")

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
        f"{_seed}: Got set of {len(rollouts_train)} training rollouts, {len(rollouts_test)} testing rollouts"
    )
    _log.info(
        f"{_seed}: Max training path length is {np.max([len(r) for r in rollouts_train])}"
    )

    # Truncate testing path set
    rollouts_test = rollouts_test[0:max_num_testpaths]

    _log.info(f"{_seed}: Solving...")
    if initialisation == "Baseline":

        # Run shortest path baseline

        _log.info(f"{_seed}: Evaluating ML paths...")
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
                float(
                    phi.feature_distance_metric(shortest_path, gt_path, gamma=xtr.gamma)
                )
            )
            pdms.append(
                float(xtr.percent_distance_missed_metric(shortest_path, gt_path))
            )
            paths.append(shortest_path)

        results = dict(
            # We don't store any initial solution values for baseline models
            init_mode_weights=[],
            init_rewards=[],
            init_paths=[],
            init_fds=[],
            init_pdms=[],
            # Fill a dummy array with NLL values
            nlls=[np.nan for _ in range(len(rollouts_test))],
            paths=[np.array(p).tolist() for p in paths],
            pdms=pdms,
            fds=fds,
            iterations=np.nan,
            learned_resp=[],
            learned_mode_weights=[],
            learned_rewards=[],
            nll=np.nan,
            reason="",
        )

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

        # ============================================ Evaluate initial mixture model

        # NLL for each path is computed per-reward for efficiency reasons
        _log.info("Evaluating initial model NLLs...")
        maxent_path_logprobs

        # Prepare mixture of inference models
        _log.info(f"{_seed}: Preparing initial mixture for inference...")
        init_models = []
        for r in init_rewards:
            init_models.append(PortoInference(xtr, phi, r.theta))

        init_paths = []
        init_fds = []
        init_pdms = []
        for gt_path in rollouts_test:  # tqdm.tqdm(rollouts_test):
            # Get start, end state
            s1 = gt_path[0][0]
            sg = gt_path[-1][0]

            # Query mixture model for ML path
            init_model_path = mixture_ml_path(
                xtr, phi, init_mode_weights, init_models, init_rewards, s1, sg
            )
            init_fds.append(
                float(
                    phi.feature_distance_metric(
                        init_model_path, gt_path, gamma=xtr.gamma
                    )
                )
            )
            init_pdms.append(
                float(xtr.percent_distance_missed_metric(init_model_path, gt_path))
            )
            init_paths.append(init_model_path)

        # ============================================ Run actual BV EM algorithm

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
        iterations = len(resp_history)
        learned_resp = resp_history[-1]
        learned_mode_weights = mode_weights_history[-1]
        learned_rewards = rewards_history[-1]
        nll = nll_history[-1]

        # Log training values NLLs as metrics
        for _learned_mode_weights in mode_weights_history:
            _run.log_scalar("training.mode_weights", _learned_mode_weights.tolist())
        for _learned_rewards in rewards_history:
            _run.log_scalar(
                "training.rewards", [r.theta.tolist() for r in _learned_rewards]
            )
        for _nll in nll_history:
            _run.log_scalar("training.nll", float(_nll))

        _log.info(f"{_seed}: Iterations: {iterations}")
        _log.info(f"{_seed}: Responsibility Matrix")
        _log.info(learned_resp)
        _log.info(f"{_seed}: Mode Weights: {learned_mode_weights}")
        _log.info(f"{_seed}: Rewards: {[r.theta for r in learned_rewards]}")
        _log.info(f"{_seed}: Model NLL: {nll}")

        # ============================================ Evaluate trained model

        # NLL for each path is computed per-reward for efficiency reasons
        _log.info("Evaluating NLLs...")
        mode_nlls = []
        for reward in learned_rewards:
            mode_nlls.append(
                -1.0 * maxent_path_logprobs(xtr, phi, reward, rollouts_test)
            )
        nlls = np.average(mode_nlls, axis=0, weights=learned_mode_weights)

        # Prepare mixture of inference models
        _log.info(f"{_seed}: Preparing mixture for inference...")
        models = []
        for r in learned_rewards:
            models.append(PortoInference(xtr, phi, r.theta))

        _log.info(f"{_seed}: Evaluating ML paths...")
        paths = []
        fds = []
        pdms = []
        for gt_path in tqdm.tqdm(rollouts_test):
            # Get start, end state
            s1 = gt_path[0][0]
            sg = gt_path[-1][0]

            # Query mixture model for ML path
            model_path = mixture_ml_path(
                xtr, phi, learned_mode_weights, models, learned_rewards, s1, sg
            )
            fds.append(
                float(phi.feature_distance_metric(model_path, gt_path, gamma=xtr.gamma))
            )
            pdms.append(float(xtr.percent_distance_missed_metric(model_path, gt_path)))
            paths.append(model_path)

        results = dict(
            # Store initial solution values for baseline models
            init_nlls=init_nlls,
            init_paths=init_paths,
            init_fds=init_fds,
            init_pdms=init_pdms,
            #
            nlls=nlls.tolist(),
            paths=[np.array(p).tolist() for p in paths],
            pdms=pdms,
            fds=fds,
            iterations=int(iterations),
            learned_resp=learned_resp.tolist(),
            learned_mode_weights=learned_mode_weights.tolist(),
            learned_rewards=[learned_r.theta.tolist() for learned_r in learned_rewards],
            nll=float(nll),
            reason=reason,
        )

    _log.info(f"{_seed}: Done")

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

    # # Non-parallel loop for debugging
    # for config in tqdm.tqdm(configs):
    #     run(config, mongodb_url)

    print("META: Finished replicate sweep")


if __name__ == "__main__":
    main()
