import copy
import json
import logging
import pathlib
import time

import hydra
import numpy as np
import scipy.stats
import yaml

from skanas.infra import script_utils


# also log the warnings
logging.captureWarnings(True)
_logger = logging.getLogger("skanas.run_surrogate_regression")

PROJECT_ROOT = pathlib.Path.cwd()
CONFIG_PATH = PROJECT_ROOT / "configs"

CONFIG_NAME = "run_surrogate_regression"
HYDRA_VERSION_BASE = "1.2"


def verify_arg_values(project_root, args):
    # get the absolute path to the data_path
    # since the current working directory is handled by hydra
    absolute_data_path = script_utils.get_absolute_data_path(
        data_path=args.data_path, project_root=project_root
    )
    args.data_path = absolute_data_path

    # if instead of a list of seeds, a single value is given
    try:
        args.seeds = [int(args.seeds)]
    except (ValueError, TypeError):
        args.seeds = [int(i) for i in args.seeds]

    # if instead of a list of n_train_values, a single value is given
    try:
        args.n_train_values = [int(args.n_train_values)]
    except (ValueError, TypeError):
        args.n_train_values = [int(i) for i in args.n_train_values]

    try:
        args.n_test = int(args.n_test)
    except (ValueError, TypeError):
        raise ValueError(
            f"Arg `n_test` needs to be an integer. Received {args.n_test!r}"
        )

    if args.y_log not in {True, False}:
        raise ValueError(f"Arg `y_log` needs to be a boolean. Received {args.y_log!r}")
    if args.rs_only not in {True, False}:
        raise ValueError(
            f"Arg `rs_only` needs to be a boolean. Received {args.rs_only!r}"
        )


def read_configs_and_y_data(
    data_path: pathlib.Path,
    y_log: bool,
    rs_only: bool,
    take_only_first_n_configs: int = 100,
):
    configs, y = [], []

    # expected directory structure example:
    # "data_path/bayesian_optimization_gp_evolution_pool200/888/results"
    # "data_path/f/s/results"

    # want to iterate all available methods (exist as subdirectories in data_path)
    for f in data_path.iterdir():
        # skip not relevant items (image files, etc.)
        if (
            not f.is_dir()
            or (rs_only and "random_search" != f.name)
            or ("fixed_1_none" in str(data_path) and "gp_evo" in f.name)
            or "naswot" in f.name
        ):
            continue

        # want to iterate all seeds for a method
        for s in f.iterdir():
            # skip not relevant items (image files, etc.)
            if not s.is_dir() or "bug" in s.name:
                continue

            results_dir = data_path / f / s / "results"
            _logger.info("Found results dir: %s", results_dir)

            configs_to_take = list(range(1, take_only_first_n_configs + 1))
            previous_results = dict()

            for config_dir in results_dir.iterdir():
                result_file = config_dir / "result.yaml"
                if result_file.exists():
                    with result_file.open("rb") as results_file_stream:
                        result = yaml.safe_load(results_file_stream)
                    config_file = config_dir / "config.yaml"
                    with config_file.open("rb") as config_file_stream:
                        config = yaml.safe_load(config_file_stream)
                    config_id = int(config_dir.name.split("config_")[1])
                    previous_results[config_id] = {"config": config, "result": result}

            if any(conf_id not in previous_results for conf_id in configs_to_take):
                err_msg = (
                    "Missing config in previous results: "
                    + f"{set(configs_to_take) - set(previous_results.keys())}"
                )
                raise Exception(err_msg)

            # add the seen configs
            configs += [previous_results[i]["config"] for i in configs_to_take]

            # add the seen y values
            if "cifar10" in str(results_dir) or "ImageNet" in str(results_dir):
                y += [
                    1 - previous_results[i]["result"]["info_dict"]["x-valid_1"] / 100
                    for i in configs_to_take
                ]
            else:
                y += [
                    previous_results[i]["result"]["info_dict"]["val_score"]
                    for i in configs_to_take
                ]

    if y_log:
        y = np.log(y)

    return configs, y


def get_x_y_train_test_split(n_train, n_test, search_space, configs, y):
    indices = list(range(len(configs)))
    np.random.shuffle(indices)

    # avoid overlap between train and test indices
    if n_train + n_test > len(configs):
        raise ValueError(
            "Not enough configs. "
            + f"Wanted {n_train} train indices and {n_test} test indices. "
            + f"{len(configs)} configs available"
        )

    train_indices = indices[:n_train]
    # take the test indices from the end
    # in order to have a stable config test set
    test_indices = indices[-n_test:]

    x_train, y_train = [], []
    for idx in train_indices:
        copied_search_space = copy.deepcopy(search_space)
        copied_search_space.load_from(configs[idx])
        x_train.append(copied_search_space)
        y_train.append(y[idx])

    x_test, y_test = [], []
    for idx in test_indices:
        copied_search_space = copy.deepcopy(search_space)
        copied_search_space.load_from(configs[idx])
        x_test.append(copied_search_space)
        y_test.append(y[idx])

    return x_train, y_train, x_test, y_test


def train_and_evaluate_surrogate(
    *, n_train, n_test, configs, y, search_space, surrogate_model
):
    x_train, y_train, x_test, y_test = get_x_y_train_test_split(
        n_train=n_train,
        n_test=n_test,
        search_space=search_space,
        configs=configs,
        y=y,
    )

    # train surrogate
    fitting_start_time = time.monotonic()
    surrogate_model.fit(train_x=x_train, train_y=y_train)
    fitting_time = time.monotonic() - fitting_start_time
    _logger.info("Fitting took %.2f seconds", fitting_time)

    # evaluate surrogate
    predicting_start_time = time.monotonic()
    y_pred, y_pred_var = surrogate_model.predict(x_test)
    predicting_time = time.monotonic() - predicting_start_time
    _logger.info("Predicting took %.2f seconds", predicting_time)

    (y_pred, y_pred_var) = (
        y_pred.cpu().detach().numpy(),
        y_pred_var.cpu().detach().numpy(),
    )

    # evaluate regression performance
    pearson = scipy.stats.pearsonr(y_test, y_pred)[0]
    spearman = scipy.stats.spearmanr(y_test, y_pred)[0]
    kendalltau = scipy.stats.kendalltau(y_test, y_pred)[0]
    y_pred_std = np.sqrt(y_pred_var)
    nll = -np.mean(
        scipy.stats.norm.logpdf(np.array(y_test), loc=y_pred, scale=y_pred_std)
    )

    return {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "kendalltau": float(kendalltau),
        "nll": float(nll),
        "y_pred": [float(y_) for y_ in y_pred],
        "y_test": [float(y_) for y_ in y_test],
        "fitting_time": fitting_time,
        "predicting_time": predicting_time,
        "total_time": fitting_time + predicting_time,
    }


@hydra.main(
    config_path=str(CONFIG_PATH),
    config_name=CONFIG_NAME,
    version_base=HYDRA_VERSION_BASE,
)
def main(args):
    verify_arg_values(project_root=PROJECT_ROOT, args=args)

    # log the args and other related info
    script_utils.log_runtime_info(
        logger=_logger, project_root=PROJECT_ROOT, hydra_args=args
    )

    configs, y = read_configs_and_y_data(
        data_path=pathlib.Path(args.data_path),
        y_log=args.y_log,
        rs_only=args.rs_only,
    )
    _logger.info("Loaded %d configs", len(configs))
    _logger.info("Loaded %d y values", len(y))

    for n_train_value in args.n_train_values:
        _logger.info("Running for n_train value %r", n_train_value)

        result_filename_prefix = "surrogate" if not args.rs_only else "rsOnly_surrogate"
        result_filename = (
            f"{result_filename_prefix}_{args.surrogate_model.name}_{n_train_value}.json"
        )
        result_file_path = pathlib.Path.cwd() / result_filename
        _logger.info("Result file path: %s", result_file_path)
        if result_file_path.exists():
            _logger.debug("Removing existing result file %s", result_file_path)
            result_file_path.unlink()

        regression_perf_by_seed = {}

        for seed in args.seeds:
            _logger.info("Running for seed %r", seed)
            start_time = time.monotonic()

            # code below depends on the random seed being set previously
            script_utils.set_seeds(seed=seed)

            search_space_partial = hydra.utils.instantiate(args.search_space.model)
            search_space = search_space_partial(args=args)
            _logger.debug("Search space: %s", search_space)

            surrogate_model_partial = hydra.utils.instantiate(
                args.surrogate_model.bo_model,
            )
            (
                surrogate_model_class,
                surrogate_model_args,
            ) = surrogate_model_partial(args=args)
            surrogate_model = surrogate_model_class(**surrogate_model_args)
            _logger.debug("Surrogate model: %s", surrogate_model)

            perf = train_and_evaluate_surrogate(
                n_train=n_train_value,
                n_test=args.n_test,
                configs=configs,
                y=y,
                search_space=search_space,
                surrogate_model=surrogate_model,
            )
            assert len(perf["y_test"]) == args.n_test, (
                "Missmatched `args.n_test` and prediction result size: "
                + f"args.n_test={args.n_test}, result size={len(perf['y_test'])}"
            )

            _logger.info(
                f"seed={seed}, "
                + f"n_train={n_train_value}, n_test={args.n_test}: "
                + f"pearson={perf['pearson']:.3f}, spearman={perf['spearman']:.3f}, "
                + f"kendalltau={perf['kendalltau']:.3f}, NLL={perf['nll']}"
            )

            regression_perf_by_seed[seed] = perf
            _logger.info(
                "Run for seed %d took %.2f seconds",
                seed,
                (time.monotonic() - start_time),
            )

        with open(result_file_path, "w", encoding="utf-8") as result_file_obj:
            json.dump(regression_perf_by_seed, result_file_obj, indent=2)

    _logger.info("Finished")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
