import logging
import pathlib

import hydra

import neps
from skanas.infra import script_utils


# also log the warnings
logging.captureWarnings(True)
_logger = logging.getLogger("skanas.run_search")

PROJECT_ROOT = pathlib.Path.cwd()
CONFIG_PATH = PROJECT_ROOT / "configs"

CONFIG_NAME = "run_search"
HYDRA_VERSION_BASE = "1.2"


def verify_arg_values(project_root, args):
    # get the absolute path to the data_path
    # since the current working directory is handled by hydra
    absolute_data_path = script_utils.get_absolute_data_path(
        data_path=args.data_path, project_root=project_root
    )
    args.data_path = absolute_data_path

    args.seed = int(args.seed)
    args.n_init = int(args.n_init)
    args.max_evaluations_total = int(args.max_evaluations_total)

    if args.max_evaluations_per_run is not None:
        args.max_evaluations_per_run = int(args.max_evaluations_per_run)

    if args.y_log not in {True, False}:
        raise ValueError(f"Arg `y_log` needs to be a boolean. Received {args.y_log!r}")

    # `pool_size` is currently used in the experiment path,
    #   but its value is fixed to 200
    #   (the default for the `Evolution` acquisition_sampler).
    # Can be removed or adjusted to be taken into account.
    args.pool_size = int(args.pool_size)
    if args.pool_size != 200:
        raise ValueError("The pool_size value is fixed and cannot be set.")


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

    seed = args.seed
    _logger.info("Running for seed %r", seed)

    # code below depends on the random seed being set previously
    script_utils.set_seeds(seed=seed)

    search_space_partial = hydra.utils.instantiate(args.search_space.model)
    search_space = search_space_partial(args=args)
    _logger.debug("Search space: %s", search_space)

    surrogate_model_partial = hydra.utils.instantiate(args.surrogate_model.bo_model)
    (
        surrogate_model_class,
        surrogate_model_args,
    ) = surrogate_model_partial(args=args)
    _logger.debug("Surrogate model: %s", surrogate_model_class)

    objective_partial = hydra.utils.instantiate(args.objective.model)
    objective = objective_partial(args=args)
    _logger.debug("Objective: %s", objective)

    if hasattr(objective, "set_seed"):
        objective.set_seed(args.seed)

    patience = args.search_space.patience
    working_directory = pathlib.Path.cwd()
    searcher = "bayesian_optimization"
    acquisition = args.acquisition.name
    acquisition_sampler = args.acquisition_sampler.name
    max_evaluations_total = args.max_evaluations_total
    max_evaluations_per_run = args.max_evaluations_per_run
    initial_design_size = args.n_init

    neps.run(
        run_pipeline=objective,
        pipeline_space=search_space,
        working_directory=working_directory,
        max_evaluations_total=max_evaluations_total,
        max_evaluations_per_run=max_evaluations_per_run,
        searcher=searcher,
        acquisition=acquisition,
        acquisition_sampler=acquisition_sampler,
        surrogate_model=surrogate_model_class,
        surrogate_model_args=surrogate_model_args,
        initial_design_size=initial_design_size,
        patience=patience,
    )

    _logger.info("Finished")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
