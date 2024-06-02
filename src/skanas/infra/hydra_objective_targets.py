import functools

from ..benchmarks.objectives.addNIST import AddNISTObjective
from ..benchmarks.objectives.cifar_activation import CIFAR10ActivationObjective
from ..benchmarks.objectives.cifarTile import CifarTileObjective
from ..benchmarks.objectives.hierarchical_nb201 import NB201Pipeline


def instantiate_objective(args):
    objective_dataset = args.objective.dataset

    ObjectiveMapping = {
        "act_cifar10": functools.partial(CIFAR10ActivationObjective, dataset="cifar10"),
        "nb201_addNIST": AddNISTObjective,
        "nb201_cifarTile": CifarTileObjective,
        "nb201_cifar10": functools.partial(NB201Pipeline, dataset=objective_dataset),
        "nb201_cifar100": functools.partial(NB201Pipeline, dataset=objective_dataset),
        "nb201_ImageNet16-120": functools.partial(
            NB201Pipeline, dataset=objective_dataset
        ),
    }

    run_pipeline_fn = ObjectiveMapping[args.objective.name](
        data_path=args.data_path,
        seed=args.seed,
        log_scale=args.y_log,
    )

    return run_pipeline_fn
