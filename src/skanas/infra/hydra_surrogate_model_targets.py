import neps.optimizers.bayesian_optimization.kernels as _bo_kernels
import neps.optimizers.bayesian_optimization.models as _bo_models

_NB201_HIERARCHIES_CONSIDERED = {
    "nb201_fixed_1_none": [],
    "nb201_variable_multi_multi": [0, 1, 2, 3, 4],
    "act_cifar10": [0, 1, 2],
}


def instantiate_gp(args):  # pylint: disable=unused-argument
    hierarchy_considered = []

    wl_h = [2]
    graph_kernels = ["wl"]
    graph_kernels = [
        _bo_kernels.GraphKernelMapping[kernel](
            h=wl_h[j],
            oa=False,
            se_kernel=None,
        )
        for j, kernel in enumerate(graph_kernels)
    ]

    surrogate_model_args = {
        "graph_kernels": graph_kernels,
        "hp_kernels": [],
        "verbose": False,
        "hierarchy_consider": hierarchy_considered,
        "d_graph_features": 0,
        "vectorial_features": None,
    }

    surrogate_model = _bo_models.ComprehensiveGPHierarchy
    return surrogate_model, surrogate_model_args


def instantiate_gp_hierarchical(args):
    hierarchy_considered = _NB201_HIERARCHIES_CONSIDERED[args.search_space.name]

    wl_h = [2, 1] + [2] * len(hierarchy_considered)
    graph_kernels = ["wl"] * (len(hierarchy_considered) + 1)
    graph_kernels = [
        _bo_kernels.GraphKernelMapping[kernel](
            h=wl_h[j],
            oa=False,
            se_kernel=None,
        )
        for j, kernel in enumerate(graph_kernels)
    ]

    surrogate_model_args = {
        "graph_kernels": graph_kernels,
        "hp_kernels": [],
        "verbose": False,
        "hierarchy_consider": hierarchy_considered,
        "d_graph_features": 0,
        "vectorial_features": None,
    }

    surrogate_model = _bo_models.ComprehensiveGPHierarchy
    return surrogate_model, surrogate_model_args


def instantiate_gp_string_hierarchical(args):
    kernel_type = args.surrogate_model.kernel_type
    hierarchy_considered = args.surrogate_model.hierarchy_considered
    optimize_kernel_weights = args.surrogate_model.optimize_kernel_weights
    fit_num_iters = args.surrogate_model.fit_num_iterations
    combining_kernel_variant = args.surrogate_model.combining_kernel_variant

    # kernels at different hierarchy levels
    kernels = [
        _bo_models.StringKernelModelMapping[kernel_type](
            hierarchy_level=hierarchy_level,
            learnable_weights=optimize_kernel_weights,
        )
        for hierarchy_level in hierarchy_considered
    ]

    surrogate_model_args = {
        "combining_kernel_variant": combining_kernel_variant,
        "graph_kernels": kernels,
        "surrogate_model_fit_args": {
            "iters": fit_num_iters,
        },
    }

    surrogate_model = _bo_models.GPStringHierarchy
    return surrogate_model, surrogate_model_args
