from neps.search_spaces.search_space import SearchSpace

from ..benchmarks.search_spaces.activation_function_search.graph import ActivationSpace
from ..benchmarks.search_spaces.hierarchical_nb201.graph import NB201Spaces


def instantiate_search_space(args):
    space = args.search_space.space
    objective_dataset = args.objective.dataset
    if space == "act":
        base_architecture = args.search_space.base_architecture
        search_space_arch = ActivationSpace(
            base_architecture=base_architecture, dataset=objective_dataset
        )
    else:
        search_space_arch = NB201Spaces(
            space=space,
            dataset=objective_dataset,
        )
    search_space = SearchSpace(architecture=search_space_arch)
    return search_space
