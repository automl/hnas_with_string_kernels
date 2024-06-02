from __future__ import annotations

import hydra
import pytest

import skanas.scripts.run_search as _run_search_script


RUN_CONFIGS = {
    "act__gp_string_hierarchical__nask": (
        [
            "search_space=act_cifar10",
            "objective=act_cifar10",
            "surrogate_model=gp_string_hierarchical",
            "surrogate_model.kernel_type=nask",
            "surrogate_model.hierarchy_considered=[-2, -1]",
            "experiment_group=dev",
            "data_path='data/search/nb201_cifar10'",
            "seed=777",
            "n_init=1",
            "max_evaluations_total=2",
        ]
    ),
    "nb201__gp_string_hierarchical__nask": (
        [
            "objective=nb201_cifar10",
            "surrogate_model=gp_string_hierarchical",
            "surrogate_model.kernel_type=nask",
            "surrogate_model.hierarchy_considered=[-2, -1]",
            "experiment_group=dev",
            "data_path='data/search/nb201_cifar10'",
            "seed=777",
            "n_init=1",
            "max_evaluations_total=2",
        ]
    ),
    "nb201__gp_hierarchical": (
        [
            "objective=nb201_cifar10",
            "surrogate_model=gp_hierarchical",
            "experiment_group=dev",
            "data_path='data/search/nb201_cifar10'",
            "seed=777",
            "n_init=1",
            "max_evaluations_total=2",
        ]
    ),
}


@pytest.mark.skip(reason="Slow GPU required test")
@pytest.mark.slow
@pytest.mark.parametrize(
    "surrogate_model",
    list(RUN_CONFIGS.keys()),
)
def test_run_search_script(surrogate_model, use_tmp_results_dir):
    config_overrides = RUN_CONFIGS[surrogate_model]
    with hydra.initialize_config_dir(
        config_dir=str(_run_search_script.CONFIG_PATH),
        job_name=_run_search_script.CONFIG_NAME,
        version_base=_run_search_script.HYDRA_VERSION_BASE,
    ):
        cfg = hydra.compose(
            config_name=_run_search_script.CONFIG_NAME,
            overrides=config_overrides,
        )
        assert _run_search_script.main(cfg) is None
