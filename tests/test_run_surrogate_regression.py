from __future__ import annotations

import hydra
import pytest

import skanas.scripts.run_surrogate_regression as _run_surrogate_regression_script


RUN_CONFIGS = {
    "gp_string_hierarchical__nask": (
        [
            "objective=nb201_cifar10",
            "surrogate_model=gp_string_hierarchical",
            "surrogate_model.kernel_type=nask",
            "surrogate_model.hierarchy_considered=[-2, -1]",
            "experiment_group=dev",
            "data_path='data/surrogate_regression/train_eval/nb201_cifar10'",
            "seeds=[1]",
            "n_train_values=[10]",
            "n_test=10",
        ]
    ),
    "gp_hierarchical": (
        [
            "objective=nb201_cifar10",
            "surrogate_model=gp_hierarchical",
            "experiment_group=dev",
            "data_path='data/surrogate_regression/train_eval/nb201_cifar10'",
            "seeds=1",
            "n_train_values=10",
            "n_test=10",
        ]
    ),
}


@pytest.mark.parametrize(
    "surrogate_model",
    list(RUN_CONFIGS.keys()),
)
def test_run_surrogate_regression_script(surrogate_model, use_tmp_results_dir):
    config_overrides = RUN_CONFIGS[surrogate_model]
    with hydra.initialize_config_dir(
        config_dir=str(_run_surrogate_regression_script.CONFIG_PATH),
        job_name=_run_surrogate_regression_script.CONFIG_NAME,
        version_base=_run_surrogate_regression_script.HYDRA_VERSION_BASE,
    ):
        cfg = hydra.compose(
            config_name=_run_surrogate_regression_script.CONFIG_NAME,
            overrides=config_overrides,
        )
        assert _run_surrogate_regression_script.main(cfg) is None
