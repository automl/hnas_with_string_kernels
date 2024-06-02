from __future__ import annotations

import pytest
import torch

import neps.optimizers.bayesian_optimization.kernels as _bo_kernels
import neps.optimizers.bayesian_optimization.kernels.string_hierarchy.config_string as _config_string


STRING_KERNEL_MAPPING = _bo_kernels.StringKernelMapping
TOLERANCE = 1E-6


def tensor_is_symmetric(tensor: torch.Tensor):
    return bool((torch.abs(tensor - tensor.T) <= TOLERANCE).all())


def tensor_is_positive_semi_definite(tensor: torch.Tensor):
    return (
        tensor_is_symmetric(tensor)
        and bool((torch.linalg.eigvalsh(tensor) >= 0).all())
    )


@pytest.mark.parametrize(
    "config_strings",
    ("simple_config_strings", "complex_config_strings"),
)
@pytest.mark.parametrize(
    "kernel_type",
    list(STRING_KERNEL_MAPPING.keys()),
)
def test_forward_values(kernel_type, config_strings, request):
    config_strings = request.getfixturevalue(config_strings)

    n_configs = len(config_strings)
    configs = tuple(
        _config_string.ConfigString(config_string=c) for c in config_strings
    )

    kernel_type = STRING_KERNEL_MAPPING[kernel_type]
    kernel = kernel_type()
    K = kernel.forward(configs=configs)

    assert n_configs > 0
    assert K.size() == (n_configs, n_configs)
    assert tensor_is_symmetric(K)
    assert tensor_is_positive_semi_definite(K)
