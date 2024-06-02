import pathlib
import random

import gitinfo
import omegaconf


def _get_hydra_args_for_pprint(args):
    args_yaml = omegaconf.OmegaConf.to_yaml(args)
    args_yaml = "\n".join(f"    {line}" for line in args_yaml.splitlines())
    return args_yaml


def log_runtime_info(logger, project_root, hydra_args):
    logger.info("Project root: %s", project_root)
    logger.info("Working directory %s", pathlib.Path.cwd())

    hydra_args_yaml = _get_hydra_args_for_pprint(hydra_args)
    logger.info("Args:\n%s", hydra_args_yaml)

    try:
        git_info = gitinfo.gitinfo.get_git_info(str(project_root))
        logger.info("Git information: Commit hash=%s", git_info['commit'])
        logger.info("Git information: Commit date=%s", git_info['author_date'])
    except TypeError:
        logger.info("No git information is available")


def get_absolute_data_path(data_path: str, project_root) -> str:
    data_path = pathlib.Path(project_root / data_path).absolute()
    if not data_path.is_dir():
        raise ValueError(f"Given data_path={data_path} is not a directory")
    return str(data_path)


def set_seeds(seed: int, for_numpy=True, for_torch=True, for_tf=False):
    seed = int(seed)
    random.seed(seed)  # important for NePS optimizers

    if for_numpy:
        import numpy as np

        np.random.seed(seed)  # important for NePS optimizers

    if for_torch:
        import torch

        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if for_tf:
        import tensorflow as tf

        tf.random.set_seed(seed)
