"""Reproducibility utils for Deep learning."""

from __future__ import annotations

from os import environ
from random import seed as rnd_seed

from numpy import random
from torch import manual_seed
from torch.backends import cudnn
from torch.cuda import manual_seed as cuda_manual_seed
from torch.cuda import manual_seed_all


def set_global_seed(seed: int) -> None:
    """Set seed to enable reproducible result.

    Args:
    ----
        seed (int): number of randomness block.

    """
    rnd_seed(a=seed)
    environ["PYTHONHASHSEED"] = str(object=seed)
    random.seed(seed=seed)  # noqa: NPY002
    manual_seed(seed=seed)
    cuda_manual_seed(seed=seed)
    manual_seed_all(seed=seed)

    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
