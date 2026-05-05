"""Single global seed setter; called once at every entry point.

Heavy imports (numpy, torch) are done lazily so importing this module stays
cheap for non-ML scripts and tests.
"""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Set seeds for python random, numpy, torch (CPU + CUDA), and PYTHONHASHSEED."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
