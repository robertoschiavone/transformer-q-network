from typing import NewType

import numpy as np

Transition = NewType(
    "Transition",
    tuple[np.ndarray, int, np.ndarray, float, float]
)
