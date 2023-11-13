from typing import NewType

import torch as T

Batch = NewType(
    "Batch",
    tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor]
)
