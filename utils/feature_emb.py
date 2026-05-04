"""Embedding layers for small continuous feature grids."""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

from collections import OrderedDict


class LinearEmbeddings(nn.Module):
    """Linear embeddings for 2-dimensional continuous features.

    **Shape**

    - Input: `(*, num_features1, num_features2)`
    - Output: `(*, num_features1, num_features2, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> num_features1 = 3
    >>> num_features2 = 4
    >>> x = torch.randn(batch_size, num_features1, num_features2)
    >>> d_embedding = 5
    >>> m = LinearEmbeddings((num_features1, num_features2), d_embedding)
    >>> m(x).shape
    torch.Size([2, 3, 4, 5])
    """

    def __init__(self, num_features1, num_features2, d_embedding: int) -> None:
        if num_features1 <= 0 or num_features2 <= 0:
            raise ValueError(
                "Each feature dimension must be positive, "
                f"got num_features1={num_features1}, num_features2={num_features2}"
            )
        if d_embedding <= 0:
            raise ValueError(f"d_embedding must be positive, got {d_embedding=}")

        super().__init__()
        self.weight = Parameter(torch.empty(num_features1, num_features2, d_embedding))
        self.bias = Parameter(torch.empty(num_features1, num_features2, d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.weight.shape[-1] ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """Broadcast the learned affine transform over the last axis."""
        return torch.addcmul(self.bias, self.weight, x[..., None])


class LinearReLUEmbeddings(nn.Sequential):
    """Simple non-linear embeddings for continuous features.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>>
    >>> d_embedding = 32
    >>> m = LinearReLUEmbeddings(n_cont_features, d_embedding)
    >>> m(x).shape
    torch.Size([2, 3, 32])
    """

    def __init__(self, n_features1, n_features2: int, d_embedding: int = 32) -> None:
        super().__init__(
            OrderedDict(
                [
                    (
                        "linear",
                        LinearEmbeddings(n_features1, n_features2, d_embedding),
                    ),
                    ("activation", nn.ReLU()),
                ]
            )
        )
