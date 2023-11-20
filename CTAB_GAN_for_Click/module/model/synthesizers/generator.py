import torch
import torch.nn as nn
from typing import List


class Generator(nn.Module):
    """
    This class represents the generator network of the model.

    Attributes:
    1) model_layers: layers of the network used by the generator.

    Methods:
    1) __init__: initializes and builds the layers of the generator model.
    2) forward: executes a forward pass using noise as input to generate data.
    """

    def __init__(self,
                 layers: List[nn.Module]):
        super().__init__()
        self.model_layers = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model_layers(data)
