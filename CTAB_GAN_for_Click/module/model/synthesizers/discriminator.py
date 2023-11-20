import torch
import torch.nn as nn
from typing import List, Tuple


class Discriminator(nn.Module):
    """
    This class represents the discriminator network of the model.

    Attributes:
    1) model_layers: layers of the network used for making the final prediction of the discriminator model.
    2) info_layers: layers of the discriminator network used for computing the information loss.

    Methods:
    1) __init__: initializes and builds the layers of the discriminator model.
    2) forward: executes a forward pass on the input data to output the final predictions and corresponding
                feature information associated with the penultimate layer used to compute the information loss.
    """

    def __init__(self,
                 layers: List[nn.Module]):
        super().__init__()
        self.model_layers = nn.Sequential(*layers)
        self.info_layers = nn.Sequential(*layers[:len(layers) - 2])

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model_layers(data), self.info_layers(data)
