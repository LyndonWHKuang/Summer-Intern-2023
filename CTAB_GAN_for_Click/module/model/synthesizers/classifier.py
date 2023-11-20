import torch
import torch.nn as nn
from typing import List


class Classifier(nn.Module):
    """
    This class represents the classifier module used alongside the discriminator to train the generator network.

    Attributes:
    1) dim: column dimensionality of the transformed input data after removing target column.
    2) class_dims: list of dimensions used for the hidden layers of the classifier network.
    3) target_column_pos: tuple containing the starting and ending positions of the target column in the transformed
    input data.

    Methods:
    1) __init__: initializes and builds the layers of the classifier module.
    2) forward: executes the forward pass of the classifier module on the corresponding input data and
                outputs the predictions and corresponding true labels for the target column.
    """

    def __init__(self,
                 input_dim: int,
                 class_dims: List[int],
                 target_column_pos):
        super().__init__()

        self.dim = input_dim - (target_column_pos[1] - target_column_pos[0])
        self.target_column_pos = target_column_pos
        self.network_layers = self.build_network_layers(class_dims)

    def build_network_layers(self, class_dims: List[int]) -> nn.Sequential:
        layers = []
        input_dim = self.dim
        for item in class_dims:
            layers += [
                nn.Linear(input_dim, item),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5)
            ]
            input_dim = item

        if (self.target_column_pos[1] - self.target_column_pos[0]) == 2:
            layers += [nn.Linear(input_dim, 1), nn.Sigmoid()]
        else:
            layers += [nn.Linear(input_dim, (self.target_column_pos[1] - self.target_column_pos[0]))]

        return nn.Sequential(*layers)

    def forward(self, data: torch.Tensor):
        label = torch.argmax(data[:, self.target_column_pos[0]:self.target_column_pos[1]], dim=-1)
        input_data = torch.cat((data[:, :self.target_column_pos[0]], data[:, self.target_column_pos[1]:]), 1)

        if (self.target_column_pos[1] - self.target_column_pos[0]) == 2:
            return self.network_layers(input_data).view(-1), label
        else:
            return self.network_layers(input_data), label
