import torch
import torch.nn as nn

class MLPHead(nn.Module):
    """
    A PyTorch module that implements a multi-layer perceptron (MLP) head for token embeddings.

    This module allows configuration of the number and size of hidden layers and their activation functions.

    Args:
        in_channels (int): Number of input channels.
        hidden_layers (list of int): Sizes of hidden layers.
        num_labels (int): Number of output labels.
        activations (callable or list of callables, optional): Activation function(s) to use after each hidden layer.
            Can be a single nn.Module (e.g., nn.ReLU()), or a list of nn.Module instances (one per hidden layer).
            Default: nn.ReLU.

    Example:
        MLPHead(in_channels=256, hidden_layers=[128, 64], num_labels=10, activations=[nn.ReLU(), nn.GELU()])
    """
    def __init__(
        self,
        in_channels: int,
        hidden_layers: list = [512, 256],
        num_labels: int = 1,
        activations=None
    ) -> None:
        super(MLPHead, self).__init__()
        layers = []
        prev_channels = in_channels

        # Default activation is nn.ReLU
        if activations is None:
            activations = nn.ReLU()
        # If a single activation is provided, repeat it for all hidden layers
        if not isinstance(activations, (list, tuple)):
            activations = [activations] * len(hidden_layers)
        assert len(activations) == len(hidden_layers), "Length of activations must match hidden_layers"

        for hidden_dim, activation in zip(hidden_layers, activations):
            layers.append(nn.Conv2d(prev_channels, hidden_dim, kernel_size=1))
            layers.append(activation)
            prev_channels = hidden_dim
        layers.append(nn.Conv2d(prev_channels, num_labels, kernel_size=1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.mlp(embeddings)
