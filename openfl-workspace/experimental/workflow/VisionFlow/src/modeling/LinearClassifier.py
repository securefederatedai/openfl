import torch


class LinearClassifier(torch.nn.Module):
    """
    A PyTorch module that implements a linear classifier for token embeddings.

    This module takes token embeddings as input, reshapes and permutes them to match
    the expected input format for a 2D convolutional layer, and applies a 1x1 convolution
    to produce the classification output.

    Attributes:
        in_channels (int): The number of input channels in the token embeddings.
        width (int): The width of the token grid.
        height (int): The height of the token grid.
        classifier (torch.nn.Conv2d): A 2D convolutional layer with a kernel size of 1x1
            that performs the classification.

    Args:
        in_channels (int): The number of input channels in the token embeddings.
        tokenW (int, optional): The width of the token grid. Default is 16.
        tokenH (int, optional): The height of the token grid. Default is 16.
        num_labels (int, optional): The number of output labels for classification. Default is 1.

    Methods:
        forward(embeddings):
            Reshapes and permutes the input embeddings, then applies the classifier
            to produce the output logits.

            Args:
                embeddings (torch.Tensor): A tensor of shape (batch_size, height * width * in_channels)
                    representing the token embeddings.

            Returns:
                torch.Tensor: A tensor of shape (batch_size, num_labels, height, width) representing
                    the classification logits.
    """
    def __init__(
        self,
        in_channels: int,
        num_labels: int = 1
    ) -> None:
        super(LinearClassifier, self).__init__()

        self.in_channels: int = in_channels
        self.classifier: torch.nn.Conv2d = torch.nn.Conv2d(in_channels, num_labels, (1, 1))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)