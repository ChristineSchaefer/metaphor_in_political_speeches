import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    """
        The Highway class extends the PyTorch nn.Module class.
        This class implements the highway layer. The highway layer is a
        fully connected layer with a learnable gating mechanism.

        from https://github.com/omidrohanian/metaphor_mwe/blob/master/layers/highway.py
    """

    def __init__(self, input_size, bias=-2.0):
        """
            Initializes a new Highway instance.
        """
        super(Highway, self).__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.transform = nn.Linear(input_size, input_size)
        self.transform.bias.data.fill_(bias)

    def forward(self, input):
        """
            Defines the forward pass for the highway layer.

            @param input: The input tensor

            @returns output of the highway layer which is the gated combination
                    of the projected input and the original input
        """
        proj_result = nn.functional.relu(self.proj(input))
        proj_gate = F.sigmoid(self.transform(input))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * input)
        return gated
