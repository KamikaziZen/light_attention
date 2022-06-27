import torch.nn as nn
from light_attention.nn.functional.light_softmax import light_softmax


class LightSoftmax(nn.Module):
    """Module for nn.functional.light_softmax function.
    """
    def forward(self, input):
        return light_softmax(input)