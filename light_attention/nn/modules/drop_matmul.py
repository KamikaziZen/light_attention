import torch.nn as nn
from light_attention.nn.functional.drop_matmul import drop_matmul


class DropMatmul(nn.Module):

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if inplace:
            raise NotImplementedError('There is no in-place variant.')
        self.p = p

    def forward(self, lhs, rhs):
        return drop_matmul(lhs, rhs, self.p)