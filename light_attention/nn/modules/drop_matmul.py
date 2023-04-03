import torch
import torch.nn as nn
from light_attention.nn.functional.drop_matmul import drop_matmul


class DropMatmul(nn.Module):
    """Module for nn.functional.drop_matmul function.
    """
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if inplace:
            raise NotImplementedError('There is no in-place variant.')
        self.p = p

    def forward(self, lhs, rhs):
        return drop_matmul(lhs, rhs, self.p, self.training)
    
    def __repr__(self):
        return '{}(p={})'.format(self._get_name(), self.p)