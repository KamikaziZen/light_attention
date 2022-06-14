import torch
import torch.nn.functional as F


class DropMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat1, mat2, pdrop):
        # using torch.functional for fast kernels
        if pdrop > 0:
            mat1_masked = F.dropout(mat1, pdrop)
            mask = torch.ne(mat1_masked, 0)
            ctx.save_for_backward(mat1, mat2, mask, torch.tensor(pdrop))
            return torch.matmul(mat1_masked, mat2)
        else:
            ctx.save_for_backward(mat1, mat2)
            return torch.matmul(mat1, mat2)

    @staticmethod
    def backward(ctx, grad_output):
        mat1, mat2, *other = ctx.saved_tensors
        if len(other) == 0:
            return torch.matmul(grad_output, torch.transpose(mat2, -1, -2)), \
                   torch.matmul(torch.transpose(mat1, -1, -2), grad_output), None
        elif len(other) == 2:
            mask, pdrop = other
            return torch.matmul(grad_output, torch.transpose(mat2, -1, -2)) * mask / (1 - pdrop), \
                   torch.matmul(torch.transpose(mat1 * mask / (1 - pdrop), -1, -2), grad_output), None
        else:
            raise ValueError('Incorrect number of saved tensors.')

drop_matmul = DropMatmul.apply
