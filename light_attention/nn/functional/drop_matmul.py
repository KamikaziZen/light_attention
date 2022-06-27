import torch
import torch.nn.functional as F


class DropMatmul(torch.autograd.Function):
    """torch.autograd function that replaces consecutive dropout and matmul operations
    with a single merged operation. This allows storing less activations.
    """
    @staticmethod
    def forward(ctx, mat1, mat2, pdrop):
        """Merged forward operation for consecutive dropout and matmul.
        It applies dropout to mat1 and multiplies(matmul) the result by mat2.

        Parameters
        ----------
        ctx :
            context to save tensors that are required for backward
        mat1 : torch.tensor
            input matrix 1
        mat2 : torch.tensor
            input matrix 2
        pdrop : float
            probability for dropout operation

        Returns
        -------
        torch.tensor
            matmul(dropout(mat1), mat2)
        """
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
        """Merged backward operation for consecutive dropout and matmul.
        Returns loss gradients with respect to both inputs.

        Parameters
        ----------
        ctx :
            context to save tensors that are required for backward
        grad_output : torch.tensor
            loss gradient with respect to the output of this layer

        Returns
        -------
        torch.tensor
            loss gradient with respect to mat1
        torch.tensor
            loss gradient with respect to mat2
        """
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
