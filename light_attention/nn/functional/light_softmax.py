import torch
import torch.nn.functional as F


class LightSoftmax(torch.autograd.Function):
    """torch.autograd function that replaces torch.nn.functional.softmax with an implementation
    that requires less activations to compute backward. It only depends on the output of the function.
    """
    @staticmethod
    def forward(ctx, input):
        """Forward operation that is similar to torch.nn.softmax.
        Except it only stores outputs for future backward.
        (Unlike torch<=1.10 softmax that stores both inputs and outputs).

        Parameters
        ----------
        ctx :
            context to save tensors that are required for backward
        input : torch.tensor
            input tensor

        Returns
        -------
        torch.tensor
            result of a softmax operation
        """
        # using torch.functional for fast kernels
        sm_res = F.softmax(input, dim=-1)
        ctx.save_for_backward(sm_res)
        return sm_res

    @staticmethod
    def backward(ctx, grad_output):
        """Backward softmax operation, that uses only outputs
        to calculate loss gradients with respect to input.

        Parameters
        ----------
        ctx :
            context to save tensors that are required for backward
        grad_output : torch.tensor
            loss gradient with respect to the output of this layer

        Returns
        -------
        torch.tensor
            loss gradient with respect to input
        """
        sm_res, = ctx.saved_tensors
        if len(sm_res.shape) == 2:
            summ = torch.einsum('ij,ij->i', [grad_output, sm_res])
        elif len(sm_res.shape) == 3:
            summ = torch.einsum('ijk,ijk->ij', [grad_output, sm_res])
        elif len(sm_res.shape) == 4:
            summ = torch.einsum('ijkl,ijkl->ijk', [grad_output, sm_res])
        else:
            raise NotImplementedError('This shape is not currently supported in LightSoftmax backward. \
            Contact the developer.')
        # TODO: insert epsilon-defense against N-N !=0 mistakes?
        return sm_res * (grad_output - summ.unsqueeze(-1))
    
light_softmax = LightSoftmax.apply