import torch
import torch.nn.functional as F


class LightSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # using torch.functional for fast kernels
        sm_res = F.softmax(input, dim=-1)
        ctx.save_for_backward(sm_res)
        return sm_res

    @staticmethod
    def backward(ctx, grad_output):
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