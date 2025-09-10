import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Boxcar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh, subthresh):
        # spike threshold, Heaviside
        # store membrane potential before reset
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        ctx.subthresh = subthresh
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        # surrogate-gradient, BoxCar
        # stored membrane potential before reset
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - ctx.thresh) < ctx.subthresh
        # return grad_input, None, None
        # return grad_input * temp.float(), None, None
        # return grad_input * temp.float() / (ctx.subthresh*2), None, None
        return grad_input * temp.float(), None, None


class HeavisideBoxcarCall(nn.Module):
    def __init__(self, thresh=0.5, subthresh=0.5, alpha=1.0, spiking=True):
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking
        self.thresh = torch.tensor(thresh)
        self.subthresh = torch.tensor(subthresh)
        self.thresh.to("cuda" if torch.cuda.is_available() else "cpu")
        self.subthresh.to("cuda" if torch.cuda.is_available() else "cpu")
        if spiking:
            self.f = Boxcar.apply
        else:
            self.f = self.primitive_function

    def forward(self, x):
        return self.f(x, self.thresh, self.subthresh)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return x * alpha

class cTRACEClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eU, clamp_value):
        ctx.set_materialize_grads(False)  
        return torch.clamp(eU, max=clamp_value)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

class cTRACE2eSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eU, ethreshold):
        ctx.set_materialize_grads(False)
        eS = eU.gt(ethreshold).float()
        return eS.float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class cTRACESwitch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_spk_mul_w, input_spk_b_mul_w):
        ctx.set_materialize_grads(False)
        return input_spk_mul_w.float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return None, grad_input


class AlwaysOnbeyondAcc(torch.autograd.Function):
    # Define approximate firing function
    # 항상 모든 step에서 output에서는 grad가 만들어지게 유도, acc_nueron에 들어오는 spk가 0이여도 작동
    # always update - 상관없을 듯, 어자피 spike 나오면, Vmem이 0이므로 update 안됨.
    @staticmethod
    def forward(ctx, input_spk):
        ctx.set_materialize_grads(False)
        return input_spk

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
