import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import wandb

import random
import secrets
import gc
from typing import Tuple

import snntorch as snn
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import surrogate


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

class LIF_Node(nn.Module):
    def __init__(
        self,
        LIF_tau: float = 1.0,
        subthresh: float = 0.5,
        thresh: float = 0.5,
        alpha: float = 1.0,
        spiking: bool = True
    ):
        super().__init__()
        # decay constant
        self.LIF_decay = torch.sigmoid(torch.tensor(LIF_tau))
        self.to("cuda" if torch.cuda.is_available() else "cpu")

        # create surrogate_function instance with passed-in values
        self.surrogate_function = surrogate.HeavisideBoxcarCall(
            thresh=thresh,
            subthresh=subthresh,
            alpha=alpha,
            spiking=spiking
        )

    def forward(
        self,
        LIF_U: torch.Tensor,
        S_before: torch.Tensor,
        I_in: torch.Tensor,
    ):
        print("[DEBUG] LIF_U.shape:", LIF_U.shape)
        print("[DEBUG] S_before.shape:", S_before.shape)
        print("[DEBUG] I_in.shape:", I_in.shape)
        LIF_U = self.LIF_decay * LIF_U * (1 - S_before) + I_in
        LIF_S = self.surrogate_function(LIF_U)
        return LIF_U, LIF_S

class EMS_Block_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, e=0.5, **spike_params):
        super().__init__()
        padding = kernel_size // 2
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_channels = int(out_channels * e)

        # Residual path
        self.lif1 = LIF_Node(**spike_params)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.lif2 = LIF_Node(**spike_params)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut path
        self.do_shortcut_conv = in_channels < out_channels
        if self.do_shortcut_conv:
            self.lif3 = LIF_Node(**spike_params)
            self.conv3 = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=1, stride=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels - in_channels)

        self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, x, state):
        # Residual path
        mem1, spk1 = self.lif1(state["mem1"], state["spk1"], x)
        out = self.conv1(spk1)
        out = self.bn1(out)

        mem2, spk2 = self.lif2(state["mem2"], state["spk2"], out)
        out = self.conv2(spk2)
        out_res = self.bn2(out)

        # Shortcut path
        
        if self.do_shortcut_conv:
            mem3, spk3 = self.lif3(state["mem3"], state["spk3"], x)
            sc_out = self.conv3(spk3)
            sc_out = self.bn3(sc_out)
            sc_out = self.pool(sc_out)

        else:
            temp = x
            mem3, spk3 = None, None
            sc_out = self.pool(sc_out)

        #print(f"[DEBUG] EMS_Block_2: out1.shape={out1.shape}, temp.shape={temp.shape}")
        x_pooled = self.pool(x)  # MaxPool to match residual path shape
        #print(f"[DEBUG] sc_out.shape = {sc_out.shape}, x_pooled.shape = {x_pooled.shape}")
        temp = torch.cat((sc_out, x_pooled), dim=1)
        out_final = out_res + temp

        return out_final, {
            "mem1": mem1, "spk1": spk1,
            "mem2": mem2, "spk2": spk2,
            "mem3": mem3, "spk3": spk3
        }

    def init_state(self, B, H, W, device, in_ch=None):
        H1, W1 = H //2 , W //2 
        hidden_channels = int(self.out_channels * 0.5)
        in_ch = in_ch or self.in_channels

       

        mem1 = torch.zeros(B, in_ch, H1, W1, device=device)
        spk1 = torch.zeros_like(mem1)

        

        mem2 = torch.zeros(B, in_ch, H1 // self.stride, W1 // self.stride, device=device)
        spk2 = torch.zeros_like(mem2)

        if self.do_shortcut_conv:
            mem3 = torch.zeros(B, in_ch, H1, W1, device=device)
            spk3 = torch.zeros_like(mem3)
        else:
            mem3, spk3 = None, None

        return {
            "mem1": mem1, "spk1": spk1,
            "mem2": mem2, "spk2": spk2,
            "mem3": mem3, "spk3": spk3
        }


class BasicBlock_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,  e = 0.5,  **spike_params):
        super().__init__()
        padding = kernel_size // 2
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        c_ = int(out_channels * e) 

        self.lif1 = LIF_Node(**spike_params)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)

        self.lif2 = LIF_Node(**spike_params)
        self.conv2 = nn.Conv2d(c_, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            
       
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x, state):
        print(f"[DEBUG] x.shape: {x.shape}")
        print(f"[DEBUG] state[mem1].shape: {state['mem1'].shape}")
        mem1, spk1 = self.lif1(state["mem1"], state["spk1"], x)
        out = self.conv1(spk1)
        out = self.bn1(out)

        print(f"[DEBUG] after conv1: {out.shape}")
        print(f"[DEBUG] state[mem2].shape: {state['mem2'].shape}")

        mem2, spk2 = self.lif2(state["mem2"], state["spk2"], out)
        out = self.conv2(spk2)
        out = self.bn2(out)

        shortcut_out = self.shortcut(x)

        print(f"[DEBUG] shortcut(x): {shortcut_out.shape}")
        print(f"[DEBUG] out before add: {out.shape}")

        out += shortcut_out

        return out, {"mem1": mem1, "spk1": spk1, "mem2": mem2, "spk2": spk2}

    def init_state(self, B, H, W, device, in_ch ):
        H1, W1 = H // self.stride, W // self.stride
        #in_ch = in_ch or self.in_channels
        
        #if self.kernel_size != 1 and self.stride == 1:
        H1 //= 2
        W1 //= 2


        

        mem1 = torch.zeros(B, in_ch, H1, W1, device=device)
        spk1 = torch.zeros_like(mem1)

        mem2 = torch.zeros(B, self.out_channels // 2, H1, W1, device=device)
        spk2 = torch.zeros_like(mem2)


        return {"mem1": mem1, "spk1": spk1, "mem2": mem2, "spk2": spk2}



class Upsample(nn.Module):
    def __init__(self, scale=2, mode='nearest'):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale, mode=mode)

    def forward(self, x):
        return self.upsample(x)

class Concat_s(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.out_channels = None  # ★ 여기에 추가

    def forward(self, x):
        x_out = torch.cat(x, self.d)
        if self.out_channels is None:
            self.out_channels = x_out.shape[1]  # ★ forward 중 첫 번에 채널 저장
        return x_out