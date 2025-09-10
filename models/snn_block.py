# snn_block.py
# ------------------------------------------------------------
# Spiking blocks for EMS-YOLO-style SNN with top-level time loop
# - LCB order per stage: LIF -> Conv -> BN
# - Single global config setter: set_global_snn(...)
# - No standalone LCB class; blocks implement stages inline
# - mem_init is also globally configurable for experiments
# ------------------------------------------------------------

from typing import Dict, Optional, Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import surrogate


# ============================================================
# ============== Single global SNN shared config =============
# ============================================================

class _SharedSNN:
    """Global SNN config shared by all LIF nodes unless overridden."""
    def __init__(self,
                 thresh: float = 0.5,
                 subthresh: float = 0.5,
                 alpha: float = 1.0,
                 spiking: bool = True,
                 LIF_tau: float = 1.0,
                 mem_init: float = 0.0,
                 device: Optional[torch.device] = None):
        dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.thresh = torch.tensor(float(thresh), device=dev)
        self.subthresh = torch.tensor(float(subthresh), device=dev)
        self.alpha = float(alpha)
        self.spiking = bool(spiking)
        self.tau = torch.tensor(float(LIF_tau), device=dev)  # raw tau
        self.mem_init = float(mem_init)

    def to(self, device: torch.device):
        """Move internal tensors to device."""
        self.thresh = self.thresh.to(device)
        self.subthresh = self.subthresh.to(device)
        self.tau = self.tau.to(device)

    def decay(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Return sigmoid(tau) with correct dtype/device."""
        return torch.sigmoid(self.tau.to(device=device, dtype=dtype))

    def surrogate_apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply surrogate firing function using shared params."""
        if self.spiking:
            th = self.thresh.to(device=x.device, dtype=x.dtype)
            st = self.subthresh.to(device=x.device, dtype=x.dtype)
            return surrogate.Boxcar.apply(x, th, st)
        else:
            return x * self.alpha


_SHARED = _SharedSNN()


def set_global_snn(thresh: Optional[float] = None,
                   subthresh: Optional[float] = None,
                   alpha: Optional[float] = None,
                   spiking: Optional[bool] = None,
                   LIF_tau: Optional[float] = None,
                   mem_init: Optional[float] = None,
                   device: Optional[torch.device] = None):
    """
    Set global SNN parameters (surrogate + LIF tau + mem_init) at once.
    Call once before training/inference if you want to override defaults.
    """
    global _SHARED
    if thresh is not None:     _SHARED.thresh = torch.tensor(float(thresh), device=_SHARED.thresh.device)
    if subthresh is not None:  _SHARED.subthresh = torch.tensor(float(subthresh), device=_SHARED.subthresh.device)
    if alpha is not None:      _SHARED.alpha = float(alpha)
    if spiking is not None:    _SHARED.spiking = bool(spiking)
    if LIF_tau is not None:    _SHARED.tau = torch.tensor(float(LIF_tau), device=_SHARED.tau.device)
    if mem_init is not None:   _SHARED.mem_init = float(mem_init)
    if device is not None:     _SHARED.to(device)


# ============================================================
# ===================== LIF Node (shared) ====================
# ============================================================

class LIF_Node(nn.Module):
    """
    LIF neuron using a single global SNN config by default.
    V_t = sigmoid(tau) * V_{t-1} * (1 - S_{t-1}) + I_t
    S_t = surrogate(V_t)
    """
    def __init__(
        self,
        thresh: Optional[float] = None,
        subthresh: Optional[float] = None,
        alpha: Optional[float] = None,
        spiking: Optional[bool] = None,
        LIF_tau: Optional[float] = None,
        mem_init: Optional[float] = None,
        use_shared: bool = True,
        debug_print: bool = False,
    ):
        super().__init__()
        self.use_shared = bool(use_shared)
        self.debug_print = bool(debug_print)
        self.mem_init = mem_init  # private override if not None

        if self.use_shared:
            self._private = None
        else:
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.register_buffer("decay",
                                 torch.sigmoid(torch.tensor(1.0 if LIF_tau is None else float(LIF_tau),
                                                            device=dev)))
            self.surr = surrogate.HeavisideBoxcarCall(
                thresh=(0.5 if thresh is None else thresh),
                subthresh=(0.5 if subthresh is None else subthresh),
                alpha=(1.0 if alpha is None else alpha),
                spiking=(True if spiking is None else spiking),
            )
            self.mem_init = 0.0 if mem_init is None else mem_init

    def forward(self, LIF_U: torch.Tensor, S_before: torch.Tensor, I_in: torch.Tensor):
        if self.debug_print:
            print("[DEBUG] LIF_U:", LIF_U.shape, "S_before:", S_before.shape, "I_in:", I_in.shape)

        if self.use_shared:
            d = _SHARED.decay(dtype=LIF_U.dtype, device=LIF_U.device)
            LIF_U = d * LIF_U * (1 - S_before) + I_in
            LIF_S = _SHARED.surrogate_apply(LIF_U)
        else:
            d = self.decay.to(dtype=LIF_U.dtype, device=LIF_U.device)
            LIF_U = d * LIF_U * (1 - S_before) + I_in
            LIF_S = self.surr(LIF_U)

        return LIF_U, LIF_S


# ============================================================
# ================== Inline LCB-based blocks =================
# ============================================================

class BasicBlock_ms(nn.Module):
    """Two-stage LCB residual block implemented inline."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1,
                 e: float = 0.5, lif_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        c_ = int(out_ch * e)
        p  = 1 if k == 3 else 0

        self.lif1 = LIF_Node(**(lif_kwargs or {}))
        self.cv1  = nn.Conv2d(in_ch, c_, k, s, p, bias=False)
        self.bn1  = nn.BatchNorm2d(c_)

        self.lif2 = LIF_Node(**(lif_kwargs or {}))
        self.cv2  = nn.Conv2d(c_, out_ch, k, 1, p, bias=False)
        self.bn2  = nn.BatchNorm2d(out_ch)

        self.need_down = (s != 1) or (in_ch != out_ch)
        if self.need_down:
            self.pool = nn.MaxPool2d(s, s) if s != 1 else nn.Identity()
            self.proj = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
            self.pbn  = nn.BatchNorm2d(out_ch)

        self.state = None

    @torch.no_grad()
    def init_state(self, x: torch.Tensor) -> None:
        """Initialize mem/spk for both stages."""
        mem1 = torch.full_like(x, _SHARED.mem_init)
        spk1 = torch.zeros_like(x)

        y1 = self.bn1(self.cv1(spk1))
        mem2 = torch.full_like(y1, _SHARED.mem_init)
        spk2 = torch.zeros_like(y1)

        self.state = {"mem1": mem1, "spk1": spk1, "mem2": mem2, "spk2": spk2}

    def reset_state(self) -> None:
        self.state = None

    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        s = self.state
        u1, spk1 = self.lif1(s["mem1"], s["spk1"], x)
        s["mem1"], s["spk1"] = u1, spk1
        y1 = self.bn1(self.cv1(spk1))

        u2, spk2 = self.lif2(s["mem2"], s["spk2"], y1)
        s["mem2"], s["spk2"] = u2, spk2
        y2 = self.bn2(self.cv2(spk2))

        if self.need_down:
            sc = self.pbn(self.proj(self.pool(spk1)))
        else:
            sc = spk1

        #debug spike
        self.last_spk1 = spk1.detach()
        self.last_spk2 = spk2.detach()
        self.last_spk  = self.last_spk2

        return y2 + sc


class ConcatBlock_ms(nn.Module):
    """Two-stage LCB concat-residual block implemented inline."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1,
                 e: float = 0.5, lif_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        c_ = int(out_ch * e)
        p  = 1 if k == 3 else 0

        self.lif1 = LIF_Node(**(lif_kwargs or {}))
        self.cv1  = nn.Conv2d(in_ch, c_, k, s, p, bias=False)
        self.bn1  = nn.BatchNorm2d(c_)

        self.lif2 = LIF_Node(**(lif_kwargs or {}))
        self.cv2  = nn.Conv2d(c_, out_ch, k, 1, p, bias=False)
        self.bn2  = nn.BatchNorm2d(out_ch)

        add_ch = max(0, out_ch - in_ch)
        self.short_pad = add_ch > 0
        if self.short_pad:
            self.proj = nn.Conv2d(in_ch, add_ch, 1, 1, 0, bias=False)
            self.pbn  = nn.BatchNorm2d(add_ch)

        self.pool = nn.MaxPool2d(s, s) if s != 1 else nn.Identity()
        self.state = None

    @torch.no_grad()
    def init_state(self, x: torch.Tensor) -> None:
        mem1 = torch.full_like(x, _SHARED.mem_init)
        spk1 = torch.zeros_like(x)

        y1 = self.bn1(self.cv1(spk1))
        mem2 = torch.full_like(y1, _SHARED.mem_init)
        spk2 = torch.zeros_like(y1)

        self.state = {"mem1": mem1, "spk1": spk1, "mem2": mem2, "spk2": spk2}

    def reset_state(self) -> None:
        self.state = None

    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        s = self.state
        u1, spk1 = self.lif1(s["mem1"], s["spk1"], x)
        s["mem1"], s["spk1"] = u1, spk1
        y1 = self.bn1(self.cv1(spk1))

        u2, spk2 = self.lif2(s["mem2"], s["spk2"], y1)
        s["mem2"], s["spk2"] = u2, spk2
        y2 = self.bn2(self.cv2(spk2))

        if self.short_pad:
            temp = self.pbn(self.proj(spk1))
            cat  = torch.cat([temp, spk1], dim=1)
        else:
            cat  = spk1

        cat = self.pool(cat)


        #debug spike
        self.last_spk1 = spk1.detach()
        self.last_spk2 = spk2.detach()
        self.last_spk  = self.last_spk2
        
        return y2 + cat


# ============================================================
# =================== glue / utility modules =================
# ============================================================

class Concat(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(xs, dim=self.dim)


class Sample(nn.Module):
    def __init__(self, _, scale: int, mode: str = "nearest"):
        super().__init__()
        self.scale = scale
        self.mode = mode
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode)


class Conv_2(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 2, p: Optional[int] = None):
        super().__init__()
        if p is None: p = k // 2
        self.cv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.cv(x))
