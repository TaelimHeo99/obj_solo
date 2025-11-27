# ------------------------------------------------------------
# Spiking YOLOv3-tiny (SOLO_YOLOTiny)
# - Mirrors a tiny YOLO topology with spiking blocks (LCB order)
# - Temporal loop outside; each _ms block holds internal LIF states
# - SOLO-style loop: all early timesteps are no-grad; last timestep is grad-enabled
# - Two detection scales: P4 (1/16), P5 (1/32)
# - Ultralytics compatibility: self.model[-1] is Detect
# - Batch-wise state reset available with safe batch/shape handling
# - IMPORTANT: Adds YOLOv5-style Detect bias initialization here
#   (no need to modify models/yolo.py)
# ------------------------------------------------------------

from typing import List, Optional, Sequence
import math  # for bias initialization

import torch
import torch.nn as nn
import tqdm  # use tqdm.tqdm.write for clean logs with progress bars

# Spiking blocks and utilities
from models.snn_block import (
    BasicBlock_ms,
    ConcatBlock_ms,
    Concat,
    Sample,
    Conv_2,
)

# Ultralytics-style Detect head
from models.yolo import Detect
from utils.autoanchor import check_anchor_order


class SOLO_YOLOTiny(nn.Module):
    """
    Spiking YOLOv3-tiny reimplementation (LCB per stage, time-loop outside).

    Topology:
        Backbone:
            0: Conv_2(3, 32, k=3, s=2)
            1: ConcatBlock_ms(32, 64, k=3, s=2)
            2: BasicBlock_ms(64, 64, k=3, s=1)
            3: ConcatBlock_ms(64, 128, k=3, s=2)
            4: BasicBlock_ms(128, 128, k=3, s=1)
            5: ConcatBlock_ms(128, 256, k=3, s=2)   # P4/16
            6: BasicBlock_ms(256, 256, k=3, s=1)
            7: ConcatBlock_ms(256, 512, k=3, s=2)   # P5/32
            8: BasicBlock_ms(512, 512, k=3, s=1)

        Head:
            9 : BasicBlock_ms(512, 256, k=3, s=1)
            10: ConcatBlock_ms(256, 512, k=3, s=1)  # P5/32 (large)
            11: BasicBlock_ms(256, 128, k=1, s=1)   # from layer 9
            12: Sample(scale=2, mode='nearest')
            13: Concat([up, layer6], dim=1)
            14: BasicBlock_ms(384, 256, k=3, s=1)   # P4/16 (medium)

        Detect inputs: [P4(256), P5(512)]
    """

    def __init__(
        self,
        nc: int = 80,
        anchors: List[List[int]] = (
            (10, 14, 23, 27, 37, 58),
            (81, 82, 135, 169, 344, 319),
        ),
        num_steps: int = 5,
        log_spike: bool = False,
        log_every: int = 1,
        layers_to_log: Optional[Sequence[str]] = None,  # e.g., ["m1","m5","m10","m14"]
        reset_on_batch: bool = True,  # reset SNN states at the start of every batch (train)
        log_reset: bool = False,      # log when states are reset (for verification)
    ):
        super().__init__()

        self.nc = nc
        self.num_steps = num_steps

        # ---- logging controls ----
        self.log_spike = bool(log_spike)
        self.log_every = max(int(log_every), 1)
        self.layers_to_log = set(layers_to_log) if layers_to_log is not None else {
            "m1",
            "m2",
            "m3",
            "m5",
            "m7",
            "m9",
            "m10",
            "m11",
            "m14",
        }
        self.log_reset = bool(log_reset)

        # ---- state reset policy ----
        self.reset_on_batch = bool(reset_on_batch)

        # ---------------- Backbone (0..8) ----------------
        self.m0 = Conv_2(3, 32, k=3, s=2)              # 0
        self.m1 = ConcatBlock_ms(32, 64, k=3, s=2)     # 1
        self.m2 = BasicBlock_ms(64, 64, k=3, s=1)      # 2
        self.m3 = ConcatBlock_ms(64, 128, k=3, s=2)    # 3
        self.m4 = BasicBlock_ms(128, 128, k=3, s=1)    # 4
        self.m5 = ConcatBlock_ms(128, 256, k=3, s=2)   # 5 (P4/16)
        self.m6 = BasicBlock_ms(256, 256, k=3, s=1)    # 6
        self.m7 = ConcatBlock_ms(256, 512, k=3, s=2)   # 7 (P5/32)
        self.m8 = BasicBlock_ms(512, 512, k=3, s=1)    # 8

        # ---------------- Head (9..14) ----------------
        self.m9 = BasicBlock_ms(512, 256, k=3, s=1)    # 9
        self.m10 = ConcatBlock_ms(256, 512, k=3, s=1)  # 10 (P5/32)
        self.m11 = BasicBlock_ms(256, 128, k=1, s=1)   # 11 (from layer 9)
        self.up = Sample(None, 2, "nearest")           # 12
        self.cat = Concat(dim=1)                       # 13
        self.m14 = BasicBlock_ms(384, 256, k=3, s=1)   # 14 (P4/16)

        # ---------------- Detect ----------------
        self.detect = Detect(nc=self.nc, anchors=anchors, ch=[256, 512], inplace=True)
        self.detect.stride = torch.tensor([16.0, 32.0])  # P4, P5
        self.stride = self.detect.stride  # compatibility with some utils
        check_anchor_order(self.detect)

        # Normalize anchors and initialize biases (done here so yolo.py is untouched)
        with torch.no_grad():
            # Normalize anchors by stride (standard YOLO practice)
            self.detect.anchors /= self.detect.stride.view(-1, 1, 1)

            # ---- YOLOv5-style Detect bias initialization ----
            # This accelerates early training by providing useful priors for obj and cls.
            # (Assume ~8 objects per 640x640 image; uniform class prior if cf is unknown.)
            for mi, s in zip(self.detect.m, self.detect.stride):
                b = mi.bias.view(self.detect.na, -1)  # (na, 5+nc)
                # objectness bias
                b[:, 4] += math.log(8 / (640.0 / float(s)) ** 2)
                # class bias
                b[:, 5 : 5 + self.nc] += math.log(0.6 / (self.nc - 0.99999))
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        # For Ultralytics trainer compatibility (expects model[-1] is Detect)
        self.model = nn.ModuleList([self.detect])

        # ---- lazy-state flags ----
        self._states_inited = False
        self.hw = None        # (H, W) cache to detect resolution changes
        self._state_B = None  # batch size used for current states

        # ---- spike statistics for firing rate ----
        # These are per-forward (per-batch) stats, used by the trainer.
        self.last_batch_spikes = 0.0
        self.last_batch_neuron_steps = 0.0

    # ---------------- helpers: states ----------------
    @torch.no_grad()
    def _lazy_init_states(self, x: torch.Tensor):
        """
        Run a minimal pass to allocate internal LIF states based on shapes.
        Each _ms block exposes init_state(tensor) and forward_step(tensor).
        NOTE: This warms states via one forward_step per block.
        """
        # Backbone init
        y0 = self.m0(x)
        self.m1.init_state(y0)
        y1 = self.m1.forward_step(y0)

        self.m2.init_state(y1)
        y2 = self.m2.forward_step(y1)

        self.m3.init_state(y2)
        y3 = self.m3.forward_step(y2)

        self.m4.init_state(y3)
        y4 = self.m4.forward_step(y3)

        self.m5.init_state(y4)
        y5 = self.m5.forward_step(y4)  # P4

        self.m6.init_state(y5)
        y6 = self.m6.forward_step(y5)

        self.m7.init_state(y6)
        y7 = self.m7.forward_step(y6)

        self.m8.init_state(y7)
        y8 = self.m8.forward_step(y7)

        # Head init
        self.m9.init_state(y8)
        y9 = self.m9.forward_step(y8)

        self.m10.init_state(y9)
        _ = self.m10.forward_step(y9)

        # m11 takes y9 (not y10)
        self.m11.init_state(y9)
        y11 = self.m11.forward_step(y9)

        y12 = self.up(y11)
        y13 = self.cat([y12, y6])

        self.m14.init_state(y13)
        _ = self.m14.forward_step(y13)

        # Mark states ready and record current resolution and batch size
        self._states_inited = True
        self.hw = (int(x.shape[2]), int(x.shape[3]))
        self._state_B = int(x.shape[0])

    @torch.no_grad()
    def reset_states(self):
        """
        Clear LIF states between sequences/batches.

        This does NOT allocate new states; it just clears internal buffers.
        Actual shapes are re-established by _lazy_init_states(x).
        """
        for m in [
            self.m1,
            self.m2,
            self.m3,
            self.m4,
            self.m5,
            self.m6,
            self.m7,
            self.m8,
            self.m9,
            self.m10,
            self.m11,
            self.m14,
        ]:
            m.reset_state()
        self._states_inited = False
        self._state_B = None
        if self.log_reset:
            tqdm.tqdm.write("[SNN] states reset")

    # ---------------- helpers: firing-rate stats ----------------
    def _accumulate_spike_stats(self, module: nn.Module, B: int):
        """
        Accumulate spike statistics for firing-rate computation.

        Assumes spiking blocks store their last spikes in module.last_spk
        with shape [B, ...].
        """
        if not hasattr(module, "last_spk") or module.last_spk is None:
            return
        with torch.no_grad():
            spk = module.last_spk
            self.last_batch_spikes += float(spk.float().sum().item())
            # (#neurons per sample) * batch_size * 1 timestep
            self.last_batch_neuron_steps += float(spk[0].numel() * B)

    # ---------------- helpers: freezing ----------------
    def freeze_backbone(self):
        """
        Freeze SNN backbone (m0..m8) and keep head + Detect trainable.
        Used when opt.snn_freeze == 'backbone'.
        """
        backbone_modules = [
            self.m0,
            self.m1,
            self.m2,
            self.m3,
            self.m4,
            self.m5,
            self.m6,
            self.m7,
            self.m8,
        ]
        for m in backbone_modules:
            for p in m.parameters():
                p.requires_grad = False

    def freeze_all_but_detect(self):
        """
        Freeze all modules except the Detect head.
        Used when opt.snn_freeze == 'detect_only'.
        """
        for name, p in self.named_parameters():
            # Detect parameters are under 'detect.' or 'model.0.'
            if name.startswith("detect.") or name.startswith("model.0."):
                p.requires_grad = True
            else:
                p.requires_grad = False

    # ---------------- logging helper ----------------
    def _log_spike(self, name: str, module: nn.Module, t: int):
        """
        Log mean spike rate of a given module at timestep t.
        Uses tqdm.tqdm.write so it doesn't break the progress bar.
        """
        if not self.log_spike:
            return
        if (t % self.log_every) != 0:
            return
        if name not in self.layers_to_log:
            return
        if hasattr(module, "last_spk") and module.last_spk is not None:
            with torch.no_grad():
                rate = float(module.last_spk.float().mean().item())
            tqdm.tqdm.write(f"[spike] t={t:02d} {name}: {rate:.4f}")

    # ---------------- forward ----------------
    def forward(self, x: torch.Tensor):
        """
        SOLO-style temporal accumulation:
        - For t < num_steps-1: update spike/membrane states under no-grad.
        - For t == num_steps-1: run with gradient enabled.

        Input:
            x: [B, 3, H, W]
        Output:
            Detect head predictions from averaged P4/P5 features.
        """
        B, _, H, W = x.shape
        dev = x.device

        # ------------------------------------------------
        # Safe state management:
        # - If reset_on_batch and training  -> always reset + re-init
        # - If batch size changes           -> reset + re-init
        # - If resolution (H,W) changes     -> reset + re-init
        # - If states not initialized yet   -> init
        # This avoids mismatches like mem(B=16) vs input(B=8).
        # ------------------------------------------------
        need_reinit = False

        if not self._states_inited:
            # First call: states not yet allocated
            need_reinit = True
        elif self.hw != (H, W):
            # Image resolution changed
            need_reinit = True
        elif self._state_B is None or self._state_B != B:
            # Batch size changed (e.g., train batch=16 -> val batch=8)
            need_reinit = True
        elif self.reset_on_batch and self.training:
            # Policy: always reset states at the start of each training batch
            need_reinit = True

        if need_reinit:
            self.reset_states()
            self._lazy_init_states(x)

        # Ensure Detect buffers are on the right device/dtype
        if isinstance(self.detect.stride, torch.Tensor):
            self.detect.stride = self.detect.stride.to(device=dev, dtype=torch.float32)
            self.detect.anchors = self.detect.anchors.to(device=dev, dtype=torch.float32)

        # Temporal accumulators (use FP32 for numerical stability under AMP)
        P4 = torch.zeros(B, 256, H // 16, W // 16, device=dev, dtype=torch.float32)
        P5 = torch.zeros(B, 512, H // 32, W // 32, device=dev, dtype=torch.float32)

        # Reset batch spike statistics
        self.last_batch_spikes = 0.0
        self.last_batch_neuron_steps = 0.0

        # ---------------- SOLO temporal loop ----------------
        for t in range(self.num_steps):
            is_last = (t == self.num_steps - 1)

            # Early timesteps: no-grad (state updates only), last timestep: grad on
            if not is_last:
                with torch.no_grad():
                    # Backbone
                    y0 = self.m0(x)  # non-spiking stem (stateless)

                    y1 = self.m1.forward_step(y0)
                    self._accumulate_spike_stats(self.m1, B)
                    self._log_spike("m1", self.m1, t)

                    y2 = self.m2.forward_step(y1)
                    self._accumulate_spike_stats(self.m2, B)
                    self._log_spike("m2", self.m2, t)

                    y3 = self.m3.forward_step(y2)
                    self._accumulate_spike_stats(self.m3, B)
                    self._log_spike("m3", self.m3, t)

                    y4 = self.m4.forward_step(y3)
                    self._accumulate_spike_stats(self.m4, B)

                    y5 = self.m5.forward_step(y4)  # P4 / 16
                    self._accumulate_spike_stats(self.m5, B)
                    self._log_spike("m5", self.m5, t)

                    y6 = self.m6.forward_step(y5)
                    self._accumulate_spike_stats(self.m6, B)

                    y7 = self.m7.forward_step(y6)
                    self._accumulate_spike_stats(self.m7, B)
                    self._log_spike("m7", self.m7, t)

                    y8 = self.m8.forward_step(y7)  # / 32
                    self._accumulate_spike_stats(self.m8, B)

                    # Head
                    y9 = self.m9.forward_step(y8)
                    self._accumulate_spike_stats(self.m9, B)

                    y10 = self.m10.forward_step(y9)  # P5 / 32
                    self._accumulate_spike_stats(self.m10, B)
                    self._log_spike("m10", self.m10, t)

                    y11 = self.m11.forward_step(y9)  # from layer 9
                    self._accumulate_spike_stats(self.m11, B)
                    self._log_spike("m11", self.m11, t)

                    y12 = self.up(y11)
                    y13 = self.cat([y12, y6])        # concat with backbone layer-6
                    y14 = self.m14.forward_step(y13) # P4 / 16
                    self._accumulate_spike_stats(self.m14, B)
                    self._log_spike("m14", self.m14, t)

                    # Accumulate without building autograd graph
                    P4 += y14.float()
                    P5 += y10.float()
            else:
                # Last timestep: gradient-enabled pass
                # Backbone
                y0 = self.m0(x)  # non-spiking stem

                y1 = self.m1.forward_step(y0)
                self._accumulate_spike_stats(self.m1, B)
                self._log_spike("m1", self.m1, t)

                y2 = self.m2.forward_step(y1)
                self._accumulate_spike_stats(self.m2, B)
                self._log_spike("m2", self.m2, t)

                y3 = self.m3.forward_step(y2)
                self._accumulate_spike_stats(self.m3, B)
                self._log_spike("m3", self.m3, t)

                y4 = self.m4.forward_step(y3)
                self._accumulate_spike_stats(self.m4, B)

                y5 = self.m5.forward_step(y4)  # P4 / 16
                self._accumulate_spike_stats(self.m5, B)
                self._log_spike("m5", self.m5, t)

                y6 = self.m6.forward_step(y5)
                self._accumulate_spike_stats(self.m6, B)

                y7 = self.m7.forward_step(y6)
                self._accumulate_spike_stats(self.m7, B)
                self._log_spike("m7", self.m7, t)

                y8 = self.m8.forward_step(y7)  # / 32
                self._accumulate_spike_stats(self.m8, B)

                # Head
                y9 = self.m9.forward_step(y8)
                self._accumulate_spike_stats(self.m9, B)

                y10 = self.m10.forward_step(y9)  # P5 / 32
                self._accumulate_spike_stats(self.m10, B)
                self._log_spike("m10", self.m10, t)

                y11 = self.m11.forward_step(y9)  # from layer 9
                self._accumulate_spike_stats(self.m11, B)
                self._log_spike("m11", self.m11, t)

                y12 = self.up(y11)
                y13 = self.cat([y12, y6])         # concat with backbone layer-6
                y14 = self.m14.forward_step(y13)  # P4 / 16
                self._accumulate_spike_stats(self.m14, B)
                self._log_spike("m14", self.m14, t)

                # Accumulate with gradient attached for the last-step terms
                P4 += y14.float()
                P5 += y10.float()

        # Average over time (last step carries grad; earlier steps were no-grad)
        P4 /= self.num_steps
        P5 /= self.num_steps

        # Detect once
        return self.detect([P4, P5])
