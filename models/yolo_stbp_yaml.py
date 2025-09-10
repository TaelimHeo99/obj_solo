import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

from models.yolo import Detect
from models.snn_block import (
    ConcatBlock_ms, BasicBlock_ms, Sample, Concat, Conv_2
)

class STBP_YAML(nn.Module):
    """
    YAML-defined Spiking YOLOv3-tiny (EMS-YOLO style)
    - Uses _ms blocks (ConcatBlock_ms / BasicBlock_ms) which maintain their own internal membrane/spike states
    - init_states() sets up states for each ms block using dummy inputs
    - forward() runs multiple timesteps, only applying the Detect layer once at the end
    """

    def __init__(self, cfg, ch=3, nc=80, num_steps=5, log_spike=True):
        super().__init__()

        # Load YAML config if given as a path
        if isinstance(cfg, str) and os.path.isfile(cfg):
            with open(cfg, "r") as f:
                cfg = yaml.safe_load(f)

        self.yaml = cfg
        self.num_steps = num_steps
        self.nc = nc
        self.log_spike = log_spike

        self.model_defs = cfg["backbone"] + cfg["head"]
        self.model, self.save, self.out_channels = self.parse_model(cfg, ch=ch)
        self.stride = self.model[-1].stride  # From Detect layer (e.g., tensor([16., 32.]))

    # --------------------- Model Parsing ---------------------
    def parse_model(self, cfg, ch):
        """
        Parse the YAML-defined backbone and head.
        Creates the model layers and determines output channels for each layer.
        """
        layers, save, out_channels = [], [], []
        nc = cfg.get("nc", 80)
        anchors = cfg.get("anchors", [])

        # Track output channels for each layer (index 0 = input channels)
        ch_track = [ch]

        for i, (f, n, m, args) in enumerate(self.model_defs):
            m = eval(m) if isinstance(m, str) else m
            if not isinstance(f, list):
                f = [f]
            ch_in = [ch_track[j] if j >= 0 else ch_track[-1] for j in f]

            if m in (BasicBlock_ms, ConcatBlock_ms, Conv_2):
                # Stateful or standard conv-like block
                module = m(*args)  # args = [in_ch, out_ch, k, s]
                c2 = int(args[1])
                setattr(module, "out_channels", c2)

            elif m is Concat:
                # Channel concatenation
                module = m(*args)  # args = [dim]
                c2 = int(sum(ch_in))
                setattr(module, "out_channels", c2)

            elif m is Sample:
                # Upsample
                module = m(*args)  # args = [dummy, scale]
                c2 = int(ch_in[0])  # channels unchanged
                setattr(module, "out_channels", c2)

            elif m is Detect:
                # Final detection head
                detect_ch = [ch_track[j] if j >= 0 else ch_track[-1] for j in f]
                module = m(nc=nc, anchors=anchors, ch=detect_ch, inplace=True)
                module.stride = torch.tensor([16., 32.])
                na = len(anchors[0]) if anchors else 3
                c2 = nc * na

            else:
                raise NotImplementedError(f"Unsupported module: {m}")

            layers.append(module)
            ch_track.append(c2)
            out_channels.append(c2)

        return nn.ModuleList(layers), sorted(set(save)), out_channels

    # --------------------- State Initialization ---------------------
    @torch.no_grad()
    def init_states(self, x: torch.Tensor):
        """
        Initializes membrane/spike states for all stateful (_ms) blocks
        using dummy tensors with the correct spatial resolution and channels.
        """
        B, C_in, H, W = x.shape
        device, dtype = x.device, x.dtype

        # Store shapes for each layer output: index 0 = input shape
        outputs_shape = [(C_in, H, W)]

        for i, (f, n, mtype, args) in enumerate(self.model_defs):
            m = self.model[i]
            f_list = f if isinstance(f, list) else [f]

            # Function to get shape from a 'from' index
            def get_shape_from_fromidx(j):
                if j == -1:
                    return outputs_shape[-1]
                else:
                    return outputs_shape[j + 1]

            # Get input shapes
            if len(f_list) == 1:
                in_shapes = [get_shape_from_fromidx(f_list[0])]
            else:
                in_shapes = [get_shape_from_fromidx(j) for j in f_list]

            # Initialize state if block supports it
            if hasattr(m, "init_state"):
                Cin, Hin, Win = in_shapes[0]
                dummy = torch.zeros(B, Cin, Hin, Win, device=device, dtype=dtype)
                m.init_state(dummy)

            # Predict output shape for channel tracking
            if mtype in ("BasicBlock_ms", "ConcatBlock_ms", "Conv_2"):
                s = int(args[3]) if len(args) >= 4 else 1
                Cout = int(args[1])
                Hin, Win = in_shapes[0][1], in_shapes[0][2]
                Hout, Wout = Hin // s, Win // s
                outputs_shape.append((Cout, Hout, Wout))

            elif mtype == "Sample":
                scale = int(args[1]) if len(args) > 1 else 2
                Cin, Hin, Win = in_shapes[0]
                outputs_shape.append((Cin, Hin * scale, Win * scale))

            elif mtype == "Concat":
                Hout, Wout = in_shapes[0][1], in_shapes[0][2]
                Cout = sum(s_[0] for s_ in in_shapes)
                outputs_shape.append((Cout, Hout, Wout))

            elif mtype == "Detect":
                na = len(self.yaml["anchors"][0]) if self.yaml.get("anchors") else 3
                outputs_shape.append((self.nc * na, in_shapes[0][1], in_shapes[0][2]))

            else:
                raise NotImplementedError(f"Unknown module type: {mtype}")

    # --------------------- Single Timestep Forward ---------------------
    def forward_once(self, x):
        """
        Forward pass for one timestep:
        - ms blocks: forward_step(x)
        - Concat: concatenate feature maps
        - Sample/Conv_2: normal forward
        - Detect: skipped during timesteps (only used at the end)
        """
        stateful_blocks = (BasicBlock_ms, ConcatBlock_ms)
        feat_maps = {}
        outputs = [x]  # outputs[0] = model input

        for i, m in enumerate(self.model):
            f = self.model_defs[i][0]

            # Gather inputs for this layer
            if isinstance(f, int):
                x_in = outputs[-1] if f == -1 else outputs[f]
            else:
                x_in = [outputs[j if j != -1 else -1] for j in f]

            # Run layer
            if isinstance(m, stateful_blocks):
                x_out = m.forward_step(x_in)
            elif isinstance(m, Concat):
                x_out = m(x_in)
            elif isinstance(m, Detect):
                # Skip during intermediate timesteps
                x_out = x_in
            else:
                x_out = m(x_in)

            # Optional: log spike rate for debugging
            if self.log_spike and isinstance(m, stateful_blocks) and hasattr(m, "last_spk") and m.last_spk is not None:
                with torch.no_grad():
                    rate = float(m.last_spk.float().mean().item())
                tqdm.write(f"[spike] Layer {i:02d} {type(m).__name__}: {rate:.4f}")

            outputs.append(x_out)

            # Save features for Detect inputs (based on YAML detect_from indices)
            if i in [14, 10]:  # Adjust if detect_from changes
                feat_maps[i] = x_out

        return feat_maps

    # --------------------- Multi-Timestep Forward ---------------------
    def forward(self, x):
        """
        Full forward pass over num_steps timesteps:
        - Accumulates Detect input features over timesteps
        - Runs Detect only once at the end using averaged features
        """
        # Initialize states for all ms blocks
        self.init_states(x)

        accum_feats = {}
        for t in range(self.num_steps):
            feat_maps = self.forward_once(x)
            for k, feat in feat_maps.items():
                if k not in accum_feats:
                    accum_feats[k] = feat
                else:
                    accum_feats[k] += feat

        # Average features over timesteps
        detect_from = self.model_defs[-1][0]  # e.g., [14, 10]
        avg_feats = [accum_feats[i] / self.num_steps for i in detect_from]

        # Final detection
        return self.model[-1](avg_feats)
