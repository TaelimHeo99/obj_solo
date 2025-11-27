# convert_ann_to_snn.py
#
# ANN YOLOv3-tiny (yolov3-tiny.pt) 의 Conv/BN weight를
# STBP_YOLOTiny SNN 모델로 복사해서 새로운 체크포인트를 만드는 스크립트

import torch
import torch.nn as nn

from ultralytics.utils.patches import torch_load
from models.stbp_model import STBP_YOLOTiny


def get_conv_bn_lists(model: nn.Module):
    """
    model 안의 Conv2d / BatchNorm2d 를 등장 순서대로 리스트로 뽑아준다.
    (forward 그래프 순서와 modules() 순서가 동일하다는 가정)
    """
    convs = []
    bns = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            convs.append(m)
        elif isinstance(m, nn.BatchNorm2d):
            bns.append(m)
    return convs, bns


def main():
    # 1) ANN YOLOv3-tiny 체크포인트 로드
    ann_ckpt_path = "yolov3-tiny.pt"  # 필요하면 경로 수정
    print(f"[INFO] Loading ANN checkpoint from: {ann_ckpt_path}")
    ann_ckpt = torch_load(ann_ckpt_path, map_location="cpu")

    # Ultralytics 스타일: ckpt["model"] 안에 모델이 들어있음
    ann_model = ann_ckpt["model"].float()
    ann_convs, ann_bns = get_conv_bn_lists(ann_model)
    print(f"[INFO] ANN conv layers: {len(ann_convs)}, bn layers: {len(ann_bns)}")

    # 2) SNN STBP_YOLOTiny 모델 생성 (COCO용 nc=80, anchors는 train.py에서 쓰던 것과 동일하게)
    #    나중에 rf100에 쓸 때는 어차피 Detect head는 새로 초기화되니 여기선 COCO 기준으로만 맞춰도 됨
    anchors = ((10, 14, 23, 27, 37, 58),
               (81, 82, 135, 169, 344, 319))
    nc = 80  # COCO 클래스 수
    num_steps = 5  # 너가 쓰는 기본 T

    print("[INFO] Building SNN STBP_YOLOTiny model...")
    snn_model = STBP_YOLOTiny(nc=nc, anchors=anchors, num_steps=num_steps)
    snn_convs, snn_bns = get_conv_bn_lists(snn_model)
    print(f"[INFO] SNN conv layers: {len(snn_convs)}, bn layers: {len(snn_bns)}")

    # 3) Conv weight 복사
    n_conv = min(len(ann_convs), len(snn_convs))
    copied_conv = 0
    for i in range(n_conv):
        a = ann_convs[i]
        s = snn_convs[i]

        if a.weight.shape == s.weight.shape:
            s.weight.data.copy_(a.weight.data)
            if a.bias is not None and s.bias is not None and a.bias.shape == s.bias.shape:
                s.bias.data.copy_(a.bias.data)
            copied_conv += 1
        else:
            print(f"[WARN] Conv[{i}] shape mismatch: ANN {a.weight.shape} vs SNN {s.weight.shape} (skip)")

    print(f"[INFO] Copied {copied_conv}/{n_conv} Conv layers")

    # 4) BatchNorm weight 복사
    n_bn = min(len(ann_bns), len(snn_bns))
    copied_bn = 0
    for i in range(n_bn):
        a = ann_bns[i]
        s = snn_bns[i]

        if (a.weight.shape == s.weight.shape and
            a.bias.shape == s.bias.shape and
            a.running_mean.shape == s.running_mean.shape and
            a.running_var.shape == s.running_var.shape):
            s.weight.data.copy_(a.weight.data)
            s.bias.data.copy_(a.bias.data)
            s.running_mean.data.copy_(a.running_mean.data)
            s.running_var.data.copy_(a.running_var.data)
            copied_bn += 1
        else:
            print(f"[WARN] BN[{i}] shape mismatch (skip)")

    print(f"[INFO] Copied {copied_bn}/{n_bn} BatchNorm layers")

    # 5) 새로운 체크포인트로 저장
    out_path = "yolov3tiny_ann2snn_stbp.pt"
    torch.save({"model": snn_model}, out_path)
    print(f"[INFO] Saved converted SNN checkpoint to: {out_path}")


if __name__ == "__main__":
    main()
