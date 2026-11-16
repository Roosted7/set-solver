"""Export the v3 models for the web app.

    python scripts/v3/export_onnx.py --det runs/det_v3/weights/best.pt --cls runs/cls_v3/best_calibrated.pt --out docs/models

detector_v3.onnx    input [1,3,640,640] RGB 0..1 letterboxed (pad 114)
                    output [1,6,N] rows cx,cy,w,h,score,angle(rad) in input pixels (no NMS needed for
                    YOLO26's one-to-one head; the app still drops overlapping duplicates)
classifier_v3.onnx  input [n,3,144,96] RGB 0..1 (normalisation inside), output [n,13]:
                    number(3) color(3) shape(3) fill(3) logits, temperature-calibrated, + readable-card logit
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_classifier import CardNet  # noqa: E402
from src.synth.crops import CH, CW  # noqa: E402


def export_cls(weights, out, backbone):
    m = CardNet(backbone, pretrained=False)
    m.load_state_dict(torch.load(weights, map_location='cpu'))
    m.eval()
    x = torch.rand(2, 3, CH, CW)
    torch.onnx.export(m, x, str(out), input_names=['images'], output_names=['logits'],
                      dynamic_axes={'images': {0: 'n'}, 'logits': {0: 'n'}}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    import numpy as np
    d = np.abs(ort.InferenceSession(str(out), providers=['CPUExecutionProvider']).run(None, {'images': x.numpy()})[0] - m(x).detach().numpy()).max()
    print(f'classifier -> {out}  max|onnx-torch| = {d:.2e}')


def export_det(weights, out, imgsz):
    from ultralytics import YOLO
    p = YOLO(str(weights)).export(format='onnx', imgsz=imgsz, opset=17, simplify=True, dynamic=False)
    shutil.move(p, out)
    print(f'detector -> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', type=Path)
    ap.add_argument('--cls', type=Path)
    ap.add_argument('--backbone', default='mobilenetv4_conv_small')
    ap.add_argument('--out', type=Path, default=Path('docs/models'))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cls: export_cls(a.cls, a.out / 'classifier_v3.onnx', a.backbone)
    if a.det: export_det(a.det, a.out / 'detector_v3.onnx', 640)
