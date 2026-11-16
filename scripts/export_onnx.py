"""
Export the trained models to ONNX for the in-browser web app (docs/).

    python scripts/export_onnx.py \
        --detector weights/detector/weights/best.pt \
        --classifier weights/classifier_best.pt \
        --out docs/models

Weights: https://huggingface.co/wangtianthu/set-solver-models
  detector/weights/best.pt  and  classifier/classifier_best.pt

Produces:
  detector_{320,480,640}.onnx      input [1,3,S,S] RGB 0..1, letterboxed (pad 114)
                                          output [1,5,N] = cx,cy,w,h,score (no NMS)
  classifier.onnx                         input [n,3,224,224] RGB 0..1 (ImageNet normalisation
                                          is baked in), output [n,12] logits =
                                          number(3) | color(3) | shape(3) | fill(3)
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.train.classifier import SetCardClassifier  # noqa: E402


class ClassifierForWeb(nn.Module):
    """Wraps the 4-head classifier: normalises input and concatenates the heads."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        o = self.model((x - self.mean) / self.std)
        return torch.cat([o["number"], o["color"], o["shape"], o["fill"]], 1)


def export_classifier(weights: Path, out: Path):
    model = SetCardClassifier(pretrained=False)
    ckpt = torch.load(weights, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    wrapped = ClassifierForWeb(model).eval()
    x = torch.rand(2, 3, 224, 224)
    torch.onnx.export(
        wrapped, x, str(out), input_names=["images"], output_names=["logits"],
        dynamic_axes={"images": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False,
    )
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out))
        diff = np.abs(sess.run(None, {"images": x.numpy()})[0] - wrapped(x).detach().numpy()).max()
        print(f"classifier: max |onnx - torch| = {diff:.2e}")
    except ImportError:
        pass


def export_detector(weights: Path, out_dir: Path, sizes):
    from ultralytics import YOLO
    for s in sizes:
        path = YOLO(str(weights)).export(format="onnx", imgsz=s, opset=17, simplify=True, dynamic=False)
        shutil.move(path, out_dir / f"detector_{s}.onnx")
        print(f"detector: {out_dir / f'detector_{s}.onnx'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", type=Path, default=Path("weights/detector/weights/best.pt"))
    ap.add_argument("--classifier", type=Path, default=Path("weights/classifier_best.pt"))
    ap.add_argument("--out", type=Path, default=Path("docs/models"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[320, 480, 640])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    export_classifier(args.classifier, args.out / "classifier.onnx")
    export_detector(args.detector, args.out, args.sizes)


if __name__ == "__main__":
    main()
