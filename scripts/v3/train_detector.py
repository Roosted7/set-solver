"""Train the v3 card detector: YOLO26n-OBB (rotated boxes, NMS-free) on the synthetic v3 scenes.

    python scripts/v3/train_detector.py --data data/synth_v3/data.yaml --epochs 60

Rotated boxes give each card's orientation directly, which the web app uses to cut the card
out straight for the classifier, and separate touching cards far better than upright boxes.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'data/synth_v3/data.yaml'))
    ap.add_argument('--model', default=str(ROOT / 'data/weights/yolo26n-obb.pt'))
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--name', default='det_v3')
    ap.add_argument('--project', default=str(ROOT / 'runs'))
    ap.add_argument('--resume', action='store_true', help='continue the interrupted run (runs/<name>/weights/last.pt)')
    a = ap.parse_args()
    if a.resume:
        YOLO(str(Path(a.project) / a.name / 'weights/last.pt')).train(resume=True)
        return
    YOLO(a.model).train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=12, cache=False,
        project=a.project, name=a.name, exist_ok=True, patience=100, cos_lr=True, close_mosaic=8,
        # scenes are already rotated/tilted/lit at random; keep the built-in augmentation moderate
        degrees=0, translate=0.1, scale=0.6, fliplr=0.5, flipud=0.5, mosaic=1.0, mixup=0.0,
        hsv_h=0.02, hsv_s=0.5, hsv_v=0.4, plots=True,
    )


if __name__ == '__main__':
    main()
