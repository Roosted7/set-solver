# v3 training and evaluation

Everything is written inside the project. `data/`, `runs/`, `scratch/` and `archive/` are gitignored.

```
src/synth/cards.py    draws any of the 81 cards with a random deck style (squiggle traced from a real deck)
src/synth/boards.py   table scenes: layouts, clutter, hands, camera angle, lighting, blur, JPEG
src/synth/crops.py    the card crop geometry, identical to docs/js/classify.js
```

## 1. Synthetic data

```bash
python scripts/v3/gen_dataset.py --out data/synth_v3 --train 12000 --val 600       # detector
python scripts/v3/gen_dataset.py --out data/synth_v3_cls2 --train 8000 --val 400 --seed 3   # classifier
```

## 2. Train

```bash
# detector: YOLO26n-OBB; put yolo26n-obb.pt in data/weights and run from there
# (Ultralytics downloads helper files into the working directory)
cd data/weights && python ../../scripts/v3/train_detector.py --epochs 40    # --resume to continue
# classifier: MobileNetV4-small, synthetic only ...
python scripts/v3/train_classifier.py --data data/synth_v3_cls2 --epochs 8 --out runs/cls_v3_synth
# ... then a short fine-tune that also uses real crops from your own games
python scripts/v3/train_classifier.py --data data/synth_v3_cls2 --real data/real/game_hq/crops \
    --init runs/cls_v3_synth/best.pt --epochs 3 --lr 5e-4 --out runs/cls_v3_ft
```

On a 4 GB laptop GPU, run one training at a time.

## 3. Real game videos → labels

```bash
python scripts/v3/pipeline.py data/videos/game_hq.mp4 --every 6 --out scratch/preds.json
python scripts/v3/label_video.py tracks data/videos/game_hq.mp4 scratch/preds.json data/real/game_hq
# look at data/real/game_hq/sheet_*.jpg, write data/real/game_hq/labels.json (one label per card track)
python scripts/v3/label_video.py export data/videos/game_hq.mp4 data/real/game_hq --until 36   # crops: first half only
```

Use the first part of a video for training crops and the rest as the test set. Neighbouring frames
are almost identical, so a random frame split would mostly test memory.

## 4. Export and evaluate

```bash
python scripts/v3/export_onnx.py --det runs/det_v3/weights/best.pt --cls runs/cls_v3_ft/best_calibrated.pt --out docs/models
python scripts/v3/eval_real.py data/videos/game_hq.mp4 data/real/game_hq --from 36 --old <v1 classifier.onnx>
# whole app in headless Chromium with the video as camera (see scripts/v3/replay/replay.mjs), then:
node scripts/v3/replay/replay.mjs docs scratch/game_hq.mjpeg scratch/replay.json 76 '{"set.camRes":"1080"}'
python scripts/v3/score_replay.py scratch/replay.json data/real/game_hq/truth.json
```
