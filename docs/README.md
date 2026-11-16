# SET! — on-device Set solver (web, v3)

Runs entirely in the phone's browser: no server, no upload. Two small models run with
onnxruntime-web on WebGPU (WebAssembly as fallback):

- **Card detector:** YOLO26n-OBB, which finds each card as a rotated box at 640 px.
- **Card reader:** MobileNetV4-small, which reads number, colour, shape and fill from a 96×144 crop
  and also says whether the crop is a fully visible card.

Both are trained from scratch-made synthetic tables (`src/synth/`, `scripts/v3/`). Your own game
videos can be added as extra training data and are used as the test set.

## Run it on your phone

The camera only works over **https** (or `localhost`). GitHub Pages serves this `docs/` folder.
Locally: `python3 -m http.server 8000` in this folder, then open `http://localhost:8000`.

## How it decides (v3)

- **Every card on its own.** Each card is tracked and read separately. A set is called as soon as
  *its own three cards* are read reliably. A hand, a card being dealt, or a misread card elsewhere on
  the table no longer blocks everything.
- **Reliable reading.** A card counts as known when its last two readings agree and the combined
  certainty is high. The reader is calibrated, so "90% sure" means roughly that. A card that was
  covered, moved, or swapped is read again before it is trusted.
- **Compute where it matters.** New, uncovered and moved cards are read first. Known cards get a
  quick re-check about every 0.7 s. Reading may take up to half as long as detecting (at least
  40 ms), so a fast phone reads every card that needs it each step and a slow one a few.
- **No lock-in.** The detector searches the whole frame every other step and a region around the
  cards in between, so it can't get stuck on part of the table. It always runs at 640 px; the
  320/480 fallbacks are gone because they couldn't see the cards.
- **Sets stay up** until one of their cards is picked up. Several sets are shown in different colours.
- Models are loaded one after another, so both get WebGPU. Settings → bottom line shows the backend
  per model.

## How well it works (test video, 1080p, warm indoor light)

Whole app replayed with the video as camera (headless Chromium on single-threaded WASM, about 3
steps/s, so a phone is faster). Last 36 s of the video were never used for training.

| | v1 | v2 | v3 |
|---|---|---|---|
| Sets on the table ≥1.5 s that got called | 2 of 10 | 0 of 10 | 9 of 10 |
| Wrong "Set!" calls | most calls | none (no calls) | 0 of 11 |
| Median time to call | 8.3 s | – | 1.0 s |
| Single-frame card reads correct (test part) | 76% | 76% | 99% |

The one missed set had its third card still in the dealer's hand.

## Files

- `index.html`: layout and settings
- `js/main.js`: main loop (detect → track → read → call)
- `js/runtime.js`: onnxruntime-web set-up and model loading
- `js/detect.js`: detector pre- and post-processing
- `js/classify.js`: straight card crops (GPU canvas) and the reader
- `js/tracker.js`: card tracks, evidence, what to read next
- `js/sets.js`: calling sets
- `js/draw.js`: overlay and set list
- `js/camera.js`, `js/sound.js`, `js/geometry.js`: camera, sound and geometry helpers
- `models/`: `detector_v3.onnx`, `classifier_v3.onnx`

## Notes

- onnxruntime-web is loaded from the jsDelivr CDN.
- Licences: the original repo is MIT. The detector is an Ultralytics YOLO26 model, and Ultralytics
  models are AGPL-3.0. That's fine for a personal demo; check it before distributing anything.
