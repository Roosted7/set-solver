# SET! — on-device Set solver (web)

Runs entirely in the phone's browser: no server, no upload. Built on the trained models from
[wangtian24/set-solver](https://github.com/wangtian24/set-solver) (YOLO11n card detector +
MobileNetV3-small classifier), converted to ONNX and run with onnxruntime-web
(WebGPU where available, otherwise WebAssembly).

## Run it on your phone

The camera only works over **https** (or `localhost`), so host the folder somewhere with HTTPS.

**GitHub Pages (easiest):**
1. Create a new repo, upload everything in this folder (keep the `models/` folder).
2. Settings → Pages → Deploy from branch → `main`, `/ (root)`.
3. Open `https://<you>.github.io/<repo>/` on the phone, then *Add to Home Screen* for a full-screen app.

**Local test on a PC:** `python3 -m http.server 8000` in this folder, then open
`http://localhost:8000`. Use *⚙︎ → Test on a photo* if the PC has no camera.

## Using it

- **Lens buttons** (0.5× / 1× / Tele / Auto) appear when the phone exposes several back cameras.
  iPhone Safari names them, so the ultra-wide is picked automatically the first time.
  Android Chrome often shows generic names ("Lens 1, 2, 3"); try each one. On some Android
  phones a **zoom slider** going below 1× switches to the ultra-wide instead.
- **🔦 Flash** appears only when the browser supports torch control for the current lens
  (Chrome on Android: yes; iOS Safari: depends on version).
- It waits until the minimum number of cards is down (default 12, see ⚙︎), each card has been
  read consistently over a few frames, and the same set has been found twice. Then it says "Set!"
  and outlines the cards. Tap the screen to cycle through other sets. When the set's cards are
  picked up, it starts looking again.
- If there's no set among the cards, it says "deal 3 more".
- ⚙︎ → Sound: voice, beep, or your own recording (pick an audio file).
- The dashed rectangle is the region the detector is currently looking at (see below).

## Notes

- The detector was trained on photos where the cards fill most of the frame, so with a wide-angle
  lens it detects on a crop around the cards. When it has lost the cards, it tries progressively
  smaller centre crops. Keep the layout roughly central when starting.
- Detector resolution 480 is faster; 640 helps when the cards are far away.
- onnxruntime-web is loaded from the jsDelivr CDN.
- Licences: the original repo is MIT. The detector is an Ultralytics YOLO11 model, and Ultralytics
  models are AGPL-3.0. That's fine for a personal demo; check it before distributing anything.
