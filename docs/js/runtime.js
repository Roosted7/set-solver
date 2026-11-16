// Model loading with onnxruntime-web.
// Sessions are created one after another: onnxruntime refuses a second WebGPU session while
// another is still being created, which used to push the classifier silently onto single-threaded WASM.

const ORT_VER = '1.31.1';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
export const ort = await import(ORT_BASE + 'ort.webgpu.min.mjs');
ort.env.wasm.wasmPaths = ORT_BASE;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;
try { ort.env.webgpu.powerPreference = 'high-performance'; } catch {}

export const backends = {};   // model name -> execution provider actually used

async function create(name, url) {
  const eps = ('gpu' in navigator) ? ['webgpu', 'wasm'] : ['wasm'];
  let lastErr;
  for (const ep of eps) {
    try {
      const s = await ort.InferenceSession.create(url, { executionProviders: [ep], graphOptimizationLevel: 'all' });
      backends[name] = ep;
      return s;
    } catch (e) { lastErr = e; console.warn(`${name}: ${ep} failed`, e); }
  }
  throw new Error(`Could not load ${url}: ${lastErr?.message}`);
}

export const models = {};
export const modelsReady = (async () => {
  models.det = await create('detector', 'models/detector_v3.onnx');
  models.cls = await create('classifier', 'models/classifier_v3.onnx');
  // warm-up so the first real frame isn't slow (shader compilation)
  await models.det.run({ [models.det.inputNames[0]]: new ort.Tensor('float32', new Float32Array(3 * 640 * 640), [1, 3, 640, 640]) });
  await models.cls.run({ [models.cls.inputNames[0]]: new ort.Tensor('float32', new Float32Array(4 * 3 * 144 * 96), [4, 3, 144, 96]) });
  return models;
})();

export const backendText = () => Object.entries(backends).map(([k, v]) => `${k} ${v}`).join(', ');
