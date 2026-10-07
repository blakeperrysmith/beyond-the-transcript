// On-device engine: the same ONNX model the server runs, executed in the browser by ONNX Runtime Web (WebAssembly).
// Contract with the page: see docs/ON_DEVICE.md. Nothing here makes a request except the one-time download
// of the model and the runtime, both from this site.
import { featurize, melPreview, N_MELS, WIN_FRAMES } from "./features.js";

const BASE = new URL("./ort/", import.meta.url);

// Each step is named in its error, so a failure says which one it was (runtime files, model download, session).
async function step(name, fn) {
  try { return await fn(); } catch (e) { throw new Error(`${name}: ${e && e.message ? e.message : e}`); }
}

export async function load({ modelUrl }) {
  const ort = await step("runtime script", () => import(new URL("ort.wasm.min.mjs", BASE).href));
  ort.env.wasm.wasmPaths = BASE.href;
  ort.env.wasm.numThreads = 1; // no cross-origin isolation needed, and the model is tiny
  ort.env.wasm.proxy = false;
  const bytes = await step("model download", async () => {
    const r = await fetch(modelUrl);
    if (!r.ok) throw new Error(`${modelUrl} returned ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
  });
  const session = await step("WebAssembly runtime", () => ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" }));
  const inName = session.inputNames[0], outName = session.outputNames[0];
  const feed = (data) => ({ [inName]: new ort.Tensor("float32", data, [1, 1, N_MELS, WIN_FRAMES]) });
  await step("first run", () => session.run(feed(new Float32Array(N_MELS * WIN_FRAMES)))); // warm-up, so the first real run is not charged for start-up

  return {
    engine: "onnxruntime-web",
    async run(pcm, sampleRate) {
      const t0 = performance.now();
      const win = featurize(pcm, sampleRate);
      const t1 = performance.now();
      const res = await session.run(feed(win.data));
      const t2 = performance.now();
      return { logits: Array.from(res[outName].data), features_ms: t1 - t0, model_ms: t2 - t1, mel: melPreview(win.data), cropped: win.cropped };
    },
  };
}
