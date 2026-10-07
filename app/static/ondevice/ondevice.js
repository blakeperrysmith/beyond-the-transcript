// On-device engine: the same ONNX model the server runs, executed in the browser by ONNX Runtime Web (WebAssembly).
// Contract with the page: see docs/ON_DEVICE.md. Nothing here makes a request except the one-time download
// of the model and the runtime, both from this site.
import { featurize, melPreview, N_MELS, WIN_FRAMES } from "./features.js";

const BASE = new URL("./ort/", import.meta.url);

export async function load({ modelUrl }) {
  const ort = await import(new URL("ort.wasm.min.mjs", BASE).href);
  ort.env.wasm.wasmPaths = BASE.href;
  ort.env.wasm.numThreads = 1; // no cross-origin isolation needed, and the model is tiny
  ort.env.wasm.proxy = false;
  const bytes = new Uint8Array(await (await fetch(modelUrl)).arrayBuffer());
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  const inName = session.inputNames[0], outName = session.outputNames[0];
  const feed = (data) => ({ [inName]: new ort.Tensor("float32", data, [1, 1, N_MELS, WIN_FRAMES]) });
  await session.run(feed(new Float32Array(N_MELS * WIN_FRAMES))); // warm-up, so the first real run is not charged for start-up

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
