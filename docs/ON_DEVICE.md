# On-device inference

The page has a "Where the analysis runs" switch. On the server (default) the audio is posted to `/api/analyze`.
On your device the same ONNX model runs in the browser as WebAssembly and nothing is sent for the analysis.

## What is built

- `app/static/ondevice/features.js`: a JavaScript port of `btt/features.py` (resample to 16 kHz with the same
  Kaiser polyphase filter as `scipy.signal.resample_poly`, RMS normalise, 512-point FFT, 64 HTK mel bands,
  log, most-energetic 3 s window). It also applies the 16-bit conversion the server's input goes through, so both
  engines see identical samples.
- `app/static/ondevice/ondevice.js`: loads ONNX Runtime Web, creates a session from `/model.onnx` (the file the
  server itself runs) and returns logits.
- `app/static/ondevice/ort/`: the vendored ONNX Runtime Web files (MIT licence, `ort.wasm.min.mjs`,
  `ort-wasm-simd-threaded.mjs`, `ort-wasm-simd-threaded.wasm`, version 1.30.0). About 14 MB, almost all of it the
  WebAssembly runtime. Single-threaded, so no cross-origin isolation headers are needed.
- `app/static/ondevice/manifest.json`: its presence switches the option on. Delete the folder contents and the
  option shows as disabled with an explanation; nothing else changes.

## Contract with the page

`ondevice.js` is an ES module with one export:

```js
export async function load({ modelUrl })  // -> { run(pcm, sampleRate) }
// run(pcm: Float32Array, sampleRate: number) -> Promise<{
//   logits: number[6],      // same class order as meta.json, before temperature scaling
//   model_ms: number, features_ms: number,
//   mel?: string,           // base64, 32x100 uint8, same preview the server sends
//   cropped?: boolean
// }>
```

The page applies the temperature and abstain threshold from `/api/model`, so calibration lives in one place.
In this mode there is no request to `/api/analyze`, no timings are sent, and Praat shows as off (Praat has no
WebAssembly build here). If `load` throws, the page says so and falls back to the server.

## How it is checked

- `tests/test_ondevice_features.py`: the JavaScript log-mel against the Python one, 8, 16, 22.05, 44.1 and
  48 kHz input, short and long clips. Required: 1e-3 maximum absolute error. Measured: under 2e-4.
- `tests/test_ondevice_browser.py`: headless Chromium runs the WebAssembly model and the server runs tract on the same
  clip at 16, 44.1 and 48 kHz. Required: class probabilities within 2e-3.

If the features disagree the model would give confident wrong answers, which is why that test is the gate.

## What it is not

- It is ONNX Runtime Web, not tract compiled to WebAssembly. The tract-in-Wasm route (a Rust crate with `tract-onnx`
  and `wasm-bindgen`, built with `wasm-pack`) was not attempted because the `wasm32-unknown-unknown` target could not
  be installed in the build sandbox. It would shrink the download from about 14 MB to well under 1 MB and is the
  obvious next step.
- Praat is not available on device.
- Timings from on-device runs are not added to the shared latency statistics.
