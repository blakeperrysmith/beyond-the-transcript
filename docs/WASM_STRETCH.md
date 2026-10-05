# On-device inference (the door is built, nothing is behind it yet)

The page already has the "Where the analysis runs" panel with a server / on-your-device switch.
Today the on-device option is disabled and says why. It switches on by itself when these files exist
under `app/static/ondevice/`. No other front-end change is needed.

## Contract

`manifest.json`
```json
{ "engine": "tract-wasm", "model": "model.onnx", "size_kb": 480 }
```
`engine` must be present or the page treats the folder as empty. `size_kb` is shown to the visitor.

`ondevice.js`: an ES module with one export.
```js
export async function load({ modelUrl })  // -> { run(pcm, sampleRate) }
// run(pcm: Float32Array, sampleRate: number) -> Promise<{
//   logits: number[6],      // same class order as meta.json, before temperature scaling
//   model_ms: number,
//   features_ms: number,
//   mel?: string            // optional: base64, 32x100 uint8, same as the server preview
// }>
```
The page applies the server's temperature and abstain threshold (from `/api/model`) to the logits,
so calibration stays in one place. In this mode the page makes no request to the server for the
analysis, does not send timings, and shows Praat measurements as off (Praat has no Wasm build here).
If `load` throws, the page says so and falls back to the server.

Checked in a headless browser with a stub engine: with the stub present the switch enables, the
analysis makes zero calls to `/api/analyze`, and switching back re-runs on the server. With the
folder empty the switch is disabled and the server path is unaffected. The stub was deleted afterwards;
no real on-device model exists yet.

## Still to build behind the door

1. A Rust crate: `tract-onnx` + `wasm-bindgen`, `run(logmel: Float32Array) -> Float32Array` for the
   (1,1,64,300) input. `wasm-pack build --target web`. The build sandbox could not install the
   `wasm32-unknown-unknown` target, so none of this has been compiled.
2. A JavaScript port of `btt/features.py` (resample to 16 kHz, RMS-normalise, 512-point FFT, hop 160,
   64 HTK mel bands 50 to 7600 Hz, log with eps 1e-6, 300-frame window). Write a test that feeds one
   clip to both front ends and requires the log-mel to match to 1e-3. Mismatched features would give
   confident wrong answers, so this test is the gate.
3. Run the same `model.onnx` the server uses, and add a headless-browser parity test against PyTorch.
4. Fallback if tract-in-Wasm gives trouble: onnxruntime-web. Then set `"engine": "onnxruntime-web"`
   and do not call it tract anywhere.
