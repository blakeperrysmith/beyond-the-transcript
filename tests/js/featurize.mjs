// Usage: node featurize.mjs jobs.json out.json
// jobs.json: [{ "file": "clip.f32", "sr": 44100 }, ...]; clip files are raw little-endian float32.
import { readFileSync, writeFileSync } from "node:fs";
import { featurize } from "../../app/static/ondevice/features.js";

const jobs = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = jobs.map((j) => {
  const b = readFileSync(j.file);
  const pcm = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  const w = featurize(pcm, j.sr);
  return { cropped: w.cropped, data: Array.from(w.data) };
});
writeFileSync(process.argv[3], JSON.stringify(out));
