// Browser port of btt/features.py. Same steps, same constants, plain JS with no dependencies.
// tests/test_ondevice_features.py feeds the same clips to this file and to the Python code and
// requires the log-mel to agree to 1e-3, because a mismatch here would give confident wrong answers.

export const SR = 16000, N_FFT = 512, HOP = 160, N_MELS = 64, FMIN = 50, FMAX = 7600;
export const WIN_FRAMES = 300, LOG_EPS = 1e-6, FLOOR = Math.log(LOG_EPS);

const gcd = (a, b) => (b ? gcd(b, a % b) : a);

function bessel0(x) { // modified Bessel function I0, series
  let sum = 1, term = 1;
  for (let k = 1; k < 60; k++) { term *= (x / (2 * k)) ** 2; sum += term; if (term < 1e-18 * sum) break; }
  return sum;
}

function kaiser(n, beta) { // symmetric, as scipy.signal.get_window(("kaiser", beta), n, fftbins=False)
  const w = new Float64Array(n), d = bessel0(beta);
  for (let i = 0; i < n; i++) { const r = (2 * i) / (n - 1) - 1; w[i] = bessel0(beta * Math.sqrt(Math.max(0, 1 - r * r))) / d; }
  return w;
}

const sinc = (x) => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x));

// scipy.signal.resample_poly(y, up, down) with its defaults (Kaiser beta 5, zero padding at the edges).
export function resamplePoly(x, up, down) {
  if (up === 1 && down === 1) return Float32Array.from(x);
  const half = 10 * Math.max(up, down), N = 2 * half + 1, cutoff = 1 / Math.max(up, down);
  const win = kaiser(N, 5), h = new Float64Array(N);
  let sum = 0;
  for (let i = 0; i < N; i++) { h[i] = cutoff * sinc(cutoff * (i - half)) * win[i]; sum += h[i]; }
  for (let i = 0; i < N; i++) h[i] = (h[i] / sum) * up;
  const nOut = Math.ceil((x.length * up) / down), out = new Float32Array(nOut);
  for (let m = 0; m < nOut; m++) {
    const c = m * down + half; // centre of the filter in the zero-stuffed signal
    let jLo = Math.ceil((c - 2 * half) / up), jHi = Math.floor(c / up);
    if (jLo < 0) jLo = 0;
    if (jHi > x.length - 1) jHi = x.length - 1;
    let acc = 0;
    for (let j = jLo; j <= jHi; j++) acc += x[j] * h[c - j * up];
    out[m] = acc;
  }
  return out;
}

export function to16k(y, sr) {
  if (sr === SR) return Float32Array.from(y);
  const g = gcd(SR, Math.round(sr));
  return resamplePoly(y, SR / g, Math.round(sr) / g);
}

export function rmsNormalise(y, target = 0.1) {
  let s = 0;
  for (let i = 0; i < y.length; i++) s += y[i] * y[i];
  const rms = y.length ? Math.sqrt(s / y.length) : 0;
  if (rms < 1e-8) return Float32Array.from(y);
  const k = target / rms, out = new Float32Array(y.length);
  for (let i = 0; i < y.length; i++) out[i] = y[i] * k;
  return out;
}

const hz2mel = (f) => 2595 * Math.log10(1 + f / 700);
const mel2hz = (m) => 700 * (10 ** (m / 2595) - 1);

let _fb = null;
function melFilterbank() { // (N_MELS, N_FFT/2+1), triangular, HTK scale, area-normalised
  if (_fb) return _fb;
  const nb = N_FFT / 2 + 1, pts = [];
  const lo = hz2mel(FMIN), hi = hz2mel(FMAX);
  for (let i = 0; i < N_MELS + 2; i++) pts.push(mel2hz(lo + ((hi - lo) * i) / (N_MELS + 1)));
  const fb = [];
  for (let i = 0; i < N_MELS; i++) {
    const row = new Float64Array(nb), [l, c, r] = [pts[i], pts[i + 1], pts[i + 2]];
    let s = 0;
    for (let k = 0; k < nb; k++) {
      const f = (SR / 2) * (k / (nb - 1));
      row[k] = Math.max(0, Math.min((f - l) / Math.max(c - l, 1e-9), (r - f) / Math.max(r - c, 1e-9)));
      s += row[k];
    }
    s = Math.max(s, 1e-9);
    for (let k = 0; k < nb; k++) row[k] /= s;
    fb.push(row);
  }
  return (_fb = fb);
}

let _win = null;
const hann = () => _win || (_win = Float64Array.from({ length: N_FFT }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N_FFT)));

let _tw = null;
function fft(re, im) { // in place radix-2, length N_FFT
  const n = re.length;
  if (!_tw) _tw = { c: Float64Array.from({ length: n / 2 }, (_, k) => Math.cos((2 * Math.PI * k) / n)), s: Float64Array.from({ length: n / 2 }, (_, k) => -Math.sin((2 * Math.PI * k) / n)) };
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const step = n / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const wr = _tw.c[k * step], wi = _tw.s[k * step], a = i + k, b = a + len / 2;
        const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
      }
    }
  }
}

const reflect = (i, n) => { // numpy "reflect" padding index, no repeat of the edge sample
  if (n === 1) return 0;
  const p = 2 * (n - 1);
  i = ((i % p) + p) % p;
  return i < n ? i : p - i;
};

// Returns { data: Float32Array(N_MELS * T), frames: T }, row-major (mel band, frame).
export function logMel(y16) {
  let y = rmsNormalise(y16);
  if (y.length < N_FFT) { const z = new Float32Array(N_FFT); z.set(y); y = z; }
  const pad = N_FFT / 2, nFrames = 1 + Math.floor((y.length + 2 * pad - N_FFT) / HOP);
  const fb = melFilterbank(), w = hann(), nb = N_FFT / 2 + 1;
  const out = new Float32Array(N_MELS * nFrames), re = new Float64Array(N_FFT), im = new Float64Array(N_FFT), power = new Float64Array(nb);
  for (let t = 0; t < nFrames; t++) {
    for (let i = 0; i < N_FFT; i++) { re[i] = y[reflect(t * HOP + i - pad, y.length)] * w[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k < nb; k++) power[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < N_MELS; m++) {
      let s = 0; const row = fb[m];
      for (let k = 0; k < nb; k++) s += power[k] * row[k];
      out[m * nFrames + t] = Math.log(s + LOG_EPS);
    }
  }
  return { data: out, frames: nFrames };
}

// Exactly WIN_FRAMES frames: the most energetic window if longer, end-padded with the log floor if shorter.
export function fitWindow({ data, frames }, n = WIN_FRAMES) {
  const out = new Float32Array(N_MELS * n);
  if (frames < n) {
    out.fill(FLOOR);
    for (let m = 0; m < N_MELS; m++) out.set(data.subarray(m * frames, (m + 1) * frames), m * n);
    return { data: out, cropped: false };
  }
  let start = 0;
  if (frames > n) {
    const e = new Float64Array(frames);
    for (let t = 0; t < frames; t++) { let s = 0; for (let m = 0; m < N_MELS; m++) s += data[m * frames + t]; e[t] = s / N_MELS; }
    let run = 0;
    for (let t = 0; t < n; t++) run += e[t];
    let best = run;
    for (let s0 = 1; s0 + n <= frames; s0++) { run += e[s0 + n - 1] - e[s0 - 1]; if (run > best) { best = run; start = s0; } }
  }
  for (let m = 0; m < N_MELS; m++) out.set(data.subarray(m * frames + start, m * frames + start + n), m * n);
  return { data: out, cropped: frames > n };
}

// What the server receives: 16-bit PCM, divided by 32768. Doing the same here keeps both engines on identical samples.
export function quantise16(f32) {
  const out = new Float32Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    out[i] = Math.trunc(s < 0 ? s * 32768 : s * 32767) / 32768;
  }
  return out;
}

export function featurize(pcm, sr) {
  return fitWindow(logMel(to16k(quantise16(pcm), sr)));
}

// Same 32x100 uint8 preview the server draws from (see _mel_preview in app/main.py).
export function melPreview(win) {
  const f = new Float64Array(32 * 100), T = WIN_FRAMES;
  for (let r = 0; r < 32; r++) for (let c = 0; c < 100; c++) {
    let s = 0;
    for (let a = 0; a < 2; a++) for (let b = 0; b < 3; b++) s += win[(r * 2 + a) * T + c * 3 + b];
    f[(31 - r) * 100 + c] = s / 6; // low frequencies at the bottom
  }
  const sorted = Float64Array.from(f).sort();
  const pctl = (q) => { const p = (q / 100) * (sorted.length - 1), lo = Math.floor(p), hi = Math.ceil(p); return sorted[lo] + (sorted[hi] - sorted[lo]) * (p - lo); };
  const lo = pctl(2), hi = pctl(99.5), den = Math.max(hi - lo, 1e-6), u8 = new Uint8Array(3200);
  for (let i = 0; i < 3200; i++) u8[i] = Math.trunc(Math.min(1, Math.max(0, (f[i] - lo) / den)) * 255);
  let bin = "";
  for (let i = 0; i < u8.length; i++) bin += String.fromCharCode(u8[i]);
  return btoa(bin);
}
