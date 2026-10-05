"use strict";
/* Beyond the transcript: front end.
 * In server mode, audio leaves the browser only as the raw PCM body of one POST to /api/analyze.
 * In on-device mode it does not leave at all. On-device mode needs files under /static/ondevice/
 * (see docs/WASM_STRETCH.md for the contract); without them the option is shown but disabled.
 * Nothing here uses third-party scripts, fonts or analytics.
 */
(() => {
  const $ = (id) => document.getElementById(id);
  const MAX_REC_S = 6;
  const F0_LO = 75, F0_HI = 500; // shared pitch axis (Hz, log) so A and B are comparable
  const WINDOW_S = 3;

  const state = {
    model: null,
    manifest: null,
    ctx: null,
    slots: {
      a: { pcm: null, sr: 0, result: null, source: null, label: "Empty", playing: null, run: null },
      b: { pcm: null, sr: 0, result: null, source: null, label: "Empty", playing: null, run: null },
    },
    recording: null,
    stats: null,
    storage: null,
    pairToken: 0,
    engine: "server", // "server" | "device"
    device: { status: "checking", info: null, eng: null, error: "" }, // checking | unavailable | available | loading | ready | error
  };

  // ---------- small helpers ----------
  const el = (tag, props = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else n.setAttribute(k, v);
    }
    for (const k of kids) n.append(k);
    return n;
  };
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  const fmtMs = (v) => (v == null ? "n/a" : v >= 100 ? `${Math.round(v)} ms` : `${v.toFixed(1)} ms`);
  const pct = (v) => `${Math.round(v * 100)}%`;
  const showError = (msg) => {
    const e = $("error");
    e.textContent = msg || "";
    e.hidden = !msg;
  };
  const audioCtx = () => {
    if (!state.ctx) state.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (state.ctx.state === "suspended") state.ctx.resume();
    return state.ctx;
  };

  // ---------- network ----------
  async function getJSON(url) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`${url} returned ${r.status}`);
    return r.json();
  }

  function toInt16(f32) {
    const out = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
      const s = Math.max(-1, Math.min(1, f32[i]));
      out[i] = s < 0 ? s * 32768 : s * 32767;
    }
    return out;
  }

  async function analyse(f32, sr, source) {
    const body = toInt16(f32);
    const t0 = performance.now();
    const r = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-Sample-Rate": String(Math.round(sr)), "X-Source": source },
      body,
    });
    if (!r.ok) {
      let detail = `The server returned ${r.status}.`;
      try { detail = (await r.json()).detail || detail; } catch (_) { /* keep default */ }
      throw new Error(detail);
    }
    const data = await r.json();
    data.e2e_ms = performance.now() - t0;
    if (data.run_id == null) return data; // the server could not log this run; that is fine
    fetch("/api/client-latency", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: data.run_id, e2e_ms: data.e2e_ms }),
    })
      .then((x) => (x.ok ? x.json() : null))
      .then((j) => { if (j) { state.stats = j.stats; state.storage = j.storage || state.storage; renderAll(); } })
      .catch(() => {});
    return data;
  }

  // ---------- on-device engine (the door) ----------
  const DEVICE_DIR = "/static/ondevice/";

  async function probeDevice() {
    try {
      const info = await getJSON(`${DEVICE_DIR}manifest.json`);
      if (!info || !info.engine) throw new Error("bad manifest");
      state.device = { status: "available", info, eng: null, error: "" };
    } catch (_) {
      state.device = { status: "unavailable", info: null, eng: null, error: "" };
    }
    renderEngine();
  }

  function loadDevice() {
    const d = state.device;
    if (d.status === "ready") return Promise.resolve(true);
    if (d.promise) return d.promise;
    if (d.status !== "available" && d.status !== "error") return Promise.resolve(false);
    d.status = "loading"; renderEngine();
    d.promise = (async () => {
      try {
        const mod = await import(`${DEVICE_DIR}ondevice.js`);
        d.eng = await mod.load({ modelUrl: `${DEVICE_DIR}${d.info.model || "model.onnx"}` });
        d.status = "ready";
      } catch (e) {
        d.status = "error";
        d.error = "The on-device model could not be loaded in this browser.";
      }
      d.promise = null;
      renderEngine();
      return d.status === "ready";
    })();
    return d.promise;
  }

  function softmax(logits, temperature) {
    const z = logits.map((v) => v / temperature);
    const m = Math.max(...z);
    const e = z.map((v) => Math.exp(v - m));
    const sum = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / sum);
  }

  // Same result shape as the server's, minus Praat measurements, and with no network request at all.
  async function analyseOnDevice(f32, sr) {
    const t0 = performance.now();
    const out = await state.device.eng.run(f32, sr); // { logits, model_ms, features_ms, mel? }
    const m = state.model;
    const p = softmax(out.logits, m.temperature);
    const top = p.indexOf(Math.max(...p));
    const abstained = p[top] < m.abstain_threshold;
    return {
      engine: "device",
      label: abstained ? null : m.classes[top],
      abstained,
      confidence: p[top],
      probs: Object.fromEntries(m.classes.map((c, i) => [c, p[i]])),
      prosody: null,
      mel: out.mel || null,
      was_cropped: false,
      timings: { features_ms: out.features_ms, model_ms: out.model_ms, total_ms: performance.now() - t0 },
    };
  }

  function setEngine(which) {
    if (which === state.engine) return;
    state.engine = which;
    renderEngine();
    if (which === "device") loadDevice();
    // re-run whatever is on screen with the engine just chosen
    for (const k of ["a", "b"]) {
      const s = state.slots[k];
      if (s.pcm && s.source) runSlot(k, s.source);
    }
  }

  function renderEngine() {
    const d = state.device;
    const onDevice = $("engine").querySelector('input[value="device"]');
    const onServer = $("engine").querySelector('input[value="server"]');
    onDevice.disabled = d.status === "unavailable" || d.status === "checking";
    onServer.checked = state.engine === "server";
    onDevice.checked = state.engine === "device";

    const box = $("engine-text");
    box.textContent = "";
    const para = (...kids) => box.append(el("p", {}, ...kids));
    if (state.engine === "device" && d.status === "ready") {
      para(el("strong", { text: "Your audio stays in this browser tab. " }), "The model runs on your own device as WebAssembly, and nothing is sent to the server for the analysis.");
      para("What you give up: the model downloads once", d.info && d.info.size_kb ? ` (about ${d.info.size_kb} KB)` : "", ", speed depends on your device, and the Praat measurements are off because Praat only runs on the server. Timings from this mode are not added to the shared statistics below.");
    } else if (state.engine === "device") {
      para(el("strong", { text: "Your audio stays in this browser tab. " }), "The model is being loaded onto your device. Nothing is sent to the server for the analysis.");
    } else {
      para(el("strong", { text: "Your audio is sent to this site's server. " }), "It travels over an encrypted connection, is analysed in memory and is thrown away. The server keeps one line of timings per run: no audio, no address, no account.");
      para("Praat measurements and the model both run there. It is the only mode that shows the Praat measurements.");
    }
    const st = $("engine-status");
    if (d.status === "unavailable") st.textContent = "On-device mode is not built into this version yet, so the model runs only on the server. When it is, your audio would stay in the browser.";
    else if (d.status === "checking") st.textContent = "Checking whether on-device mode is available…";
    else if (d.status === "loading") st.textContent = "Loading the on-device model…";
    else if (d.status === "error") st.textContent = `${d.error} The server option still works.`;
    else st.textContent = "";
  }

  // ---------- drawing ----------
  // Dark blue to pale yellow ramp, built once.
  const RAMP = (() => {
    const stops = [[15, 27, 43], [52, 62, 140], [140, 70, 160], [226, 118, 98], [250, 214, 140], [255, 250, 225]];
    const out = [];
    for (let i = 0; i < 256; i++) {
      const t = (i / 255) * (stops.length - 1);
      const k = Math.min(stops.length - 2, Math.floor(t));
      const f = t - k;
      out.push(stops[k].map((c, j) => Math.round(c + (stops[k + 1][j] - c) * f)));
    }
    return out;
  })();

  function drawSlot(key) {
    const s = state.slots[key];
    const canvas = $(`spec-${key}`);
    const g = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--scope") || "#0f1b2b";
    g.fillRect(0, 0, W, H);
    const res = s.result;
    if (!res) return;
    if (!res.mel) {
      g.fillStyle = "rgba(255,255,255,0.7)";
      g.font = "14px system-ui, sans-serif";
      g.fillText("No spectrogram in this mode.", 16, H / 2);
      return;
    }

    // spectrogram: 32 mel rows x 100 columns, drawn at 3 s full width
    const bytes = Uint8Array.from(atob(res.mel), (c) => c.charCodeAt(0));
    const img = new ImageData(100, 32);
    for (let i = 0; i < 3200; i++) {
      const c = RAMP[bytes[i]];
      img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2]; img.data[i * 4 + 3] = 255;
    }
    const tmp = document.createElement("canvas");
    tmp.width = 100; tmp.height = 32;
    tmp.getContext("2d").putImageData(img, 0, 0);
    g.imageSmoothingEnabled = true;
    g.drawImage(tmp, 0, 0, W, H);

    // pitch line, only when the drawn window is the whole clip
    const contour = res.prosody && res.prosody.f0_contour;
    if (contour && !res.was_cropped) {
      const line = key === "a" ? "#9fb0ff" : "#ffb27a";
      const y = (f) => H - ((Math.log(f) - Math.log(F0_LO)) / (Math.log(F0_HI) - Math.log(F0_LO))) * H;
      g.lineWidth = 2.5;
      g.strokeStyle = line;
      g.lineJoin = "round";
      g.beginPath();
      let pen = false;
      contour.forEach((f, i) => {
        const x = ((i * 0.02) / WINDOW_S) * W;
        if (f > 0) { pen ? g.lineTo(x, y(f)) : g.moveTo(x, y(f)); pen = true; }
        else pen = false;
      });
      g.stroke();
    }
    // 1 s ticks
    g.fillStyle = "rgba(255,255,255,0.55)";
    g.font = "12px system-ui, sans-serif";
    for (let t = 1; t < WINDOW_S; t++) {
      const x = (t / WINDOW_S) * W;
      g.fillRect(x, H - 6, 1, 6);
      g.fillText(`${t} s`, x + 4, H - 8);
    }
  }

  // ---------- rendering ----------
  function renderSlot(key) {
    const s = state.slots[key];
    $(`cap-${key}`).textContent = s.label;
    $(`play-${key}`).disabled = !s.pcm;
    $(`play-${key}`).textContent = s.playing ? "Stop" : "Play";
    drawSlot(key);
    const v = $(`verdict-${key}`);
    v.textContent = "";
    if (s.run === "pending") { v.textContent = "Analysing…"; return; }
    if (s.run && s.run.error) { v.textContent = s.run.error; return; }
    const r = s.result;
    if (!r) return;
    if (r.abstained) {
      const top = Object.entries(r.probs).sort((x, y) => y[1] - x[1])[0];
      v.append(el("strong", { text: "Not sure." }), ` The model is not confident enough to name a delivery (its top guess is ${top[0]} at ${pct(top[1])}).`);
    } else {
      v.append(el("strong", { text: cap(r.label) }), ` at ${pct(r.confidence)} confidence.`);
    }
    if (r.was_cropped) v.append(el("br"), "Clip is longer than 3 s, so the model heard the loudest 3 s and the pitch line is hidden.");
  }

  function fmtMeasure(key, v) {
    if (v == null) return "n/a";
    if (key === "voiced_frac" || key === "pause_frac") return pct(v);
    if (key === "f0_median_hz") return `${Math.round(v)}`;
    return v.toFixed(1);
  }

  function renderMeasures() {
    const body = $("measure-body");
    const a = state.slots.a.result, b = state.slots.b.result;
    const desc = state.model && state.model.prosody_descriptions;
    body.textContent = "";
    if (!desc || (!a && !b)) {
      body.append(el("tr", {}, el("td", { colspan: "3", class: "empty", text: "Nothing measured yet." })));
      return;
    }
    const pa = a && a.prosody, pb = b && b.prosody;
    if (!pa && !pb) {
      body.append(el("tr", {}, el("td", { colspan: "3", class: "empty", text: "Praat measurements run on the server, so they are off in on-device mode. Switch to the server to see them." })));
      return;
    }
    for (const [k, [label, unit, why]] of Object.entries(desc)) {
      if (k === "duration_s") continue;
      const va = pa ? pa[k] : undefined, vb = pb ? pb[k] : undefined;
      const name = el("td", { title: why, text: unit ? `${label} (${unit})` : label });
      const ta = el("td", { text: pa ? fmtMeasure(k, va) : a ? "n/a" : "" });
      const tb = el("td", { text: pb ? fmtMeasure(k, vb) : b ? "n/a" : "" });
      if (pa && pb && va != null && vb != null) {
        const d = vb - va;
        const share = k === "voiced_frac" || k === "pause_frac";
        const txt = share ? `${d >= 0 ? "+" : "−"}${Math.round(Math.abs(d) * 100)} pts` : `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(k === "f0_median_hz" ? 0 : 1)}`;
        tb.append(el("span", { class: "diff", text: txt }));
      }
      body.append(el("tr", {}, name, ta, tb));
    }
  }

  function renderProbs() {
    const box = $("probs");
    const a = state.slots.a.result, b = state.slots.b.result;
    box.textContent = "";
    if (!state.model || !state.model.loaded || (!a && !b)) {
      box.append(el("p", { class: "empty", text: "Nothing analysed yet." }));
      return;
    }
    for (const c of state.model.classes) {
      const bars = el("div", { class: "bars" });
      const nums = el("div", { class: "nums" });
      for (const [k, r] of [["a", a], ["b", b]]) {
        if (!r) continue;
        const p = r.probs[c];
        bars.append(el("div", { class: `bar ${k}`, role: "img", "aria-label": `${k.toUpperCase()}: ${pct(p)}` }, el("i", { style: `width:${(p * 100).toFixed(1)}%` })));
        nums.append(el("div", { text: `${k.toUpperCase()} ${pct(p)}` }));
      }
      box.append(el("div", { class: "prow" }, el("span", { text: cap(c) }), bars, nums));
    }
  }

  function renderLatency() {
    const run = $("this-run");
    run.textContent = "";
    const parts = [];
    for (const k of ["a", "b"]) {
      const r = state.slots[k].result;
      if (!r) continue;
      const t = r.timings;
      if (r.engine === "device") {
        parts.push(`${k.toUpperCase()}: on your device, model ${fmtMs(t.model_ms)}, features ${fmtMs(t.features_ms)}; no request sent`);
        continue;
      }
      parts.push(
        `${k.toUpperCase()}: server ${fmtMs(t.server_ms)} (features ${fmtMs(t.features_ms)}, Praat ${fmtMs(t.prosody_ms)}, tract ${fmtMs(t.tract_ms)})` +
          `; round trip ${fmtMs(r.e2e_ms)}` + (t.cold ? "; first request after start, excluded from the averages below" : "")
      );
    }
    if (!parts.length) { run.textContent = "No run yet."; return; }
    parts.forEach((p, i) => { if (i) run.append(el("br")); run.append(p); });

    const st = state.stats;
    const body = $("all-body");
    body.textContent = "";
    if (!st) {
      $("all-note").textContent = "Statistics are unavailable right now. Analysis still works.";
      return;
    }
    const rows = [["Model (tract)", st.tract], ["Server total", st.server], ["Round trip, your browser", st.round_trip]];
    for (const [name, s] of rows) {
      body.append(el("tr", {}, el("td", { text: name }),
        el("td", { text: s && s.n ? fmtMs(s.mean) : "n/a" }),
        el("td", { text: s && s.n ? fmtMs(s.p50) : "n/a" }),
        el("td", { text: s && s.n ? fmtMs(s.p95) : "n/a" })));
    }
    const since = st.since ? new Date(st.since).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "launch";
    const hw = state.hardware ? ` Server: ${state.hardware}, tract ${state.tractVersion || ""}.` : "";
    $("all-note").textContent = `n = ${st.n} analyses since ${since} (${st.n_warm} timed warm, ${st.n_cold} first-request ${st.n_cold === 1 ? "start" : "starts"} left out).${hw} Round trip includes your network. ${storageNote()}`;
  }

  function storageNote() {
    const sg = state.storage;
    if (!sg) return "";
    if (sg.note) return sg.note;
    return sg.persistent ? "Counts are kept in a database and survive restarts." : "Counts are kept on this server and reset when it restarts.";
  }

  function renderAll() {
    renderSlot("a"); renderSlot("b");
    renderMeasures(); renderProbs(); renderLatency();
  }

  function renderBanner() {
    const b = $("banner");
    const m = state.model;
    if (!m || !m.loaded) {
      b.hidden = false;
      b.textContent = "No model is loaded on this server yet, so analysis is unavailable.";
    } else if (m.data_source !== "real") {
      b.hidden = false;
      b.textContent = "Pipeline test build: this model was trained on synthetic tones, not speech. Its labels mean nothing.";
    } else b.hidden = true;
  }

  function renderLimits() {
    const box = $("limits-body");
    box.textContent = "";
    const m = state.model;
    if (!m || !m.loaded) return;
    const t = m.test || {};
    const ci = t.accuracy_ci95 ? ` (95% interval ${pct(t.accuracy_ci95[0])} to ${pct(t.accuracy_ci95[1])}, resampled by speaker)` : "";
    const items = [];
    if (t.accuracy != null) items.push(`On ${t.n_clips} clips from ${t.n_speakers} speakers it never heard in training, it picks the right label ${pct(t.accuracy)} of the time${ci}. Chance is ${pct(t.chance)}.`);
    if (t.coverage_at_threshold != null) items.push(`It answers on ${pct(t.coverage_at_threshold)} of clips and abstains on the rest. When it answers, it is right ${pct(t.accuracy_when_answering)} of the time.`);
    items.push(`Trained on ${m.datasets.join(" and ")}: scripted sentences acted by adult speakers, mostly North American English. Acted emotion is not spontaneous emotion, and it will fail on voices, accents, ages and recording conditions the corpora do not cover.`);
    items.push("The labels describe how a sentence was delivered, not what a person feels. It should not be used to judge anyone.");
    items.push("The author's production voice work was machine-directed speech. That is a different register from conversation, and this demo does not claim otherwise.");
    items.push(`${m.n_params.toLocaleString()} parameters. Model licence is non-commercial because RAVDESS is CC BY-NC-SA 4.0.`);
    const ul = el("ul");
    items.forEach((x) => ul.append(el("li", { text: x })));
    box.append(ul, el("p", {}, el("a", { href: "/model-card", text: "Full model card" }), " with per-group results and cross-corpus tests."));
  }

  // ---------- playback ----------
  function stopPlayback(key) {
    const s = state.slots[key];
    if (s.playing) { try { s.playing.onended = null; s.playing.stop(); } catch (_) {} s.playing = null; }
  }

  function playSlot(key) {
    const s = state.slots[key];
    stopPlayback(key);
    if (!s.pcm) return Promise.resolve();
    const ctx = audioCtx();
    const buf = ctx.createBuffer(1, s.pcm.length, s.sr);
    buf.copyToChannel(s.pcm, 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    s.playing = src;
    renderSlot(key);
    return new Promise((res) => {
      src.onended = () => { if (s.playing === src) s.playing = null; renderSlot(key); res(); };
      src.start();
    });
  }

  async function runSlot(key, source) {
    const s = state.slots[key];
    const pcm = s.pcm, sr = s.sr;
    s.run = "pending"; s.result = null;
    renderSlot(key);
    try {
      let r;
      if (state.engine === "device") {
        if (!(await loadDevice())) {
          state.engine = "server"; renderEngine();
          r = await analyse(pcm, sr, source); // fall back, and the panel says where it ran
        } else r = await analyseOnDevice(pcm, sr);
      } else r = await analyse(pcm, sr, source);
      if (s.pcm !== pcm) return; // replaced while waiting
      s.result = r; s.run = null;
      if (r.engine !== "device" && r.stats !== undefined) state.stats = r.stats;
      showError("");
    } catch (e) {
      if (s.pcm !== pcm) return;
      s.run = { error: e.message };
      showError(e.message);
    }
    renderAll();
  }

  // ---------- sample pairs ----------
  async function decodeClip(url) {
    const ab = await (await fetch(url)).arrayBuffer();
    const buf = await audioCtx().decodeAudioData(ab);
    return { pcm: buf.getChannelData(0).slice(), sr: buf.sampleRate };
  }

  async function choosePair(pair, button) {
    const token = ++state.pairToken;
    document.querySelectorAll("#pairlist button").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
    stopPlayback("a"); stopPlayback("b");
    showError("");
    $("own-sentence-wrap").hidden = true;
    const sent = $("sentence");
    sent.classList.remove("plain");
    sent.textContent = pair.sentence;
    try {
      const [ca, cb] = await Promise.all([decodeClip(`/static/samples/${pair.a.file}`), decodeClip(`/static/samples/${pair.b.file}`)]);
      if (token !== state.pairToken) return;
      for (const [k, c, side] of [["a", ca, pair.a], ["b", cb, pair.b]]) {
        const s = state.slots[k];
        s.pcm = c.pcm; s.sr = c.sr; s.source = "sample";
        s.label = side.caption || `${cap(side.emotion)}, speaker ${side.speaker}`;
        s.result = null; s.run = null;
      }
      renderAll();
      // analysis starts now and runs while the clips play
      const jobs = [runSlot("a", "sample"), runSlot("b", "sample")];
      await playSlot("a");
      if (token !== state.pairToken) return;
      await playSlot("b");
      await Promise.all(jobs);
    } catch (e) {
      showError(`Could not load that pair: ${e.message}`);
    }
  }

  async function loadManifest() {
    try {
      const m = await getJSON("/static/samples/manifest.json");
      if (!m.pairs || !m.pairs.length) throw new Error("empty");
      state.manifest = m;
    } catch (_) {
      state.manifest = null;
      return;
    }
    $("pairs").hidden = false;
    const list = $("pairlist");
    list.textContent = "";
    for (const p of state.manifest.pairs) {
      const b = el("button", { type: "button", "aria-pressed": "false", text: p.title || p.sentence });
      b.addEventListener("click", () => choosePair(p, b));
      list.append(b);
    }
    $("pairs-note").textContent = state.manifest.attribution || "";
  }

  // ---------- recording ----------
  async function toggleRecord(key) {
    if (state.recording) {
      const was = state.recording.key;
      await finishRecording();
      if (was === key) return;
    }
    showError("");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError("This browser cannot record audio. Open the page over HTTPS in a current browser.");
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
    } catch (e) {
      showError("Microphone access was blocked. Allow it in the browser address bar and try again, or use the sample pairs.");
      return;
    }
    const ctx = audioCtx();
    await ctx.audioWorklet.addModule("/static/worklet.js");
    const srcNode = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, "capture");
    const chunks = [];
    let n = 0;
    node.port.onmessage = (ev) => {
      chunks.push(ev.data); n += ev.data.length;
      if (n / ctx.sampleRate >= MAX_REC_S) finishRecording();
    };
    srcNode.connect(node);
    state.recording = { key, stream, srcNode, node, chunks, sr: ctx.sampleRate };
    stopPlayback("a"); stopPlayback("b");
    $(`rec-${key}`).setAttribute("aria-pressed", "true");
    $(`rec-${key}`).textContent = "Stop";
    state.slots[key].label = "Recording…";
    $("own-sentence-wrap").hidden = false;
    const sent = $("sentence");
    const typed = $("own-sentence").value.trim();
    sent.classList.toggle("plain", !typed);
    sent.textContent = typed || "Say the same sentence in each slot, delivered differently.";
    state.pairToken++;
    document.querySelectorAll("#pairlist button").forEach((b) => b.setAttribute("aria-pressed", "false"));
    renderSlot(key);
  }

  async function finishRecording() {
    const r = state.recording;
    if (!r) return;
    state.recording = null;
    r.node.port.onmessage = null;
    r.srcNode.disconnect(); r.node.disconnect();
    r.stream.getTracks().forEach((t) => t.stop());
    const total = r.chunks.reduce((a, c) => a + c.length, 0);
    const pcm = new Float32Array(total);
    let o = 0;
    for (const c of r.chunks) { pcm.set(c, o); o += c.length; }
    const btn = $(`rec-${r.key}`);
    btn.setAttribute("aria-pressed", "false");
    btn.textContent = "Record";
    const s = state.slots[r.key];
    s.pcm = pcm; s.sr = r.sr; s.source = "live";
    s.label = "Your recording";
    s.result = null; s.run = null;
    renderAll();
    await runSlot(r.key, "live");
  }

  async function loadArchitecture() {
    try {
      const a = await getJSON("/static/architecture.json");
      const body = $("arch-body");
      body.textContent = "";
      for (const r of a.layers) {
        body.append(el("tr", {}, el("td", { text: `${r.layer} ${r.type}` }), el("td", { text: r.output.join(" \u00d7 ") }),
          el("td", { text: r.params.toLocaleString() }), el("td", { text: r.ms_p50.toFixed(2) })));
      }
      $("arch-note").textContent = `${a.n_params.toLocaleString()} parameters. ${a.timing_note}. The live serving numbers are in the Latency section.`;
    } catch (_) {
      $("build").hidden = true;
    }
  }

  // ---------- init ----------
  function wire() {
    document.querySelectorAll('#engine input[name="engine"]').forEach((i) =>
      i.addEventListener("change", () => { if (i.checked) setEngine(i.value); }));
    for (const k of ["a", "b"]) {
      $(`play-${k}`).addEventListener("click", () => (state.slots[k].playing ? (stopPlayback(k), renderSlot(k)) : playSlot(k)));
      $(`rec-${k}`).addEventListener("click", () => toggleRecord(k));
    }
    $("own-sentence").addEventListener("input", (e) => {
      const sent = $("sentence");
      const v = e.target.value.trim();
      sent.classList.toggle("plain", !v);
      sent.textContent = v || "Say the same sentence in each slot, delivered differently.";
    });
  }

  async function init() {
    wire();
    renderAll();
    try {
      state.model = await getJSON("/api/model");
    } catch (_) {
      state.model = { loaded: false };
    }
    renderBanner(); renderLimits(); probeDevice();
    try {
      const s = await getJSON("/api/stats");
      state.stats = s.stats; state.storage = s.storage; state.hardware = s.hardware; state.tractVersion = s.tract_version;
    } catch (_) { /* footer stays empty */ }
    loadArchitecture();
    await loadManifest();
    if (!state.manifest) {
      $("sentence").classList.add("plain");
      $("sentence").textContent = "Record the same sentence twice, in two different ways, using the Record buttons.";
      $("own-sentence-wrap").hidden = false;
    }
    renderAll();
  }
  init();
})();
