"use strict";
/* Beyond the transcript: front end.
 * In server mode, audio leaves the browser only as the raw PCM body of one POST to /api/analyze.
 * In on-device mode it does not leave at all. On-device mode needs files under /static/ondevice/
 * (see docs/ON_DEVICE.md for the contract); without them the option is shown but disabled.
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
      a: { pcm: null, sr: 0, result: null, source: null, label: "Empty", playing: null, run: null, truth: null },
      b: { pcm: null, sr: 0, result: null, source: null, label: "Empty", playing: null, run: null, truth: null },
    },
    pair: null, // the sample pair on screen, if any
    board: null, // scoreboard state
    recording: null,
    stats: null,
    storage: null,
    pairToken: 0,
    mode: "samples", // "samples" | "record"
    util: 0.25, // assumed server utilisation for the cost estimate
    arch: null,
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
  const DS = { cremad: "CREMA-D", ravdess: "RAVDESS" };
  const dsName = (k) => DS[k] || k;
  const fmtMs = (v) => (v == null ? "n/a" : v >= 100 ? `${Math.round(v)} ms` : `${v.toFixed(1)} ms`);
  const pct = (v) => (Number.isFinite(v) ? `${Math.round(v * 100)}%` : "n/a");
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
        const m = d.info.model || "model.onnx";
        d.eng = await mod.load({ modelUrl: m.startsWith("/") ? m : `${DEVICE_DIR}${m}` });
        d.status = "ready";
      } catch (e) {
        console.error("On-device load failed:", e);
        d.status = "error";
        const why = e && e.message ? ` Reason: ${String(e.message).slice(0, 200)}.` : "";
        d.error = `The on-device model could not be loaded in this browser.${why}`;
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
      was_cropped: !!out.cropped,
      audio_seconds: f32.length / sr,
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

  function deviceSize() {
    const kb = state.model && state.model.ondevice_kb;
    return kb ? ` (about ${(kb / 1024).toFixed(1)} MB, mostly the WebAssembly runtime)` : "";
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
      para(el("strong", { text: "Your audio stays in this browser tab. " }), "The model runs on your own device as WebAssembly (ONNX Runtime Web), and nothing is sent to the server for the analysis.");
      para(`What you give up: a one-time download${deviceSize()}, speed that depends on your device, and the Praat measurements, which are off because Praat only runs on the server. Timings from this mode are not added to the shared statistics below.`);
    } else if (state.engine === "device") {
      para(el("strong", { text: "Your audio stays in this browser tab. " }), `The model and its runtime are downloading to your device${deviceSize()}. Nothing is sent to the server for the analysis.`);
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
    renderOvals(key);
    const v = $(`verdict-${key}`);
    v.textContent = "";
    if (s.run === "pending") { v.textContent = "Analysing…"; return; }
    if (s.run && s.run.error) { v.textContent = s.run.error; return; }
    const r = s.result;
    if (!r) return;
    v.append(outcomeView(r, s.truth));
  }

  const OUTCOME = { right: ["\u2713", "Correct"], wrong: ["\u2715", "Incorrect"], held: ["\u2013", "Unsure"] };

  function rankedProbs(r) { return Object.entries(r.probs).sort((x, y) => y[1] - x[1]); }

  function outcomeOf(r, truth) {
    if (!truth) return "plain";
    return r.abstained ? "held" : r.label === truth ? "right" : "wrong";
  }

  // One result as a verdict line, a confidence meter with the answer-or-abstain line, and a sentence of detail.
  function outcomeView(r, truth) {
    const m = state.model, thr = m.abstain_threshold;
    const kind = outcomeOf(r, truth);
    const [topLabel, topP] = rankedProbs(r)[0];
    const wrap = el("div", { class: `outcome ${kind}` });
    const head = el("div", { class: "ohead" });
    if (kind !== "plain") head.append(el("span", { class: "chip" }, el("span", { "aria-hidden": "true", text: OUTCOME[kind][0] }), ` ${OUTCOME[kind][1]}`));
    if (r.abstained) head.append(el("strong", { text: "Unsure" }));
    else head.append(el("strong", { text: cap(r.label) }), el("span", { class: "conf", text: ` at ${pct(r.confidence)} confidence` }));
    wrap.append(head);

    const meter = el("div", { class: "meter", role: "img", "aria-label": `The model's top guess, ${topLabel}, has ${pct(topP)} confidence. It answers only at ${pct(thr)} or more.` },
      el("i", { style: `width:${(topP * 100).toFixed(1)}%` }),
      el("b", { class: "tick", style: `left:${(thr * 100).toFixed(1)}%` }));
    wrap.append(el("div", { class: "meterwrap" }, meter, ...(Number.isFinite(thr) ? [el("span", { class: "ticklabel", style: `left:${(thr * 100).toFixed(1)}%`, text: `Confidence threshold = ${pct(thr)}` })] : [])));

    const bits = [];
    if (r.abstained) {
      bits.push(`Its top guess, ${topLabel} at ${pct(topP)}, is below the confidence threshold of ${pct(thr)}, so it does not answer.`);
      if (truth) bits.push(`That guess would have been ${topLabel === truth ? "correct" : "incorrect"}.`);
    } else if (truth && kind === "right") {
      bits.push(`Acted as ${truth}.`);
    } else if (truth) {
      bits.push(`Acted as ${truth}. The model gave ${truth} ${pct(r.probs[truth] || 0)}.`);
    }
    if (bits.length) wrap.append(el("p", { class: "odetail", text: bits.join(" ") }));
    if (r.was_cropped) wrap.append(el("p", { class: "odetail", text: "Clip is longer than 3 s, so the model heard the loudest 3 s and the pitch line is hidden." }));
    return wrap;
  }

  function fmtMeasure(key, v) {
    if (v == null) return "n/a";
    if (key === "voiced_frac" || key === "pause_frac") return pct(v);
    if (key === "f0_median_hz") return `${Math.round(v)}`;
    return v.toFixed(1);
  }

  function renderMeasures() {
    const desc = state.model && state.model.prosody_descriptions;
    const res = { a: state.slots.a.result, b: state.slots.b.result };
    const boxes = { a: $("measures-a"), b: $("measures-b") };
    boxes.a.textContent = ""; boxes.b.textContent = "";
    const say = (t) => { boxes.a.append(el("p", { class: "empty", text: t })); };
    if (!desc || (!res.a && !res.b)) { say("Nothing measured yet."); return; }
    const pa = res.a && res.a.prosody, pb = res.b && res.b.prosody;
    if (!pa && !pb) { say("Praat runs on the server, so its measurements are off in on-device mode. Switch to the server to see them."); return; }
    for (const [key, [label, unit, why]] of Object.entries(desc)) {
      if (key === "duration_s") continue;
      for (const k of ["a", "b"]) {
        const r = res[k], p = k === "a" ? pa : pb;
        if (!r) continue;
        const name = unit ? `${label} (${unit})` : label;
        const dd = el("dd", { text: p ? fmtMeasure(key, p[key]) : "n/a" });
        if (k === "b" && pa && pb && pa[key] != null && pb[key] != null) {
          const d = pb[key] - pa[key];
          const share = key === "voiced_frac" || key === "pause_frac";
          const txt = share ? `${d >= 0 ? "+" : "−"}${Math.round(Math.abs(d) * 100)} pts` : `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(key === "f0_median_hz" ? 0 : 1)}`;
          dd.append(el("span", { class: "diff", text: txt }));
        }
        boxes[k].append(el("div", { class: "mrow", title: why }, el("dt", { text: name }), dd));
      }
    }
  }

  function renderProbs() {
    const res = { a: state.slots.a.result, b: state.slots.b.result };
    const ready = state.model && state.model.loaded;
    for (const k of ["a", "b"]) {
      const box = $(`probs-${k}`);
      box.textContent = "";
      if (!ready || !res[k]) { if (k === "a" && !res.a && !res.b) box.append(el("p", { class: "empty", text: "Nothing analysed yet." })); continue; }
      const top = Object.entries(res[k].probs).sort((x, y) => y[1] - x[1])[0][0];
      for (const c of state.model.classes) {
        const p = res[k].probs[c];
        const label = el("span", { text: cap(c) });
        if (c === top && !res[k].abstained) label.style.fontWeight = "650";
        box.append(el("div", { class: "prow" }, label,
          el("div", { class: `bar ${k}`, role: "img", "aria-label": `${cap(c)}: ${pct(p)}` }, el("i", { style: `width:${(p * 100).toFixed(1)}%` })),
          el("span", { class: "nums", text: pct(p) })));
      }
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
        `${k.toUpperCase()}: server ${fmtMs(t.server_ms)} (features ${fmtMs(t.features_ms)}, Praat ${fmtMs(t.prosody_ms)}, model ${fmtMs(t.tract_ms)})` +
          `; round trip ${fmtMs(r.e2e_ms)}` + (t.cold ? "; first request after start, excluded from the averages below" : "")
      );
    }
    const scored = state.board && state.board.rows.filter((r) => r.result);
    if (!parts.length && scored && scored.length) {
      const med = (xs) => { const v = xs.filter((x) => x != null).sort((p, q) => p - q); return v.length ? v[Math.floor(v.length / 2)] : null; };
      const dev = scored.filter((r) => r.result.engine === "device");
      if (dev.length) {
        parts.push(`Scored ${dev.length} clips on your device. Median ${fmtMs(med(dev.map((r) => r.result.timings.total_ms)))} per clip (model ${fmtMs(med(dev.map((r) => r.result.timings.model_ms)))}, features ${fmtMs(med(dev.map((r) => r.result.timings.features_ms)))}); no requests sent`);
      } else {
        parts.push(`Scored ${scored.length} clips on the server. Median per clip: server ${fmtMs(med(scored.map((r) => r.result.timings.server_ms)))} (model ${fmtMs(med(scored.map((r) => r.result.timings.tract_ms)))}), round trip ${fmtMs(med(scored.map((r) => r.result.e2e_ms)))}`);
      }
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
    const rows = [["Model inference", st.tract], ["Server total, features and Praat included", st.server], ["Round trip, including your network", st.round_trip]];
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
    renderMeasures(); renderProbs(); renderLatency(); renderCost(); renderModelCard();
  }

  function renderBanner() {
    const b = $("banner");
    const m = state.model;
    if (state.modelError) {
      b.hidden = false;
      b.textContent = "The model details did not load, so the confidence line, group chart and measurement notes are missing. Reload the page; analysis itself still works.";
    } else if (!m || !m.loaded) {
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
    const limits = [];
    if (t.accuracy != null) limits.push(`On ${t.n_clips} clips from ${t.n_speakers} speakers it never heard in training, it names the acted delivery ${pct(t.accuracy)} of the time${ci}. Chance is ${pct(t.chance)}.`);
    if (t.coverage_at_threshold != null) limits.push(`It answers on ${pct(t.coverage_at_threshold)} of clips and abstains on the rest. When it answers, it is correct ${pct(t.accuracy_when_answering)} of the time.`);
    const cc = t.cross_corpus && Object.entries(t.cross_corpus).filter(([, v]) => v && v.accuracy != null);
    if (cc && cc.length) limits.push("Trained on one corpus and tested on the other, accuracy was " + cc.map(([k, v]) => `${pct(v.accuracy)} (${k.split("_to_").map(dsName).join(" to ")})`).join(" and ") + ". That is the better guide to how it behaves on voices and recording conditions it has not met.");
    limits.push(`Trained on ${m.datasets.map(dsName).join(" and ")}: scripted sentences acted by adult speakers, mostly North American English. Acted emotion is not spontaneous emotion, and it will fail on voices, accents, ages and recording conditions the corpora do not cover.`);
    limits.push("The labels describe how a sentence was delivered, not what a person feels. It should not be used to try and judge real emotions.");
    const next = [
      "Evaluate on spontaneous, noisy, multi-speaker speech, with a breakdown by accent, age and recording device. The corpora here carry no accent labels, so that gap cannot be measured with them.",
      "Compare against a large pretrained speech encoder with a small head on top, to learn how much of the remaining error is a data limit and how much is a model limit.",
      "Flag the clips where the Praat measurements and the model disagree, and study those cases. Today they sit side by side and nothing compares them.",
      "Score overlapping one-second windows over a live stream and smooth over time, instead of one clip at a time.",
      "Check calibration group by group, not only overall, and re-fit the abstain threshold on conversational data.",
    ];
    const h = (t0) => el("h3", { class: "lhead", text: t0 });
    const list = (arr) => { const ul = el("ul"); arr.forEach((x) => ul.append(el("li", { text: x }))); return ul; };
    box.append(h("Limitations"), list(limits), h("Next steps"), list(next),
      el("p", {}, el("a", { href: "/model-card", text: "Full model card" }), " with per-group results and cross-corpus tests."));
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

  async function analyseWithEngine(pcm, sr, source) {
    if (state.engine === "device") {
      if (await loadDevice()) return analyseOnDevice(pcm, sr);
      state.engine = "server"; renderEngine(); // fall back, and the panel says where it ran
    }
    return analyse(pcm, sr, source);
  }

  async function runSlot(key, source) {
    const s = state.slots[key];
    const pcm = s.pcm, sr = s.sr;
    s.run = "pending"; s.result = null;
    renderSlot(key);
    try {
      const r = await analyseWithEngine(pcm, sr, source);
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
    const sent = $("sentence");
    sent.classList.remove("plain");
    sent.textContent = pair.sentence;
    state.pair = pair;
    try {
      const [ca, cb] = await Promise.all([decodeClip(`/static/samples/${pair.a.file}`), decodeClip(`/static/samples/${pair.b.file}`)]);
      if (token !== state.pairToken) return;
      for (const [k, c, side] of [["a", ca, pair.a], ["b", cb, pair.b]]) {
        const s = state.slots[k];
        s.pcm = c.pcm; s.sr = c.sr; s.source = "sample";
        s.label = side.caption || `${cap(side.emotion)}, speaker ${side.speaker}`;
        s.truth = side.emotion;
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
    state.slots[key].truth = null; state.slots[key].result = null; state.pair = null;
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
    s.truth = null; state.pair = null;
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
      state.arch = a;
      renderModelCard();
      $("arch-note").textContent = `${a.n_params.toLocaleString()} parameters. ${a.timing_note}. The live inference numbers are in the Latency section.`;
    } catch (_) {
      $("arch-table").hidden = true;
    }
  }

  // ---------- scoreboard over every sample clip ----------
  async function runScoreboard() {
    if (!state.manifest || (state.board && state.board.running)) return;
    const pairs = state.manifest.pairs, total = pairs.length * 2;
    const board = (state.board = { running: true, rows: [] });
    const btn = $("score-run");
    btn.disabled = true;
    stopPlayback("a"); stopPlayback("b");
    for (const p of pairs) {
      for (const k of ["a", "b"]) {
        const side = p[k];
        $("score-status").textContent = `Scoring clip ${board.rows.length + 1} of ${total}…`;
        try {
          const c = await decodeClip(`/static/samples/${side.file}`);
          const r = await analyseWithEngine(c.pcm, c.sr, "sample");
          if (r.engine !== "device" && r.stats !== undefined) state.stats = r.stats;
          board.rows.push({ pair: p, side, key: k, result: r });
        } catch (e) {
          board.rows.push({ pair: p, side, key: k, error: e.message });
          if (/Slow down/.test(e.message)) break;
        }
        renderBoard();
      }
    }
    board.running = false;
    btn.disabled = false;
    btn.textContent = "Score all clips again";
    $("score-status").textContent = "";
    try {
      const st = await getJSON("/api/stats");
      state.stats = st.stats; state.storage = st.storage || state.storage;
    } catch (_) { /* keep the numbers already on screen */ }
    renderBoard();
    renderAll();
  }

  function renderBoard() {
    const box = $("scoreboard"), bd = state.board;
    box.textContent = "";
    if (!bd || !bd.rows.length) { box.hidden = true; return; }
    box.hidden = false;
    const rows = bd.rows.filter((r) => r.result);
    const tally = { right: 0, wrong: 0, held: 0 };
    for (const r of rows) tally[outcomeOf(r.result, r.side.emotion)]++;
    const answered = tally.right + tally.wrong;
    const t = (state.model && state.model.test) || {};
    box.append(el("p", { class: "tally" },
      el("strong", { text: `${tally.right} correct, ${tally.wrong} incorrect, ${tally.held} unsure` }),
      ` of ${rows.length} clips.` + (answered ? ` Correct on ${tally.right} of the ${answered} it answered.` : "")));

    // strip plot: confidence on the x axis, one lane per corpus
    const thr = state.model.abstain_threshold;
    const corpora = [...new Set(rows.map((r) => r.side.corpus || "Clips"))];
    const strip = el("div", { class: "strip", role: "img", "aria-label": "Each clip as a mark placed by the model's confidence in its top guess, grouped by corpus. Marks left of the confidence threshold line were marked unsure." });
    for (const c of corpora) {
      const lane = el("div", { class: "lane" }, el("span", { class: "lanename", text: c }));
      const track = el("div", { class: "lanetrack" }, el("b", { class: "tick", style: `left:${(thr * 100).toFixed(1)}%` }));
      rows.filter((r) => (r.side.corpus || "Clips") === c).forEach((r, i) => {
        const kind = outcomeOf(r.result, r.side.emotion);
        const topP = rankedProbs(r.result)[0][1];
        const mark = el("span", { class: `dot ${kind}`, style: `left:${(topP * 100).toFixed(1)}%;top:${i % 2 ? 24 : 4}px`, title: `${r.pair.sentence} (${r.key.toUpperCase()}): acted ${r.side.emotion}, ${r.result.abstained ? "unsure" : "said " + r.result.label}, ${pct(topP)}` }, el("span", { "aria-hidden": "true", text: OUTCOME[kind][0] }));
        track.append(mark);
      });
      lane.append(track);
      strip.append(lane);
    }
    strip.append(el("div", { class: "stripaxis" }, el("span"), el("div", { class: "axisrow" }, el("span", { text: "0%" }), el("span", { text: "Model confidence in its top guess" }), el("span", { text: "100%" }))));
    box.append(strip);
    box.append(el("p", { class: "note", text: `Marks left of the line were marked unsure. ${rows.length} clips is too few to measure accuracy; the held-out test${t.n_clips ? ` (${t.n_clips} clips from ${t.n_speakers} speakers)` : ""} does that. This shows what correct, incorrect and unsure look like, and how confidence relates to them.` }));

    const det = el("details", { class: "clipwise" }, el("summary", { text: "Clip by clip" }));
    const tb = el("tbody");
    for (const r of bd.rows) {
      const head = `${r.pair.sentence} (${r.key.toUpperCase()}, ${r.side.corpus || ""})`;
      if (!r.result) { tb.append(el("tr", {}, el("td", { text: head }), el("td", { colspan: "4", class: "empty", text: r.error || "Could not analyse" }))); continue; }
      const kind = outcomeOf(r.result, r.side.emotion), [topLabel, topP] = rankedProbs(r.result)[0];
      tb.append(el("tr", {}, el("td", { text: head }), el("td", { text: cap(r.side.emotion) }),
        el("td", { text: r.result.abstained ? `Unsure (${topLabel})` : cap(r.result.label) }),
        el("td", { text: pct(topP) }),
        el("td", {}, el("span", { class: `chip ${kind}` }, el("span", { "aria-hidden": "true", text: OUTCOME[kind][0] }), ` ${OUTCOME[kind][1]}`))));
    }
    det.append(el("table", { class: "clip-table" }, el("thead", {}, el("tr", {}, ...["Clip", "Acted as", "Model said", "Confidence", "Result"].map((h) => el("th", { scope: "col", text: h })))), tb));
    box.append(det);
  }

  // ---------- accuracy by group ----------
  const GROUP_NAMES = { dataset: "Corpus", sex: "Sex", age_bucket: "Age", race: "Race", ethnicity: "Ethnicity" };

  const gname = (grouping, name) => (grouping === "dataset" ? dsName(name) : cap(String(name)));

  function renderGroups() {
    const sec = $("groups"), body = $("groups-body");
    const t = state.model && state.model.test, g = t && t.groups;
    body.textContent = "";
    const names = g ? Object.keys(GROUP_NAMES).filter((k) => g[k] && Object.keys(g[k]).length) : [];
    if (!names.length) { sec.hidden = true; return; }
    sec.hidden = false;

    // plain-language summary, computed from the numbers
    const gaps = [];
    for (const k of names) {
      const ent = Object.entries(g[k]).filter(([, v]) => v.n_speakers >= 2 && v.accuracy != null);
      if (ent.length < 2) continue;
      ent.sort((x, y) => y[1].accuracy - x[1].accuracy);
      const [hiN, hi] = ent[0], [loN, lo] = ent[ent.length - 1];
      const ci = (v) => (v.ci95 && v.ci95[0] != null && v.ci95[1] != null ? v.ci95 : null);
      const overlap = ci(hi) && ci(lo) ? ci(lo)[1] >= ci(hi)[0] : null;
      gaps.push(`${GROUP_NAMES[k]}: ${gname(k, hiN)} ${pct(hi.accuracy)} against ${gname(k, loN)} ${pct(lo.accuracy)}, a gap of ${Math.round((hi.accuracy - lo.accuracy) * 100)} points. ` +
        (overlap === null ? "The interval is unavailable." : overlap ? "The intervals overlap, so this is not evidence of a difference." : "The intervals do not overlap, so this gap is worth a closer look."));
    }
    if (gaps.length) {
      const ul = el("ul", { class: "gaplist" });
      gaps.forEach((x) => ul.append(el("li", { text: x })));
      body.append(el("p", { class: "note", text: "Largest gap within each grouping:" }), ul);
    }

    const scale = el("div", { class: "gscale" }, el("span"), el("div", { class: "axisrow" }, el("span", { text: "0%" }), el("span", { text: "accuracy" }), el("span", { text: "100%" })), el("span"));
    const table = el("div", { class: "grows" });
    for (const k of names) {
      table.append(el("h3", { class: "ghead", text: GROUP_NAMES[k] }));
      for (const [name, v] of Object.entries(g[k])) {
        const ci = v.ci95 && v.ci95[0] != null && v.ci95[1] != null ? v.ci95 : null;
        const track = el("div", { class: "gtrack", role: "img", "aria-label": `${name}: ${pct(v.accuracy)}${ci ? `, interval ${pct(ci[0])} to ${pct(ci[1])}` : ""}` });
        if (t.chance != null) track.append(el("b", { class: "gchance", style: `left:${(t.chance * 100).toFixed(1)}%` }));
        if (ci) track.append(el("i", { class: "gci", style: `left:${(ci[0] * 100).toFixed(1)}%;width:${((ci[1] - ci[0]) * 100).toFixed(1)}%` }));
        track.append(el("span", { class: "gdot", style: `left:${(v.accuracy * 100).toFixed(1)}%` }));
        const few = v.n_speakers < 3 ? ", few speakers" : "";
        table.append(el("div", { class: "grow" }, el("span", { class: "gname", text: gname(k, name) }), track,
          el("span", { class: "gnum", text: `${pct(v.accuracy)}: ${v.n_speakers} ${v.n_speakers === 1 ? "speaker" : "speakers"}, ${v.n_clips} clips${few}` })));
      }
    }
    body.append(scale, table);
  }

  // ---------- what an analysis costs ----------
  // Assumption, stated on the page: a small always-on instance at this list price, running flat out.
  const PRICE_PER_HOUR = 0.0725; // USD, AWS c7g.large (2 vCPU, 4 GB), on-demand, us-east-1, checked 7 Oct 2026
  const usd = (v) => (v >= 1 ? `$${v.toFixed(2)}` : v >= 0.01 ? `$${v.toFixed(3)}` : `$${v.toPrecision(2)}`);
  let costRefs = null;

  function costFigures() {
    const sv = state.stats && state.stats.server;
    if (!sv || !sv.n) return null;
    const secs = sv.mean / 1000;
    const perAnalysis = (PRICE_PER_HOUR / 3600) * secs / state.util;
    return { secs, perAnalysis };
  }

  function updateCost() {
    const f = costFigures();
    if (!f || !costRefs) return;
    costRefs.out.textContent = `${Math.round(state.util * 100)}%`;
    costRefs.k.textContent = usd(f.perAnalysis * 1e3);
    costRefs.m.textContent = usd(f.perAnalysis * 1e6);
  }

  function renderCost() {
    const wrap = $("cost-wrap"), body = $("cost-body");
    const f = costFigures();
    if (!f) { wrap.hidden = true; return; }
    wrap.hidden = false;
    body.textContent = "";
    const sv = state.stats.server;
    const slider = el("input", { type: "range", id: "util", min: "5", max: "100", step: "5", value: String(Math.round(state.util * 100)), "aria-describedby": "util-out" });
    const out = el("output", { id: "util-out", for: "util" });
    slider.addEventListener("input", () => { state.util = Number(slider.value) / 100; updateCost(); });
    const kCell = el("td"), mCell = el("td");
    const kb = state.model && state.model.ondevice_kb;
    body.append(
      el("p", {}, `Server time per analysis averages ${fmtMs(sv.mean)}, features and Praat included. Priced on one AWS instance (c7g.large, 2 vCPU, 4 GB) at $${PRICE_PER_HOUR} an hour, on demand in US East (N. Virginia), running one analysis at a time.`),
      el("div", { class: "util" }, el("label", { for: "util", text: "How busy the instance is, on average: " }), slider, " ", out),
      el("table", { id: "cost-table" },
        el("thead", {}, el("tr", {}, el("th", { scope: "col", text: "Where it runs" }), el("th", { scope: "col", text: "Per 1,000 analyses" }), el("th", { scope: "col", text: "Per million" }))),
        el("tbody", {},
          el("tr", {}, el("td", { text: "On the server (AWS)" }), kCell, mCell),
          el("tr", {}, el("td", { text: "On your device (no server compute)" }), el("td", { text: "$0" }), el("td", { text: "$0" })))),
      el("p", { class: "note", text: `On your device the host pays for no compute, only for serving a one-time download${kb ? ` of about ${(kb / 1024).toFixed(0)} MB` : ""}. The server time was measured on this site's host, not on a c7g.large, so read the estimate as an order of magnitude. It leaves out bandwidth, storage, monitoring and the time it took to build.` }));
    costRefs = { out, k: kCell, m: mCell };
    updateCost();
  }

  // ---------- model card ----------
  function renderModelCard() {
    const m = state.model;
    if (!m || !m.loaded) return;
    $("mc-params").textContent = m.n_params.toLocaleString();
    const t = m.test || {};
    const st = state.stats;
    const arch = state.arch;
    const rows = [];
    const row = (label, main, small, wide) => rows.push(el("div", wide ? { class: "wide" } : {}, el("dt", { text: label }), el("dd", {}, main, small ? el("small", { text: ` ${small}` }) : "")));
    if (t.accuracy != null) row("Accuracy on unseen speakers", pct(t.accuracy), t.accuracy_ci95 ? `95% interval ${pct(t.accuracy_ci95[0])} to ${pct(t.accuracy_ci95[1])}` : "");
    if (t.chance != null) row("Chance", pct(t.chance), `${m.classes.length} deliveries`);
    if (t.coverage_at_threshold != null) row("Answers on", `${pct(t.coverage_at_threshold)} of clips`, `correct on ${pct(t.accuracy_when_answering)} of those`);
    const cc = t.cross_corpus && Object.values(t.cross_corpus).filter((v) => v && v.accuracy != null).map((v) => v.accuracy);
    if (cc && cc.length) row("On the other corpus", `${pct(Math.min(...cc))} to ${pct(Math.max(...cc))}`, "trained on one, tested on the other");
    row("Input", "3 s of mono audio", arch && arch.input ? `as 64 mel bands × ${arch.input[3]} frames` : "");
    row("Output", `${m.classes.length} delivery probabilities`, "or Unsure");
    const ms = (x) => (x && x.n ? fmtMs(x.p50) : "n/a");
    row("Inference time, median", ms(st && st.tract), `model alone; ${ms(st && st.server)} on the server in total`);
    if (m.ondevice_kb) row("On-device download", `about ${(m.ondevice_kb / 1024).toFixed(0)} MB`, "mostly the WebAssembly runtime");
    const dl = $("mc-stats");
    dl.textContent = "";
    rows.forEach((r) => dl.append(r));
    $("mc-flow").textContent = "Two proceesing paths of the same waveform. They share only the audio, and Praat does not feed the classifier. Today the server runs them one after the other, because Praat is not thread-safe, so the server measures one clip at a time and the total is the sum of the two.";
    const cb = $("cross-body");
    cb.textContent = "";
    for (const [k, v] of Object.entries(t.cross_corpus || {})) {
      if (!v || v.accuracy == null) continue;
      const ci = v.accuracy_ci95_speaker_bootstrap || v.accuracy_ci95;
      cb.append(el("tr", {}, el("td", { text: k.split("_to_").map(dsName).join(" to ") }), el("td", { text: String(v.n_test_speakers ?? "") }),
        el("td", { text: pct(v.accuracy) }), el("td", { text: ci ? `${pct(ci[0])} to ${pct(ci[1])}` : "n/a" })));
    }
    $("cross-table").hidden = !cb.children.length;
  }

  // ---------- principles ----------
  const CARDS = [
    { id: "privacy", title: "Privacy", short: "Run the model on your own device and your audio never leaves the tab.",
      body: ["Choose where the analysis runs. On your device the model runs in your browser as WebAssembly and nothing is sent anywhere. On the server your audio is analysed in memory and thrown away."],
      more: ["The server keeps one line of timings per run: no audio, no address, no account.", "A recording lives only until you replace it or leave the page.", "No third-party scripts, fonts or analytics.", "Praat measurements run only on the server, so they are off in on-device mode."],
      extra: { text: "Run on your device", go: () => chooseEngine("device") } },
    { id: "transparency", title: "Transparency", short: "Every answer shows its confidence, and the model says when it is unsure.",
      body: ["Each result comes with a confidence meter and the line below which the model declines to answer. The numbers behind it are published: accuracy on speakers it never heard, with an interval, and the tests where it does badly."],
      more: ["The sample scoreboard marks every clip correct, incorrect or unsure.", "A cross-corpus test shows how far accuracy falls on unfamiliar recording conditions.", "The model card is generated from the same metrics file the app reads, so the page and the card cannot disagree."],
      extra: { text: "Read the model card", href: "/model-card" } },
    { id: "latency", title: "Latency", short: "Every step is timed on every run, and real numbers are reported, not estimates.",
      body: ["Feature extraction, Praat and model inference are timed separately. On the server the total is their sum, because Praat is not thread-safe, so the server measures one clip at a time. On your device the model runs with no network request."],
      more: ["Averages, medians and slow cases (95th percentile) over all runs since launch sit near the bottom of the page.", "The first request after the server wakes is timed but left out of the averages."],
      extra: { text: "See the numbers", go: () => goTo("latency") } },
    { id: "fairness", title: "Fairness", short: "Accuracy is reported by group, on speakers the model never trained on.",
      body: ["An overall number can look healthy while one group of speakers fails. That is what the bias-assessment work I co-authored at Sonos was about. Here accuracy is broken out by sex, age, race and ethnicity wherever the data has labels, with intervals resampled by speaker."],
      more: ["Where two intervals overlap, the data cannot show a difference in either direction.", "The labels are used for evaluation only. The app never estimates them.", "Both corpora are acted speech, and neither is a representative sample of people."],
      extra: { text: "See the breakdown", go: () => goTo("groups", "groups") } },
    { id: "cost", title: "Cost-awareness", short: "What is costs to run inference at scale, estimated on AWS with different usage demands.",
      body: ["The estimate comes from the measured server time per analysis, priced on a small AWS instance. When the analysis runs on your device, it costs the host no compute."],
      more: ["You can change how busy the instance is. Real traffic is never flat out.", "It leaves out bandwidth, storage, monitoring and engineering time."],
      extra: { text: "See the estimate", go: () => goTo("cost-wrap") } },
  ];

  function goTo(id, openId) {
    const d = $("card-dialog");
    if (d.open) d.close();
    if (openId && $(openId)) $(openId).open = true;
    const t = $(id);
    if (t && !t.hidden) t.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }

  function chooseEngine(which) {
    const input = document.querySelector(`#engine input[value="${which}"]`);
    if (input && !input.disabled) { input.checked = true; setEngine(which); }
    goTo("engine");
  }

  function openCard(id) {
    const c = CARDS.find((x) => x.id === id);
    if (!c) return;
    $("cd-title").textContent = c.title;
    const body = $("cd-body");
    body.textContent = "";
    c.body.forEach((t) => body.append(el("p", { text: t })));
    const more = $("cd-more-body");
    more.textContent = "";
    const ul = el("ul");
    c.more.forEach((t) => ul.append(el("li", { text: t })));
    more.append(ul);
    $("cd-more").open = false;
    const extra = $("cd-extra");
    extra.textContent = "";
    if (c.extra && c.extra.href) extra.append(el("a", { href: c.extra.href, text: c.extra.text }));
    else if (c.extra) {
      const b = el("button", { type: "button", class: "linkbtn", text: c.extra.text });
      b.addEventListener("click", c.extra.go);
      extra.append(b);
    }
    const d = $("card-dialog");
    if (d.showModal) d.showModal(); else d.setAttribute("open", "");
  }

  function buildCards() {
    const ul = $("cards");
    ul.textContent = "";
    for (const c of CARDS) {
      const b = el("button", { type: "button", class: "card", "data-card": c.id, "aria-haspopup": "dialog" }, el("h3", { text: c.title }), el("p", { text: c.short }));
      ul.append(el("li", {}, b));
    }
  }

  // ---------- modes and your own recordings ----------
  function emptySlot() { return { pcm: null, sr: 0, result: null, source: null, label: "Empty", playing: null, run: null, truth: null }; }

  function setMode(mode) {
    state.mode = mode;
    document.body.classList.toggle("mode-samples", mode === "samples");
    document.body.classList.toggle("mode-record", mode === "record");
    const r = document.querySelector(`#mode input[value="${mode}"]`);
    if (r) r.checked = true;
    state.pairToken++;
    if (state.recording) finishRecording();
    stopPlayback("a"); stopPlayback("b");
    state.slots.a = emptySlot(); state.slots.b = emptySlot();
    state.pair = null;
    document.querySelectorAll("#pairlist button").forEach((b) => b.setAttribute("aria-pressed", "false"));
    const sent = $("sentence");
    if (mode === "record") {
      const typed = $("own-sentence").value.trim();
      sent.classList.toggle("plain", !typed);
      sent.textContent = typed || "Plan your sentence above, then record it twice in two different ways.";
    } else {
      sent.classList.add("plain");
      sent.textContent = "Choose a pair above.";
    }
    showError("");
    renderAll();
  }

  function renderOvals(key) {
    const box = $(`ovals-${key}`), s = state.slots[key];
    box.textContent = "";
    if (state.mode !== "record" || !state.model || !state.model.classes) return;
    if (state.recording && state.recording.key === key) { box.append(el("span", { class: "askpick", text: "Recording. Press Stop when you are done." })); return; }
    if (!s.pcm || s.source !== "live") { box.append(el("span", { class: "askpick", text: "Record, then tap the delivery you performed." })); return; }
    const other = state.slots[key === "a" ? "b" : "a"];
    box.append(el("span", { class: "askpick", text: s.truth ? "Change it if that was wrong:" : "Which delivery did you perform?" }));
    for (const c of state.model.classes) {
      const b = el("button", { type: "button", class: "oval", "aria-pressed": String(s.truth === c), text: cap(c) });
      if (other.truth === c && s.truth !== c) { b.disabled = true; b.title = `Already used for ${key === "a" ? "B" : "A"}`; }
      b.addEventListener("click", () => {
        s.truth = c;
        renderSlot(key); renderSlot(key === "a" ? "b" : "a");
      });
      box.append(b);
    }
    if (s.truth) box.append(el("span", { class: "saved", text: `✓ Saved: ${key.toUpperCase()} is ${s.truth}.` }));
  }

  // ---------- limits and next steps ----------
  // ---------- init ----------
  function wire() {
    document.querySelectorAll('#engine input[name="engine"]').forEach((i) =>
      i.addEventListener("change", () => { if (i.checked) setEngine(i.value); }));
    for (const k of ["a", "b"]) {
      $(`play-${k}`).addEventListener("click", () => (state.slots[k].playing ? (stopPlayback(k), renderSlot(k)) : playSlot(k)));
      $(`rec-${k}`).addEventListener("click", () => toggleRecord(k));
    }
    $("score-run").addEventListener("click", runScoreboard);
    document.querySelectorAll('#mode input[name="mode"]').forEach((i) =>
      i.addEventListener("change", () => { if (i.checked) setMode(i.value); }));
    document.addEventListener("click", (e) => { const b = e.target.closest("[data-card]"); if (b) openCard(b.dataset.card); });
    $("cd-close").addEventListener("click", () => $("card-dialog").close());
    $("card-dialog").addEventListener("click", (e) => { if (e.target === e.currentTarget) e.currentTarget.close(); });
    $("own-sentence").addEventListener("input", (e) => {
      const sent = $("sentence");
      const v = e.target.value.trim();
      sent.classList.toggle("plain", !v);
      sent.textContent = v || "Plan your sentence above, then record it twice in two different ways.";
    });
  }

  const REPO_URL = "https://github.com/blakeperrysmith/beyond-the-transcript"; // set to the public repository URL to show the footer link

  async function init() {
    wire();
    buildCards();
    if (REPO_URL) { $("repo-link").href = REPO_URL; $("repo-wrap").hidden = false; }
    renderAll();
    // The model details drive the threshold line, the group chart and the measurement labels, so retry once
    // (a cold start can time out the first request) and say so on the page if they never arrive.
    for (let attempt = 0; attempt < 2 && !state.model; attempt++) {
      try {
        state.model = await getJSON("/api/model");
      } catch (e) {
        console.error("Could not load /api/model:", e);
        if (attempt === 0) await new Promise((r) => setTimeout(r, 1500));
      }
    }
    if (!state.model) {
      state.model = { loaded: false };
      state.modelError = true;
    }
    renderBanner(); renderLimits(); renderGroups(); probeDevice();
    try {
      const s = await getJSON("/api/stats");
      state.stats = s.stats; state.storage = s.storage; state.hardware = s.hardware; state.tractVersion = s.tract_version;
    } catch (_) { /* footer stays empty */ }
    loadArchitecture();
    await loadManifest();
    if (!state.manifest) {
      const r = document.querySelector('#mode input[value="samples"]');
      r.disabled = true;
      setMode("record");
    }
    renderAll();
  }
  init();
})();
