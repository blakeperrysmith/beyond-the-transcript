# Beyond the transcript

A small demo that plays two recordings of the same sentence delivered differently and shows what the
audio carries that a transcript drops. Built as a worked example of delivery (emotion) classification
from audio alone, with Praat-measured prosody, an abstain option, and live latency numbers.

- Training: PyTorch, small CRNN (3 conv blocks, bidirectional GRU, 106,934 parameters).
- Serving: Python (FastAPI), inference in [tract](https://github.com/sonos/tract) via its Python bindings.
- Data: CREMA-D and RAVDESS only. No family or personal recordings are used anywhere.

## What is real and what is not

`artifacts/` is empty on purpose. Until you train on real data the app shows "No model is loaded".
A model trained with `--synthetic` is a pipeline test: it carries `data_source: "synthetic"` and the page
shows a warning banner. Never deploy one.

## Run it

```bash
pip install -r requirements.txt
# 1. train on Kaggle: notebooks/train_kaggle.ipynb  (download artifacts.tgz, unpack into ./artifacts)
# 2. build sample pairs from held-out speakers
python scripts/make_samples.py --ravdess <dir> --cremad <dir> --demographics <csv> \
       --split artifacts/split.json --out app/static/samples
# 3. architecture diagram (needs torchview + graphviz `dot`)
python scripts/arch_diagram.py --out app/static
# 4. serve
uvicorn app.main:app --port 8000
```

Tests: `pip install -r requirements-train.txt && pytest` (trains 3 synthetic epochs, checks PyTorch
and tract agree to 1e-3, exercises the API including error codes and rate limiting).

## Design decisions

- **One front end.** `btt/features.py` is used for training and serving, so there is no train/serve skew.
- **Speaker-disjoint evaluation.** Splits are by speaker within each corpus. Confidence intervals resample
  speakers, not clips. Cross-corpus results (train on one, test on the other) are reported separately.
- **Calibrated abstention.** Temperature scaling on validation, then a confidence threshold giving about 80% coverage.
- **tract parity test.** tract issue #2751 (GRU activation attributes ignored) means a silent mismatch
  is possible, so export is always followed by a PyTorch-vs-tract comparison.
- **Privacy.** Audio is decoded in memory and dropped. The database stores only timings. Rate limiting keys
  on a daily-salted hash held in memory. No third-party scripts, fonts or analytics.
- **No demographic inference.** The model never predicts or uses speaker attributes.

## Limits

Acted speech, adult speakers, mostly North American English. Labels describe delivery, not inner state,
and must not be used to judge people. The author's own production experience is machine-directed speech,
a different register from conversation. Model weights are non-commercial (RAVDESS CC BY-NC-SA 4.0;
CREMA-D ODbL). Sample clips are redistributed under those licences with attribution in the manifest.

## Deploy

### Render (free tier works)
1. Put this repo on GitHub with `artifacts/` and `app/static/samples/` filled in.
2. On render.com choose New, Web Service, pick the repo, set the language to **Docker**, instance type Free.
3. Set the health check path to `/api/health`. Render supplies `PORT`; the Dockerfile uses it.
4. Free instances sleep after 15 idle minutes and take about a minute to wake, and their disk is ephemeral.
   Switch to the Starter instance (always on, billed by the second) while you are sharing the link.

Hugging Face Docker Spaces also work but currently need a PRO subscription for free CPU hosting.

### Keep the latency counts: Supabase (optional)
Without a database the counts live on the server and reset on restart. To keep them:
1. Create a Supabase project. Open Connect and copy the **transaction pooler** connection string (port 6543).
2. Set it as the `DATABASE_URL` secret on the host (on Render: Environment).
3. Start the app. It creates a `btt` schema and a `runs` table with row-level security on, so
   Supabase's public REST API cannot read or write it. The app connects as the database role, which is
   not affected by row-level security.

The page says which storage is in use. If the database cannot be reached, the app falls back to local
storage and says so; an analysis never fails because of the log. Not yet tested against a live Supabase
project (no credentials were available when this was written).

### Anywhere that runs Python
Run `uvicorn app.main:app --host 0.0.0.0 --port $PORT` with `requirements.txt` installed. Optional env:
`DATABASE_URL`, `BTT_LOCAL_DB` (path of the local fallback file), `BTT_RATE_LIMIT_PER_MIN` (default 30), `BTT_ARTIFACTS`.

## Not built

In-browser inference. The interface for it is built (a server / on-your-device switch with a plain-language
privacy explanation) and stays disabled until the files described in `docs/WASM_STRETCH.md` exist.

## License

The code in this repository is released under the MIT License (see LICENSE).

The audio samples in app/static/samples/ and the trained model in artifacts/ are
not covered by that license. They derive from RAVDESS (Livingstone and Russo 2018,
CC BY-NC-SA 4.0) and CREMA-D (Cao et al. 2014, ODbL), so they carry those terms:
non-commercial use only, with attribution, and share-alike.
