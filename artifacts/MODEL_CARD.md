# Model card: delivery classifier

A small CRNN that labels how a sentence was *delivered* (neutral, happy, sad, angry,
fearful, disgust) from the audio signal alone, without using the transcript. It exists to
demonstrate that speech carries information a one-best transcript throws away.

## Intended use and non-use
- Intended: an educational demo of audio-native modelling, and a worked example of
  speaker-disjoint training, calibration, abstention and per-group evaluation.
- Not intended: judging a real person's emotional state, hiring, screening, surveillance,
  or any decision about people. Acted emotion is not felt emotion.
- The app never infers age, sex or any other demographic attribute. Demographic metadata
  is used only offline, to measure whether accuracy differs across groups.

## Model
- 106,934 parameters. Three conv blocks, a bidirectional GRU over time, mean pool, linear head.
- Input: 64-band log-mel, 3 s window, 16 kHz. The feature code is shared by training and serving.
- Trained in PyTorch, exported to ONNX, served with tract (Sonos' inference engine).
- PyTorch vs tract parity on 32 test clips: max logit difference 2.9e-06, argmax agreement 100.0%.

## Data
- CREMA-D, Cao et al. 2014, Open Database License (ODbL)
- RAVDESS, Livingstone and Russo 2018, CC BY-NC-SA 4.0 (non-commercial)
- Both corpora are acted speech by adults in North American English. RAVDESS 'calm' and
  'surprised' were dropped because CREMA-D has no counterpart.
- Split by speaker (seed 0), within each corpus, so no speaker appears in more than one split.
- Because RAVDESS is licensed non-commercial, so is this model.

## Results (held-out speakers)
- 1500 clips from 18 speakers. Chance is 16.7%.
- Accuracy 62.3% (95% CI 59-66%, bootstrap over speakers), macro-F1 0.63.
- Calibration after temperature scaling (T=1.07): ECE 0.033.
- Abstention: below confidence 0.45 the app says 'not sure'. On test this answers 81.1% of clips with 68.0% accuracy on those.

### Per group (accuracy, 95% CI over speakers)
Small groups have wide intervals. A gap smaller than the interval is not evidence of a difference.

| Grouping | Group | Clips | Speakers | Accuracy | 95% CI |
|---|---|---|---|---|---|
| dataset | cremad | 1148 | 14 | 61.9% | 59-65% |
| dataset | ravdess | 352 | 4 | 63.6% | 57-71% |
| sex | female | 756 | 9 | 64.3% | 60-69% |
| sex | male | 744 | 9 | 60.3% | 56-64% |
| age_bucket | 20-29 | 246 | 3 | 63.0% | 61-66% |
| age_bucket | 30-49 | 656 | 8 | 59.6% | 55-64% |
| age_bucket | 50+ | 246 | 3 | 67.1% | 63-73% |
| race | African American | 82 | 1 | 59.8% | n/a |
| race | Caucasian | 1066 | 13 | 62.1% | 58-66% |

Per-speaker accuracy across 18 test speakers: min 47.6%, median 61.6%, max 75.0%.
The spread between speakers matters more than the average.

## Known limits
- Acted, studio-quality, adult speech. Not validated for children, older adults beyond the corpus range,
  non-American accents, other languages, phone audio, background noise or spontaneous conversation.
- The author's earlier production speech work was machine-directed speech. Spontaneous speech has hesitations,
  self-repair, overlap and blended emotions that this model never saw.
- Six forced categories are a simplification. The abstain state exists because many real utterances fit none.
- Latency figures in the app are from the demo host, not from this build machine.

## Reproduce
`python -m btt.train --ravdess DIR --cremad DIR --demographics CSV --out artifacts`
Built 2026-10-05T01:01:53+00:00 with torch 2.11.0+cu128, Python 3.13.15.
