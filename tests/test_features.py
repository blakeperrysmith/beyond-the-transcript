import numpy as np

from btt.features import HOP, SR, WIN_FRAMES, featurize, fit_window, log_mel, to_16k


def tone(sr, secs=1.5, f=220.0):
    t = np.arange(int(sr * secs)) / sr
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_shape_is_fixed_for_any_length_and_rate():
    for sr, secs in [(16000, 0.5), (16000, 6.0), (44100, 2.0), (48000, 3.0)]:
        assert featurize(tone(sr, secs), sr).shape == (1, 64, WIN_FRAMES)


def test_resample_length():
    assert abs(len(to_16k(tone(48000, 2.0), 48000)) - 2 * SR) <= 2


def test_gain_invariance():
    y = tone(SR)
    a, b = log_mel(y), log_mel(y * 0.1)
    # RMS normalisation happens inside log_mel, so level should barely matter
    assert np.max(np.abs(a - b)) < 1e-2


def test_loud_window_is_chosen_when_cropping():
    quiet = np.zeros(SR * 2, dtype=np.float32)
    loud = tone(SR, 3.0)
    feat = log_mel(np.concatenate([quiet, loud, quiet]))
    win = fit_window(feat)
    full_mean = feat.mean()
    assert win.mean() > full_mean
    assert win.shape[1] == WIN_FRAMES


def test_frame_rate():
    assert HOP == 160 and SR == 16000  # 100 frames per second, 3 s = 300 frames
