#!/usr/bin/env python
"""Torch-free audio front-end for Indic-Conformer (numpy only).

Replaces the two torch pieces of the old server:
  * torchaudio.functional.resample  -> resample()      (exact port of torchaudio's
                                                         sinc_interp_hann kernel)
  * preprocessor.ts (TorchScript)   -> mel_spectrogram() (STFT + slaney mel + log +
                                                          per-feature norm)

Both are verified bit-close to the torch reference in verify_torchfree.py.
Nothing here imports torch; the runtime dependency is numpy alone.
"""
import math
import numpy as np

# ---- mel front-end constants (from indic_preproc_config.json / preprocessor.ts) ----
SR       = 16000
N_FFT    = 512
WIN      = 400          # 25 ms
HOP      = 160          # 10 ms
N_MELS   = 80
FMIN     = 0.0
FMAX     = SR / 2       # 8000
PREEMPH  = 0.97         # preprocessor.ts applies preemphasis (config's "null" was wrong)
LOG_GUARD = 2.0 ** -24  # ln(mel + 2^-24)
NORM_EPS  = 1e-5


# ============================================================================
# 1. Resampling — exact numpy port of torchaudio.functional.resample
#    (resampling_method="sinc_interp_hann", lowpass_filter_width=6, rolloff=0.99)
# ============================================================================
def _sinc_resample_kernel(orig_freq, new_freq, lowpass_filter_width=6, rolloff=0.99):
    gcd = math.gcd(int(orig_freq), int(new_freq))
    of, nf = int(orig_freq) // gcd, int(new_freq) // gcd
    base_freq = min(of, nf) * rolloff
    width = math.ceil(lowpass_filter_width * of / base_freq)

    idx = np.arange(-width, width + of, dtype=np.float64)[None, :] / of      # (1, K)
    t = np.arange(0, -nf, -1, dtype=np.float64)[:, None] / nf + idx          # (nf, K)
    t = t * base_freq
    t = np.clip(t, -lowpass_filter_width, lowpass_filter_width)
    window = np.cos(t * math.pi / lowpass_filter_width / 2) ** 2             # hann
    t = t * math.pi
    scale = base_freq / of
    kernels = np.where(t == 0.0, 1.0, np.sin(np.where(t == 0.0, 1.0, t)) / np.where(t == 0.0, 1.0, t))
    kernels = (kernels * window * scale).astype(np.float32)                  # (nf, K)
    return kernels, width, of, nf


def resample(waveform, orig_freq, new_freq):
    """1-D float32 waveform -> resampled float32, matching torchaudio.functional.resample."""
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    if orig_freq == new_freq:
        return waveform
    kernels, width, of, nf = _sinc_resample_kernel(orig_freq, new_freq)
    length = waveform.shape[-1]
    padded = np.pad(waveform, (width, width + of))
    K = kernels.shape[1]
    n_out = (len(padded) - K) // of + 1
    idxs = np.arange(n_out)[:, None] * of + np.arange(K)[None, :]            # (n_out, K)
    frames = padded[idxs]                                                    # (n_out, K)
    conv = frames.astype(np.float32) @ kernels.T                            # (n_out, nf)
    resampled = conv.reshape(-1)                                            # interleave nf per step
    target_length = math.ceil(new_freq * length / orig_freq)
    return resampled[:target_length].astype(np.float32)


# ============================================================================
# 2. Slaney mel filterbank (librosa-compatible: htk=False, norm="slaney")
# ============================================================================
def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / f_sp, math.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(f / min_log_hz) / logstep, mels)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / f_sp, math.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def _mel_filterbank():
    n_freq = N_FFT // 2 + 1
    fftfreqs = np.linspace(0.0, SR / 2.0, n_freq)
    mel_pts = np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MELS + 2)
    freq_pts = _mel_to_hz(mel_pts)
    fdiff = np.diff(freq_pts)
    ramps = freq_pts[:, None] - fftfreqs[None, :]
    fb = np.zeros((N_MELS, n_freq), dtype=np.float64)
    for i in range(N_MELS):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        fb[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (freq_pts[2:N_MELS + 2] - freq_pts[:N_MELS])              # slaney norm
    fb *= enorm[:, None]
    return fb.astype(np.float32)


_FB = _mel_filterbank()
# Hann window (periodic=False) of length WIN, zero-padded & centered inside N_FFT
_HANN = np.hanning(WIN).astype(np.float32)
_WIN_FULL = np.zeros(N_FFT, dtype=np.float32)
_WPAD = (N_FFT - WIN) // 2
_WIN_FULL[_WPAD:_WPAD + WIN] = _HANN


# ============================================================================
# 3. Mel spectrogram — matches preprocessor.ts (torch.stft center/reflect + log + norm)
# ============================================================================
def mel_spectrogram(samples):
    """mono float32 @16k -> (feat[1,80,T] float32, length[1] int64)."""
    x = np.ascontiguousarray(samples, dtype=np.float32)
    # preemphasis: y[0]=x[0], y[i]=x[i]-0.97*x[i-1]  (NeMo, applied before framing)
    x = np.concatenate([x[:1], x[1:] - PREEMPH * x[:-1]]).astype(np.float32)
    pad = N_FFT // 2                                                        # 256, center=True
    xp = np.pad(x, (pad, pad), mode="reflect")
    T = (len(xp) - N_FFT) // HOP + 1
    idxs = np.arange(T)[:, None] * HOP + np.arange(N_FFT)[None, :]          # (T, n_fft)
    frames = xp[idxs] * _WIN_FULL[None, :]                                  # (T, n_fft)
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)                            # (T, n_freq)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)          # magnitude^2
    mel = power @ _FB.T                                                    # (T, 80)
    mel = np.log(mel + LOG_GUARD)
    # per-feature normalization over all valid frames (ddof=1), then +eps
    mean = mel.mean(axis=0, keepdims=True)
    std = mel.std(axis=0, ddof=1, keepdims=True)
    mel = (mel - mean) / (std + NORM_EPS)
    feat = np.ascontiguousarray(mel.T[None], dtype=np.float32)             # [1, 80, T]
    length = np.array([T], dtype=np.int64)
    return feat, length
