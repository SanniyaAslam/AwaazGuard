"""Linear-Frequency Cepstral Coefficients (LFCC) - the front-end of the ASVspoof GMM baseline.

Pure NumPy/SciPy implementation so results are identical on every machine.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dct


def linear_filterbank(n_filters: int, n_fft: int, sr: int) -> np.ndarray:
    """Triangular filters equally spaced on a LINEAR (not mel) frequency axis."""
    n_bins = n_fft // 2 + 1
    edges = np.linspace(0, sr / 2, n_filters + 2)
    bin_freqs = np.linspace(0, sr / 2, n_bins)
    fb = np.zeros((n_filters, n_bins), dtype=np.float32)
    for i in range(n_filters):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (bin_freqs - lo) / (mid - lo)
        down = (hi - bin_freqs) / (hi - mid)
        fb[i] = np.maximum(0, np.minimum(up, down))
    return fb


def deltas(feat: np.ndarray, width: int = 2) -> np.ndarray:
    """Standard regression deltas over +-width frames (frames x dims)."""
    if len(feat) == 0:
        return feat
    padded = np.pad(feat, ((width, width), (0, 0)), mode="edge")
    denom = 2 * sum(k * k for k in range(1, width + 1))
    out = np.zeros_like(feat)
    for k in range(1, width + 1):
        out += k * (padded[width + k: width + k + len(feat)] - padded[width - k: width - k + len(feat)])
    return out / denom


def lfcc(wav: np.ndarray, sr: int, n_fft: int = 512, win_length: int = 320,
         hop_length: int = 160, n_filters: int = 20, n_ceps: int = 20,
         use_deltas: bool = True, pre_emphasis: float = 0.97) -> np.ndarray:
    """Return LFCC features with shape (frames, n_ceps * (3 if use_deltas else 1))."""
    wav = np.asarray(wav, dtype=np.float32)
    if len(wav) < win_length:
        wav = np.pad(wav, (0, win_length - len(wav)))
    wav = np.append(wav[0], wav[1:] - pre_emphasis * wav[:-1])

    n_frames = 1 + (len(wav) - win_length) // hop_length
    idx = np.arange(win_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    frames = wav[idx] * np.hamming(win_length).astype(np.float32)

    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    energies = power @ linear_filterbank(n_filters, n_fft, sr).T
    log_e = np.log(energies + 1e-8)
    ceps = dct(log_e, type=2, axis=1, norm="ortho")[:, :n_ceps]

    if use_deltas:
        d1 = deltas(ceps)
        ceps = np.concatenate([ceps, d1, deltas(d1)], axis=1)
    return ceps.astype(np.float32)
