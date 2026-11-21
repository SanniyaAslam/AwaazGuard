"""LFCC-GMM baseline: one GMM for bona fide frames, one for spoof frames.

Utterance score = mean log-likelihood(bona fide) - mean log-likelihood(spoof).
Higher score = more likely bona fide (same convention as AASIST).
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


class LfccGmmBaseline:
    def __init__(self, n_components: int = 64, covariance_type: str = "diag",
                 max_iter: int = 100, reg_covar: float = 1e-4, seed: int = 42):
        kw = dict(n_components=n_components, covariance_type=covariance_type,
                  max_iter=max_iter, reg_covar=reg_covar, random_state=seed)
        self.gmm_bona = GaussianMixture(**kw)
        self.gmm_spoof = GaussianMixture(**kw)

    def fit(self, bona_frames: np.ndarray, spoof_frames: np.ndarray) -> "LfccGmmBaseline":
        self.gmm_bona.fit(bona_frames.astype(np.float64))
        self.gmm_spoof.fit(spoof_frames.astype(np.float64))
        return self

    def score_utterances(self, feats: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        x = feats.astype(np.float64)
        llr = self.gmm_bona.score_samples(x) - self.gmm_spoof.score_samples(x)
        bounds = np.cumsum(np.concatenate([[0], lengths]))
        return np.array([llr[bounds[i]:bounds[i + 1]].mean() for i in range(len(lengths))])
