"""Frequency-smoothed MUSIC delay estimation on aggregated 20 MHz CSI.

Operates on the burst-aggregated complex CSI (`csi_complex` in csi_arrays.npz).
Phase sanitization removed the common linear phase slope, so absolute delay is
unavailable; only delays *relative* to the other resolved paths are meaningful.
"""

from __future__ import annotations

import numpy as np

SUBCARRIER_SPACING_HZ = 312_500.0
VALID_FIRST, VALID_LAST = 4, 59
DC_INDEX = 32
SUBARRAY_LEN = 28
MAX_PATHS = 5
MIN_RELATIVE_PATH_POWER = 0.05
DELAY_GRID_NS = np.arange(-400.0, 400.0 + 0.5, 1.0)

MUSIC_FEATURE_NAMES = (
    "music_order",
    "music_excess_delay_ns",
    "music_first_path_power_ratio",
    "music_rms_delay_spread_ns",
    "music_peak_contrast_db",
)


def _steering(n: int, delays_ns: np.ndarray) -> np.ndarray:
    k = np.arange(n)[:, None]
    return np.exp(-2j * np.pi * k * SUBCARRIER_SPACING_HZ * delays_ns[None, :] * 1e-9)


def _contiguous_band(h64: np.ndarray) -> np.ndarray:
    h = np.asarray(h64, dtype=np.complex128).copy()
    h[DC_INDEX] = 0.5 * (h[DC_INDEX - 1] + h[DC_INDEX + 1])
    return h[VALID_FIRST : VALID_LAST + 1]


def _mdl_order(eigvals_desc: np.ndarray, n_snapshots: int) -> int:
    m = len(eigvals_desc)
    lam = np.clip(eigvals_desc, 1e-12, None)
    best_k, best_val = 1, np.inf
    for k in range(1, min(MAX_PATHS, m - 1) + 1):
        tail = lam[k:]
        geo = np.exp(np.mean(np.log(tail)))
        ari = np.mean(tail)
        val = -n_snapshots * (m - k) * np.log(geo / ari) + 0.5 * k * (2 * m - k) * np.log(n_snapshots)
        if val < best_val:
            best_k, best_val = k, val
    return best_k


_A_SUB = _steering(SUBARRAY_LEN, DELAY_GRID_NS)


def music_descriptors(h64: np.ndarray) -> dict[str, float]:
    h = _contiguous_band(h64)
    n = len(h)
    if not np.any(np.abs(h) > 0):
        return {name: 0.0 for name in MUSIC_FEATURE_NAMES}
    h = h / (np.linalg.norm(h) + 1e-12)
    m_snap = n - SUBARRAY_LEN + 1
    snaps = np.stack([h[i : i + SUBARRAY_LEN] for i in range(m_snap)], axis=1)
    r = snaps @ snaps.conj().T / m_snap
    j = np.eye(SUBARRAY_LEN)[::-1]
    r = 0.5 * (r + j @ r.conj() @ j)
    eigvals, eigvecs = np.linalg.eigh(r)
    order_idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order_idx], eigvecs[:, order_idx]
    p = _mdl_order(eigvals, 2 * m_snap)
    noise = eigvecs[:, p:]
    proj = noise.conj().T @ _A_SUB
    spectrum = 1.0 / (np.sum(np.abs(proj) ** 2, axis=0) + 1e-12)
    spec_db = 10 * np.log10(spectrum / spectrum.max())

    is_peak = np.r_[False, (spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] >= spectrum[2:]), False]
    peak_idx = np.where(is_peak)[0]
    if len(peak_idx) == 0:
        peak_idx = np.array([int(np.argmax(spectrum))])
    peak_idx = peak_idx[np.argsort(spectrum[peak_idx])[::-1][:p]]
    delays = DELAY_GRID_NS[peak_idx]

    a_full = _steering(n, delays)
    gains, *_ = np.linalg.lstsq(a_full, h, rcond=None)
    powers = np.abs(gains) ** 2
    keep = powers >= MIN_RELATIVE_PATH_POWER * powers.max()
    if not keep.all():
        delays = delays[keep]
        gains, *_ = np.linalg.lstsq(_steering(n, delays), h, rcond=None)
        powers = np.abs(gains) ** 2
    total = powers.sum() + 1e-12
    order_by_delay = np.argsort(delays)
    first = order_by_delay[0]
    strongest = int(np.argmax(powers))
    mean_delay = float(np.sum(powers * delays) / total)
    rms = float(np.sqrt(np.sum(powers * (delays - mean_delay) ** 2) / total))
    return {
        "music_order": float(len(delays)),
        "music_excess_delay_ns": float(delays[strongest] - delays[first]),
        "music_first_path_power_ratio": float(powers[first] / total),
        "music_rms_delay_spread_ns": rms,
        "music_peak_contrast_db": float(-np.median(spec_db)),
    }


def music_feature_matrix(csi_complex: np.ndarray) -> np.ndarray:
    return np.array(
        [[music_descriptors(h)[name] for name in MUSIC_FEATURE_NAMES] for h in csi_complex],
        dtype=np.float64,
    )
