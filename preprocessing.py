"""Wi-Fi FTM + CSI fusion preprocessing (v2).

This pipeline replaces the original summary-only export. For every FTM ranging
event it builds a FIXED-SIZE representation by aggregating the burst of CSI
packets that surround it in time.

Why aggregation (not sequence modeling):
  Each FTM measurement is associated with a burst of 2-31 CSI packets that arrive
  near-simultaneously (within +/-80 ms). They are repeated noisy snapshots of the
  SAME channel state, not a time series. So we average them into a single complex
  CSI vector of length 64 (the subcarrier axis is the fixed, meaningful dimension).
  This denoises the channel estimate and gives every measurement identical shape
  regardless of its packet count.

Outputs (written to processed/):
  - features.csv     Flat per-measurement features for classical/shallow models:
                       metadata + per-subcarrier amplitude / sanitized phase /
                       amplitude-stability + CIR/R scalar features.
  - csi_arrays.npz   Arrays for deep models:
                       csi_complex [N, 64] complex aggregated CSI,
                       channels    [N, 64, 3] (amplitude, sanitized phase, amp std),
                       plus aligned labels and FTM scalars.

Caveats (documented in IDEAS.md):
  - CSI is 20 MHz / 64 subcarriers => CIR range resolution ~7.5 m/tap. CIR and R
    features are coarse aggregate descriptors, not true multipath profiles.
  - Nexmon CSI phase carries CFO/SFO/packet-detection offsets; phase is sanitized
    by per-packet amplitude-weighted linear detrending before aggregation.
"""

import os
import struct

import numpy as np
import pandas as pd
from scipy import stats
from scapy.all import PcapReader, UDP

# --- 1. CONFIGURATION ---
LABELS_FILE = "distance.txt"
FTM_DIR = "FTM_data"
CSI_DIR = "CSI_data"
OUT_DIR = "processed"

SEARCH_WINDOW_SEC = 0.08  # +/- 80 ms anchored window around each FTM event
N_SUB = 64                # subcarriers (20 MHz)
MIN_FRAMES = 2            # drop 1-frame bursts (failures)
MAX_FRAMES = 31           # drop oversized OS dumps
C_MPS = 299_792_458.0     # speed of light, for raw RTT -> distance

# Null / non-data FFT bins after np.fft.fftshift (64-bin, 20 MHz HT).
# DC_BIN: 802.11 null at DC (LO leakage on Nexmon/Broadcom).
# EDGE_BINS: guard-band / Nexmon artifacts (~30--50x median on sc 60--61).
DC_BIN = 32
EDGE_BINS = frozenset({0, 1, 2, 3, 60, 61, 62, 63})
NULL_BINS = EDGE_BINS | {DC_BIN}


def zero_null_subcarriers(csi: np.ndarray) -> np.ndarray:
    """Zero DC, guard-band, and Nexmon edge bins before burst aggregation."""
    out = np.asarray(csi, dtype=np.complex128).copy()
    out[list(NULL_BINS)] = 0.0
    return out


# --- 2. PARSERS (unchanged logic from v1) ---
def process_ftm_data(ftm_file):
    valid = []
    with open(ftm_file, mode="rb") as f:
        content = f.read()
    step = 0
    while len(content) >= step + 79:
        data = content[step:step + 79]
        rssi = data[18]
        if rssi > 127:
            rssi -= 256
        rtt = struct.unpack("Q", data[23:31])[0]
        ts = struct.unpack("Q", data[71:79])[0]
        if rtt > 0:
            valid.append({"ftm_rssi": rssi, "ftm_rtt": rtt, "ftm_time": ts / 1_000_000.0})
        step += 79
    return pd.DataFrame(valid)


def process_raw_csi_packets(pcap_file):
    rows = []
    header_offset = 18
    csi_len = N_SUB * 4  # 256 bytes: 64 subcarriers * (int16 real + int16 imag)
    with PcapReader(pcap_file) as pcap:
        for pkt in pcap:
            if UDP not in pkt or (pkt[UDP].dport != 5500 and pkt[UDP].sport != 5500):
                continue
            payload = bytes(pkt[UDP].payload)
            if len(payload) < 2 or struct.unpack("<H", payload[:2])[0] != 0x1111:
                continue
            if len(payload) < header_offset + csi_len:
                raise ValueError(f"Malformed CSI packet in {pcap_file}: {len(payload)} bytes")
            raw = payload[header_offset:header_offset + csi_len]
            csi = np.frombuffer(raw, dtype=np.int16)
            real = csi[0::2]
            imag = csi[1::2]
            if len(real) != N_SUB:
                raise ValueError(f"Got {len(real)} subcarriers (expected {N_SUB}) in {pcap_file}")
            shifted = np.fft.fftshift(real + 1j * imag)  # DC-centered spectrum
            rows.append({"csi_time": float(pkt.time), "csi": shifted})
    return pd.DataFrame(rows)


# --- 3. CSI AGGREGATION & FEATURE ENGINEERING ---
def _sanitize_phase(csi_frame):
    """Amplitude-weighted linear detrend of one packet's phase across subcarriers.

    Removes the linear phase ramp (timing/SFO) and constant offset (CFO) that
    dominate raw Nexmon phase. Returns sanitized phase [64].
    """
    amp = np.abs(csi_frame)
    phase = np.unwrap(np.angle(csi_frame))
    k = np.arange(N_SUB) - (N_SUB - 1) / 2.0
    w = amp + 1e-9
    # Weighted least squares fit of phase ~ a*k + b
    sw = w.sum()
    kbar = (w * k).sum() / sw
    pbar = (w * phase).sum() / sw
    var_k = (w * (k - kbar) ** 2).sum()
    a = (w * (k - kbar) * (phase - pbar)).sum() / (var_k + 1e-12)
    b = pbar - a * kbar
    return phase - (a * k + b)


def _cir_features(h_complex):
    """CIR-shape descriptors from an aggregated complex CSI vector.

    h_complex is DC-centered; undo the shift before the IFFT.

    IMPORTANT honesty caveat: per-packet phase is sanitized (linear delay ramp
    removed in `_sanitize_phase`) before aggregation, so the delay ORIGIN of this
    pseudo-CIR is not physical and the peak does not sit at tap 0. These features
    therefore describe energy CONCENTRATION / DISPERSION of the channel (driven by
    frequency selectivity), measured RELATIVE TO THE DETECTED PEAK tap, rather than
    a true resolved power-delay profile. At 20 MHz (~7.5 m/tap) only coarse,
    aggregate multipath structure is observable. They are NOT FUSIC-style resolved
    direct-path measurements.
    """
    cir = np.fft.ifft(np.fft.ifftshift(h_complex))
    mag = np.abs(cir)
    power = mag ** 2
    n = power.size
    total = power.sum() + 1e-12

    k_max = int(np.argmax(power))
    p_max = power[k_max]

    # Rician-K: dominant-tap power vs the summed power of all other taps (linear).
    rician_k = p_max / (total - p_max + 1e-12)

    # Energy concentration in a +/-1-tap window AROUND THE DETECTED PEAK (not the
    # fixed taps [0,1], which captured the empty edge of the delay axis). R in (0,1]:
    # high => energy concentrated on one tap (flat / LOS-like channel),
    # low => energy spread over many taps (frequency-selective / multipath).
    window = [(k_max + d) % n for d in (-1, 0, 1)]
    R = float(power[window].sum() / total)

    # LOS-dominance factor (replaces the numerically broken Paper-18 eta, whose old
    # normalize-to-1 step made it a ~1e6 floating-point artifact). Defined as the
    # base-10 log of the peak-tap magnitude over the mean of all other taps; finite,
    # >= 0, and O(0-2) in practice.
    rest_mean = (mag.sum() - mag[k_max]) / (n - 1) + 1e-12
    eta = float(np.log10(mag[k_max] / rest_mean + 1e-12))

    # RMS delay spread RELATIVE to the peak tap (wrapped distances). A standard
    # channel-dispersion descriptor; larger => more spread / multipath-like.
    delta = np.arange(n) - k_max
    delta = (delta + n // 2) % n - n // 2  # signed wrapped distance to peak (taps)
    mean_excess = (power * delta).sum() / total
    rms_spread = float(np.sqrt((power * (delta - mean_excess) ** 2).sum() / total))

    return {
        "cir_kmax": k_max,
        "cir_pmax": float(p_max / total),    # normalized strongest-tap power
        "rician_k": float(rician_k),
        "R": R,
        "eta": eta,
        "cir_rms_spread": rms_spread,
        "cir_mean": float(mag.mean()),
        "cir_std": float(mag.std()),
        "cir_skew": float(stats.skew(mag)),
        "cir_kurt": float(stats.kurtosis(mag)),
    }


def aggregate_burst(csi_stack):
    """Aggregate a [frames, 64] complex CSI burst into fixed per-subcarrier channels.

    Returns:
      amp        [64] non-coherent mean amplitude (stable estimate)
      phase      [64] circular-mean sanitized phase
      amp_std    [64] amplitude std across packets (stability)
      h_complex  [64] aggregated complex CSI (amp * exp(j*phase)), DC-centered
    """
    csi_stack = np.stack([zero_null_subcarriers(f) for f in csi_stack])
    mags = np.abs(csi_stack)                       # [frames, 64]
    amp = mags.mean(axis=0)
    amp_std = mags.std(axis=0)

    sanitized = np.stack([_sanitize_phase(f) for f in csi_stack])  # [frames, 64]
    phase = np.angle(np.mean(np.exp(1j * sanitized), axis=0))      # circular mean

    h_complex = amp * np.exp(1j * phase)
    return amp, phase, amp_std, h_complex


def extract_features_for_point(point_id, distance, los_type, ftm_file, csi_file):
    print(f"[{los_type}] Point {point_id} (d={distance} m)...")
    df_ftm = process_ftm_data(ftm_file).sort_values("ftm_time").reset_index(drop=True)
    df_csi = process_raw_csi_packets(csi_file).sort_values("csi_time").reset_index(drop=True)
    if df_ftm.empty or df_csi.empty:
        print(f"  -> WARNING: empty FTM or CSI for point {point_id}")
        return None

    # Pass 1: global median offset alignment between the two clocks.
    rough = pd.merge_asof(
        df_ftm[["ftm_time"]], df_csi[["csi_time"]],
        left_on="ftm_time", right_on="csi_time", direction="nearest", tolerance=1.0,
    ).dropna(subset=["csi_time"])
    if rough.empty:
        print(f"  -> ERROR: clocks desynced for point {point_id}")
        return None
    offset = (rough["ftm_time"] - rough["csi_time"]).median()
    aligned = df_csi["csi_time"].values + offset
    csi_vals = df_csi["csi"].values

    rows = []
    amp_list, phase_list, ampstd_list, hcx_list = [], [], [], []

    # Pass 2: anchored extraction + aggregation per FTM event.
    for _, ftm in df_ftm.iterrows():
        t = ftm["ftm_time"]
        mask = (aligned >= t - SEARCH_WINDOW_SEC) & (aligned <= t + SEARCH_WINDOW_SEC)
        idx = np.where(mask)[0]
        frames = len(idx)
        if frames < MIN_FRAMES or frames > MAX_FRAMES:
            continue

        csi_stack = np.stack(csi_vals[idx])  # [frames, 64] complex
        amp, phase, amp_std, h_complex = aggregate_burst(csi_stack)

        row = {
            "point_id": point_id,
            "distance": distance,
            "type": los_type,
            "is_nlos": int(los_type.upper() == "NLOS"),
            "ftm_rssi": ftm["ftm_rssi"],
            "ftm_rtt": ftm["ftm_rtt"],
            "raw_distance_m": ftm["ftm_rtt"] * 1e-12 * C_MPS / 2.0,
            "frame_count": frames,
            "burst_duration_ms": (aligned[idx[-1]] - aligned[idx[0]]) * 1000.0,
        }
        row.update(_cir_features(h_complex))
        for i in range(N_SUB):
            row[f"amp_{i}"] = amp[i]
            row[f"phase_{i}"] = phase[i]
            row[f"ampstd_{i}"] = amp_std[i]
        rows.append(row)

        amp_list.append(amp)
        phase_list.append(phase)
        ampstd_list.append(amp_std)
        hcx_list.append(h_complex)

    if not rows:
        print(f"  -> WARNING: no valid bursts for point {point_id}")
        return None

    print(f"  -> {len(rows)} measurements (dropped {len(df_ftm) - len(rows)} anomalies)")
    arrays = {
        "amp": np.stack(amp_list),
        "phase": np.stack(phase_list),
        "ampstd": np.stack(ampstd_list),
        "csi_complex": np.stack(hcx_list),
    }
    return pd.DataFrame(rows), arrays


# --- 4. EXECUTION ---
def main():
    print("=" * 52)
    print("  Wi-Fi FTM & CSI FUSION PIPELINE v2 (aggregation)")
    print("=" * 52)
    if not os.path.exists(LABELS_FILE):
        raise SystemExit(f"Missing {LABELS_FILE}")

    labels = pd.read_csv(LABELS_FILE)
    frames, all_arrays = [], []

    for _, row in labels.iterrows():
        pid = int(row["point_id"])
        dist = float(row["distance"])
        ltype = str(row["type"]).strip()
        ftm_file = os.path.join(FTM_DIR, f"FTM_ftm_point{pid}.dat")
        csi_file = os.path.join(CSI_DIR, f"csi_point{pid}.pcap")
        if not (os.path.exists(ftm_file) and os.path.exists(csi_file)):
            print(f"  -> ERROR: missing files for point {pid}")
            continue
        result = extract_features_for_point(pid, dist, ltype, ftm_file, csi_file)
        if result is not None:
            df_point, arrays = result
            frames.append(df_point)
            all_arrays.append(arrays)

    if not frames:
        raise SystemExit("Pipeline failed: no valid data.")

    master = pd.concat(frames, ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    features_path = os.path.join(OUT_DIR, "features.csv")
    master.to_csv(features_path, index=False)

    npz_path = os.path.join(OUT_DIR, "csi_arrays.npz")
    np.savez_compressed(
        npz_path,
        csi_complex=np.concatenate([a["csi_complex"] for a in all_arrays]),
        channels=np.stack(
            [
                np.concatenate([a["amp"] for a in all_arrays]),
                np.concatenate([a["phase"] for a in all_arrays]),
                np.concatenate([a["ampstd"] for a in all_arrays]),
            ],
            axis=-1,
        ),  # [N, 64, 3]
        point_id=master["point_id"].to_numpy(),
        distance=master["distance"].to_numpy(),
        is_nlos=master["is_nlos"].to_numpy(),
        ftm_rtt=master["ftm_rtt"].to_numpy(),
        raw_distance_m=master["raw_distance_m"].to_numpy(),
    )

    print("\n" + "=" * 52)
    print("              PIPELINE COMPLETE")
    print("=" * 52)
    print(f"Measurements : {len(master)}")
    print(f"Points       : {master['point_id'].nunique()}")
    print(f"Feature cols : {len(master.columns)}")
    print(f"Saved        : {features_path}")
    print(f"               {npz_path}")
    print("=" * 52)


if __name__ == "__main__":
    main()
