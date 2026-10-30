#!/usr/bin/env python3
"""Add frequency-smoothed MUSIC descriptors to processed/features.csv."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from music_features import MUSIC_FEATURE_NAMES, music_feature_matrix

FEATURES = ROOT / "processed" / "features.csv"
BACKUP = ROOT / "processed" / "features_pre_music.csv"
NPZ = ROOT / "processed" / "csi_arrays.npz"


def main() -> None:
    if not BACKUP.exists():
        pd.read_csv(FEATURES).to_csv(BACKUP, index=False)
    df = pd.read_csv(BACKUP)
    z = np.load(NPZ)
    if not np.array_equal(z["point_id"], df["point_id"].values):
        raise SystemExit("csi_arrays.npz rows are not aligned with features.csv")
    music = music_feature_matrix(z["csi_complex"])
    for i, name in enumerate(MUSIC_FEATURE_NAMES):
        df[name] = music[:, i]
    df.to_csv(FEATURES, index=False)
    print(f"Wrote {len(df)} rows with {len(MUSIC_FEATURE_NAMES)} MUSIC columns to {FEATURES}")
    print(df.groupby("type")[list(MUSIC_FEATURE_NAMES)].mean().round(3).T.to_string())


if __name__ == "__main__":
    main()
