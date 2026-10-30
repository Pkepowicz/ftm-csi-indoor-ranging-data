# Wi‑Fi FTM + CSI indoor ranging (data & preprocessing)

Supplementary material for *Integration of FTM and CSI measurements in Wi-Fi networks for precise ranging in indoor environments*.  
20 measurement locations (15 LOS, 5 NLOS), 2.4 GHz / 20 MHz.

**Contents:** ground-truth labels (`distance.txt`), raw FTM (`.dat`) and CSI (`.pcap`) captures, scripts to build `processed/features.csv` and `processed/csi_arrays.npz` (13 392 rows; not committed — see `.gitignore`).

```bash
pip install -r requirements.txt
python preprocessing.py
python scripts/prepare_features_with_music.py   # optional MUSIC columns
```

Machine-learning training code is not included.
