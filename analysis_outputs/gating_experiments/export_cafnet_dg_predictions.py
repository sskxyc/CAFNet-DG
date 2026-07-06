from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CAFNET_D_DIR = ROOT / "result_ICS" / (
    "10cd3e100f10_CAFNetDecoupled_knn=5_wd=0.001_epoch=100_lamb=0.03_"
    "lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cross=True_"
    "fusion=gate_gate=new_fa=0.5_gatdrop=0.0_mix=0.3_aw=1.0_fw=1.0_"
    "rw=0.05_popw=0.1_biasw=1.0_listw=0.1_abw=1.0_arw=1.0_cosine"
)
CAFNET_DIR = ROOT / "result_ICS" / (
    "10ICS_CAFNet_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_"
    "eps=0.5_DF=False_PCA=False_not-FC=False_cosine"
)
OUT_DIR = ROOT / "result_ICS" / "10cafnet_dg_ensemble06_cafnetd04_cafnet"


def main():
    cafnet_d = pd.read_csv(CAFNET_D_DIR / "blind_pred.csv", header=None)
    cafnet = pd.read_csv(CAFNET_DIR / "blind_pred.csv", header=None)
    if cafnet_d.shape != cafnet.shape:
        raise ValueError(f"Shape mismatch: CAFNet-D {cafnet_d.shape}, CAFNet {cafnet.shape}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fused = 0.6 * cafnet_d + 0.4 * cafnet
    fused.to_csv(OUT_DIR / "blind_pred.csv", header=False, index=False)
    (OUT_DIR / "README.txt").write_text(
        "CAFNet-DG inference-time fusion.\n"
        "score = 0.6 * CAFNet-D full + 0.4 * CAFNet.\n"
        f"CAFNet-D source: {CAFNET_D_DIR}\n"
        f"CAFNet source: {CAFNET_DIR}\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT_DIR / 'blind_pred.csv'} with shape {fused.shape}")


if __name__ == "__main__":
    main()
