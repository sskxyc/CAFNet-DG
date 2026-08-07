from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "scientific_gap_resolution_20260807" / "determinism_audit"
RUNS = [ROOT / "result_ICS" / f"10DET8R{i}_CAFNetDecoupled" for i in range(1, 4)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = ["blind_pred.csv", "blind_freq_pred.csv", "CAFNetDecoupled_result.csv"]
    hash_rows = []
    for run_index, run in enumerate(RUNS, start=1):
        for name in files:
            path = run / name
            if not path.exists():
                raise FileNotFoundError(path)
            hash_rows.append({"run": run_index, "file": name, "sha256": sha256(path), "bytes": path.stat().st_size})
    hashes = pd.DataFrame(hash_rows)
    hashes.to_csv(OUT / "file_hashes.csv", index=False)

    comparison = []
    for name in ["blind_pred.csv", "blind_freq_pred.csv"]:
        arrays = [pd.read_csv(run / name, header=None).to_numpy(dtype=float) for run in RUNS]
        for right in range(1, len(arrays)):
            delta = np.abs(arrays[0] - arrays[right])
            comparison.append(
                {
                    "file": name,
                    "comparison": f"run1_vs_run{right + 1}",
                    "bitwise_equal_array": bool(np.array_equal(arrays[0], arrays[right])),
                    "max_abs_difference": float(delta.max()),
                    "mean_abs_difference": float(delta.mean()),
                    "n_different_cells": int(np.count_nonzero(delta)),
                }
            )
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(OUT / "numeric_comparison.csv", index=False)
    report = {
        "all_file_hashes_equal_by_type": bool(hashes.groupby("file")["sha256"].nunique().eq(1).all()),
        "all_prediction_arrays_bitwise_equal": bool(comparison_df["bitwise_equal_array"].all()),
        "max_abs_difference": float(comparison_df["max_abs_difference"].max()),
    }
    (OUT / "determinism_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
