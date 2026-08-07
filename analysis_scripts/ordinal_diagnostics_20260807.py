"""Compute fold-level ordinal diagnostics from non-overwriting CAFNet-D exports."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
LABELS = np.arange(1, 6, dtype=int)


def rounded_classes(pred: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(pred, dtype=float), 1.0, 5.0)
    return np.floor(clipped + 0.5).astype(int)


def metrics(y: np.ndarray, pred: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if y.size == 0 or y.size != pred.size:
        raise ValueError(f"Invalid arrays: labels={y.size}, predictions={pred.size}")
    if not np.all(np.isin(y, LABELS)):
        raise ValueError(f"Labels outside 1..5: {np.unique(y)}")
    if not np.all(np.isfinite(pred)):
        raise ValueError("Non-finite frequency predictions detected")

    y_int = y.astype(int)
    pred_cls = rounded_classes(pred)
    cm = confusion_matrix(y_int, pred_cls, labels=LABELS)
    recalls = np.divide(
        np.diag(cm), cm.sum(axis=1), out=np.full(5, np.nan), where=cm.sum(axis=1) > 0
    )
    rho = spearmanr(y, pred).statistic
    row = {
        "n": int(y.size),
        "qwk": float(cohen_kappa_score(y_int, pred_cls, weights="quadratic")),
        "within_one_accuracy": float(np.mean(np.abs(pred_cls - y_int) <= 1)),
        "exact_accuracy": float(np.mean(pred_cls == y_int)),
        "macro_recall": float(np.nanmean(recalls)),
        "rmse_unclipped": float(np.sqrt(np.mean((pred - y) ** 2))),
        "mae_unclipped": float(np.mean(np.abs(pred - y))),
        "spearman_unclipped": float(rho),
    }
    for idx, label in enumerate(LABELS):
        row[f"recall_class_{label}"] = float(recalls[idx])
        row[f"mae_class_{label}"] = float(np.mean(np.abs(pred[y_int == label] - label)))
        row[f"n_class_{label}"] = int(np.sum(y_int == label))
    return row, cm


def cold_arrays(result_dir: Path, raw: np.ndarray, masks: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    pred = pd.read_csv(result_dir / "blind_freq_pred.csv", header=None).to_numpy(float)
    saved_raw = pd.read_csv(result_dir / "blind_raw.csv", header=None).to_numpy(float)
    expected_rows = []
    fold_sizes = []
    for fold in range(10):
        mask = masks[f"mask{fold}"]
        test_idx = np.where(mask.sum(axis=1) == 0)[0]
        fold_sizes.append(len(test_idx))
        expected_rows.append(raw[test_idx])
    expected = np.vstack(expected_rows)
    if pred.shape != expected.shape or saved_raw.shape != expected.shape:
        raise ValueError(
            f"Cold export shape mismatch: pred={pred.shape}, raw={saved_raw.shape}, expected={expected.shape}"
        )
    if not np.array_equal(saved_raw, expected):
        mismatch = int(np.sum(saved_raw != expected))
        raise ValueError(f"Cold saved labels do not match fold masks ({mismatch} cells differ)")

    arrays = []
    offset = 0
    for size in fold_sizes:
        y_mat = saved_raw[offset : offset + size]
        p_mat = pred[offset : offset + size]
        observed = y_mat > 0
        arrays.append((y_mat[observed], p_mat[observed]))
        offset += size
    return arrays


def warm_arrays(result_dir: Path, raw: np.ndarray, masks: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    freq_dir = result_dir / "full_freq_predictions"
    paths = [freq_dir / f"full_freq_pred_fold{fold}.csv" for fold in range(10)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing warm full-frequency exports:\n" + "\n".join(missing))
    arrays = []
    for fold, path in enumerate(paths):
        pred = pd.read_csv(path, header=None).to_numpy(float)
        if pred.shape != raw.shape:
            raise ValueError(f"Warm fold {fold} shape {pred.shape}, expected {raw.shape}")
        heldout = (masks[f"mask{fold}"] == 0) & (raw > 0)
        if int(heldout.sum()) == 0:
            raise ValueError(f"Warm fold {fold} has no held-out observed labels")
        arrays.append((raw[heldout], pred[heldout]))
    return arrays


def summarize(scenario: str, arrays: list[tuple[np.ndarray, np.ndarray]], out: Path) -> None:
    rows = []
    cms = []
    all_y = []
    all_p = []
    for fold, (y, pred) in enumerate(arrays):
        row, cm = metrics(y, pred)
        row = {"scenario": scenario, "fold": fold, **row}
        rows.append(row)
        cms.append(cm)
        all_y.append(y)
        all_p.append(pred)
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(out / f"ordinal_{scenario}_per_fold.csv", index=False)

    metric_cols = [c for c in per_fold.columns if c not in {"scenario", "fold", "n"} and not c.startswith("n_class_")]
    summary_rows = []
    for col in metric_cols:
        summary_rows.append(
            {
                "scenario": scenario,
                "metric": col,
                "mean": per_fold[col].mean(),
                "std": per_fold[col].std(ddof=1),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out / f"ordinal_{scenario}_summary.csv", index=False)

    pooled_row, pooled_cm = metrics(np.concatenate(all_y), np.concatenate(all_p))
    pd.DataFrame([{"scenario": scenario, **pooled_row}]).to_csv(
        out / f"ordinal_{scenario}_pooled.csv", index=False
    )
    cm_df = pd.DataFrame(pooled_cm, index=[f"true_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS])
    cm_df.to_csv(out / f"ordinal_{scenario}_confusion_counts.csv")
    cm_norm = pooled_cm / np.maximum(pooled_cm.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(cm_norm, index=cm_df.index, columns=cm_df.columns).to_csv(
        out / f"ordinal_{scenario}_confusion_row_normalized.csv"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-dir", type=Path, required=True)
    parser.add_argument("--warm-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis_outputs" / "scientific_gap_resolution_20260807" / "ordinal_diagnostics",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = sio.loadmat(ROOT / "data" / "raw_frequency_750.mat")["R"]
    cold_masks = sio.loadmat(ROOT / "data" / "blind_mask_mat_750.mat")
    warm_masks = sio.loadmat(ROOT / "data" / "mask_mat_750.mat")
    summarize("cold", cold_arrays(args.cold_dir, raw, cold_masks), args.out_dir)
    summarize("warm", warm_arrays(args.warm_dir, raw, warm_masks), args.out_dir)

    cold = pd.read_csv(args.out_dir / "ordinal_cold_summary.csv")
    warm = pd.read_csv(args.out_dir / "ordinal_warm_summary.csv")
    wanted = ["qwk", "within_one_accuracy", "exact_accuracy", "macro_recall", "rmse_unclipped", "mae_unclipped", "spearman_unclipped"]
    report = ["# CAFNet-D Ordinal Diagnostics", "", "Labels 1-5 are evaluated; label 0 is treated as unknown and excluded.", ""]
    for scenario, frame in [("Cold", cold), ("Warm", warm)]:
        report.append(f"## {scenario}")
        report.append("")
        for metric in wanted:
            row = frame.loc[frame.metric == metric].iloc[0]
            report.append(f"- {metric}: {row['mean']:.4f} +/- {row['std']:.4f}")
        report.append("")
    (args.out_dir / "ordinal_diagnostics_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
