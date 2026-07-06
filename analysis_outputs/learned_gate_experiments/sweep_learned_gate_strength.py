from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import scipy.io as sio


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "learned_gate_experiments"
BASE_SCRIPT = OUT / "run_frozen_branch_learned_gate.py"

spec = importlib.util.spec_from_file_location("learned_gate", BASE_SCRIPT)
lg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(lg)


def main() -> None:
    rng = np.random.default_rng(20260701)
    raw = lg.load_mat_array(lg.RAW_FILE, "R").astype(float)
    masks = sio.loadmat(lg.MASK_FILE)
    labels = lg.get_fold_labels(raw, masks)
    prevalence = lg.get_fold_prevalence(raw, masks)
    d_parts = lg.split_cold(lg.read_pred(lg.CAFNET_D_FILE), masks)
    c_parts = lg.split_cold(lg.read_pred(lg.CAFNET_FILE), masks)
    fixed_parts = lg.split_cold(lg.read_pred(lg.FIXED_DG_FILE), masks)
    learned_gate_parts = lg.split_cold(pd.read_csv(OUT / "learned_gate_weights.csv", header=None).values.astype(np.float32), masks)

    rows = []
    strengths = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60]
    for fold in range(10):
        raw_gate = (learned_gate_parts[fold] - 0.20) / 0.60
        raw_gate = np.clip(raw_gate, 0.0, 1.0)
        for strength in strengths:
            gate = np.clip(0.60 + strength * (raw_gate - 0.50), 0.20, 0.80)
            score = gate * d_parts[fold] + (1.0 - gate) * c_parts[fold]
            m = lg.fold_metrics(score, labels[fold], prevalence[fold])
            m["matched_AUPR"] = lg.matched_control_aupr(score, labels[fold], prevalence[fold], rng)
            m["top100_removed_MAP"] = lg.hot_removed_map(score, labels[fold], prevalence[fold], 100)
            m.update(
                {
                    "model": f"learned_gate_shrink_{strength:g}",
                    "strength": strength,
                    "fold": fold,
                    "mean_gate": float(np.mean(gate)),
                    "rare_gate": float(np.mean(gate[:, prevalence[fold] <= np.quantile(prevalence[fold], 1 / 3)])),
                    "middle_gate": float(
                        np.mean(
                            gate[
                                :,
                                (prevalence[fold] > np.quantile(prevalence[fold], 1 / 3))
                                & (prevalence[fold] <= np.quantile(prevalence[fold], 2 / 3)),
                            ]
                        )
                    ),
                    "frequent_gate": float(np.mean(gate[:, prevalence[fold] > np.quantile(prevalence[fold], 2 / 3)])),
                }
            )
            rows.append(m)

        fixed = lg.fold_metrics(fixed_parts[fold], labels[fold], prevalence[fold])
        fixed["matched_AUPR"] = lg.matched_control_aupr(fixed_parts[fold], labels[fold], prevalence[fold], rng)
        fixed["top100_removed_MAP"] = lg.hot_removed_map(fixed_parts[fold], labels[fold], prevalence[fold], 100)
        fixed.update({"model": "CAFNet-DG-fixed", "strength": np.nan, "fold": fold, "mean_gate": 0.6, "rare_gate": 0.6, "middle_gate": 0.6, "frequent_gate": 0.6})
        rows.append(fixed)

        d = lg.fold_metrics(d_parts[fold], labels[fold], prevalence[fold])
        d["matched_AUPR"] = lg.matched_control_aupr(d_parts[fold], labels[fold], prevalence[fold], rng)
        d["top100_removed_MAP"] = lg.hot_removed_map(d_parts[fold], labels[fold], prevalence[fold], 100)
        d.update({"model": "CAFNet-D", "strength": np.nan, "fold": fold, "mean_gate": 1.0, "rare_gate": 1.0, "middle_gate": 1.0, "frequent_gate": 1.0})
        rows.append(d)

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(OUT / "learned_gate_strength_sweep_by_fold.csv", index=False)

    metric_cols = [
        "MAP",
        "AUROC",
        "AUPR",
        "nDCG@10",
        "P@15",
        "R@15",
        "rare_AP",
        "middle_AP",
        "matched_AUPR",
        "top100_removed_MAP",
        "mean_gate",
        "rare_gate",
        "middle_gate",
        "frequent_gate",
    ]
    summary = fold_df.groupby(["model", "strength"], dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.to_csv(OUT / "learned_gate_strength_sweep_summary.csv", index=False)

    fixed = fold_df[fold_df["model"] == "CAFNet-DG-fixed"]
    d = fold_df[fold_df["model"] == "CAFNet-D"]
    decision_rows = []
    for model, sub in fold_df[fold_df["model"].str.startswith("learned_gate")].groupby("model"):
        decision_rows.append(
            {
                "model": model,
                "strength": float(sub["strength"].iloc[0]),
                "MAP_mean": float(sub["MAP"].mean()),
                "mAP_loss_vs_fixed": float(fixed["MAP"].mean() - sub["MAP"].mean()),
                "rare_AP_gain_vs_CAFNet_D": float(sub["rare_AP"].mean() - d["rare_AP"].mean()),
                "middle_AP_gain_vs_CAFNet_D": float(sub["middle_AP"].mean() - d["middle_AP"].mean()),
                "matched_AUPR_gain_vs_CAFNet_D": float(sub["matched_AUPR"].mean() - d["matched_AUPR"].mean()),
                "top100_removed_MAP_gain_vs_CAFNet_D": float(sub["top100_removed_MAP"].mean() - d["top100_removed_MAP"].mean()),
                "mean_gate": float(sub["mean_gate"].mean()),
                "rare_gate": float(sub["rare_gate"].mean()),
                "middle_gate": float(sub["middle_gate"].mean()),
                "frequent_gate": float(sub["frequent_gate"].mean()),
            }
        )
    decision = pd.DataFrame(decision_rows).sort_values(["mAP_loss_vs_fixed", "rare_AP_gain_vs_CAFNet_D"], ascending=[True, False])
    decision["passes_primary_acceptance"] = (
        (decision["mAP_loss_vs_fixed"] <= 0.005)
        & (decision["rare_AP_gain_vs_CAFNet_D"] > 0)
        & (decision["middle_AP_gain_vs_CAFNet_D"] > 0)
        & (decision["matched_AUPR_gain_vs_CAFNet_D"] > 0)
        & (decision["top100_removed_MAP_gain_vs_CAFNet_D"] > 0)
    )
    decision.to_csv(OUT / "learned_gate_strength_decision_table.csv", index=False)
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()

