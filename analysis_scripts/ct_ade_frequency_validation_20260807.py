"""External ordinal-frequency diagnostics on CT-ADE-PT trial proportions."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import cohen_kappa_score, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
CT_DIR = ROOT / "data_external" / "ct_ade_pt_2025"
OUT = ROOT / "analysis_outputs" / "scientific_gap_resolution_20260807" / "ct_ade_frequency_validation"
MAP_DIR = ROOT / "analysis_outputs" / "scientific_gap_resolution_20260807" / "ct_ade_pt_external_validation"
SIDE_TERMS = ROOT / "analysis_outputs" / "side_semantic_features" / "side_terms.json"
FREQ_PRED = ROOT / "result_ICS" / "10ORD8COLD2_CAFNetDecoupled" / "blind_freq_pred.csv"
LABELS = np.arange(1, 6, dtype=int)


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruct_oof(path: Path, masks: dict, shape: tuple[int, int]) -> np.ndarray:
    sequential = pd.read_csv(path, header=None).to_numpy(float)
    if sequential.shape != shape:
        raise ValueError(f"Unexpected frequency prediction shape: {sequential.shape}")
    full = np.full(shape, np.nan, dtype=float)
    offset = 0
    for fold in range(10):
        test_ids = np.flatnonzero(masks[f"mask{fold}"][:, 0] == 0)
        full[test_ids] = sequential[offset : offset + len(test_ids)]
        offset += len(test_ids)
    if not np.isfinite(full).all():
        raise ValueError("Incomplete OOF reconstruction")
    return full


def frequency_class(proportion: np.ndarray) -> np.ndarray:
    """EU/CIOMS bands: <0.01%, 0.01-0.1%, 0.1-1%, 1-10%, >=10%."""
    bins = np.array([0.0001, 0.001, 0.01, 0.1], dtype=float)
    return np.digitize(proportion, bins, right=False) + 1


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred_class = np.floor(np.clip(pred, 1.0, 5.0) + 0.5).astype(int)
    return {
        "n": int(len(y)),
        "qwk": float(cohen_kappa_score(y, pred_class, labels=LABELS, weights="quadratic")),
        "within_one_accuracy": float(np.mean(np.abs(pred_class - y) <= 1)),
        "exact_accuracy": float(np.mean(pred_class == y)),
        "spearman": float(spearmanr(y, pred).statistic),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "mae": float(np.mean(np.abs(pred - y))),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels_file = CT_DIR / "test.csv"
    frequencies_file = CT_DIR / "test_frequencies.csv"
    if not labels_file.exists() or not frequencies_file.exists():
        raise FileNotFoundError("Both CT-ADE test files are required")

    with frequencies_file.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    freq_by_norm = {norm(column[10:]): column for column in header if column.startswith("frequency_")}
    side_terms = json.loads(SIDE_TERMS.read_text(encoding="utf-8"))
    side_matches = [(idx, term, freq_by_norm[norm(term)]) for idx, term in enumerate(side_terms) if norm(term) in freq_by_norm]

    mapping = pd.read_csv(MAP_DIR / "ct_ade_exact_drug_mapping.csv").drop_duplicates(["nctid", "group_id"])
    metadata = pd.read_csv(labels_file, usecols=["nctid", "group_id", "ade_num_at_risk"])
    mapping = mapping.merge(metadata, on=["nctid", "group_id"], validate="one_to_one")
    frequency_columns = [x[2] for x in side_matches]
    frequencies = pd.read_csv(frequencies_file, usecols=["nctid", "group_id"] + frequency_columns)
    frame = mapping.merge(frequencies, on=["nctid", "group_id"], validate="one_to_one")
    frame["ade_num_at_risk"] = pd.to_numeric(frame.ade_num_at_risk, errors="coerce").fillna(0).clip(lower=0)

    raw = sio.loadmat(ROOT / "data" / "raw_frequency_750.mat")["R"].astype(float)
    masks = sio.loadmat(ROOT / "data" / "blind_mask_mat_750.mat")
    predictions = reconstruct_oof(FREQ_PRED, masks, raw.shape)

    rows = []
    for drug_id, drug in frame.groupby("drug_index"):
        drug_id = int(drug_id)
        weights = drug.ade_num_at_risk.to_numpy(float)
        if weights.sum() <= 0:
            weights = np.ones(len(drug), dtype=float)
        fold = [x for x in range(10) if masks[f"mask{x}"][drug_id, 0] == 0]
        if len(fold) != 1:
            raise ValueError(f"Cannot identify unique fold for drug {drug_id}")
        train_ids = np.flatnonzero(masks[f"mask{fold[0]}"][:, 0] != 0)
        train = raw[train_ids]
        observed = train > 0
        global_mean = float(train[observed].mean())
        side_count = observed.sum(axis=0)
        side_mean = np.divide(train.sum(axis=0), side_count, out=np.full(raw.shape[1], global_mean), where=side_count > 0)
        for side_id, term, column in side_matches:
            if raw[drug_id, side_id] > 0:
                continue
            values = drug[column].to_numpy(float)
            finite = np.isfinite(values)
            if not np.any(finite):
                continue
            local_weights = weights[finite]
            if local_weights.sum() <= 0:
                local_weights = np.ones(int(finite.sum()), dtype=float)
            proportion = float(np.average(values[finite], weights=local_weights))
            if not np.isfinite(proportion) or proportion <= 0:
                continue
            rows.append(
                {
                    "drug_index": drug_id,
                    "drug_name": drug.current_drug_name.iloc[0],
                    "side_index": side_id,
                    "side_effect": term,
                    "ct_ade_weighted_proportion": proportion,
                    "ct_ade_ordinal_class": int(frequency_class(np.array([proportion]))[0]),
                    "CAFNet-D_frequency": float(predictions[drug_id, side_id]),
                    "Training-side-mean": float(side_mean[side_id]),
                }
            )
    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT / "ct_ade_external_frequency_pairs.csv", index=False)
    if len(pairs) < 100:
        raise ValueError(f"Insufficient non-SIDER external frequency pairs: {len(pairs)}")

    models = ["CAFNet-D_frequency", "Training-side-mean"]
    summary = []
    for model in models:
        row = {"model": model, **metrics(pairs.ct_ade_ordinal_class.to_numpy(int), pairs[model].to_numpy(float))}
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT / "ct_ade_external_frequency_summary.csv", index=False)

    per_drug = []
    for drug_id, drug in pairs.groupby("drug_index"):
        if len(drug) < 3 or drug.ct_ade_ordinal_class.nunique() < 2:
            continue
        row = {"drug_index": int(drug_id), "drug_name": drug.drug_name.iloc[0], "n": len(drug)}
        for model in models:
            for key, value in metrics(drug.ct_ade_ordinal_class.to_numpy(int), drug[model].to_numpy(float)).items():
                row[f"{model}_{key}"] = value
        per_drug.append(row)
    per_drug_df = pd.DataFrame(per_drug)
    per_drug_df.to_csv(OUT / "ct_ade_external_frequency_per_drug.csv", index=False)

    tests = []
    for metric in ["qwk", "within_one_accuracy", "exact_accuracy", "spearman", "rmse", "mae"]:
        a = per_drug_df[f"CAFNet-D_frequency_{metric}"]
        b = per_drug_df[f"Training-side-mean_{metric}"]
        delta = (a - b).dropna().to_numpy()
        if np.allclose(delta, 0):
            statistic, p = 0.0, 1.0
        else:
            test = wilcoxon(delta, alternative="two-sided")
            statistic, p = float(test.statistic), float(test.pvalue)
        tests.append({"metric": metric, "n_drugs": len(delta), "mean_delta_model_minus_baseline": float(delta.mean()), "wilcoxon_statistic": statistic, "p_raw": p})
    p = np.asarray([row["p_raw"] for row in tests])
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adjusted[idx] = running
    for row, value in zip(tests, adjusted):
        row["p_holm"] = float(value)
        row["significant_holm"] = bool(value < 0.05)
    pd.DataFrame(tests).to_csv(OUT / "ct_ade_external_frequency_paired_tests.csv", index=False)

    y = pairs.ct_ade_ordinal_class.to_numpy(int)
    for model in models:
        pred_class = np.floor(np.clip(pairs[model].to_numpy(float), 1, 5) + 0.5).astype(int)
        cm = confusion_matrix(y, pred_class, labels=LABELS)
        pd.DataFrame(cm, index=[f"true_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS]).to_csv(
            OUT / f"{model}_confusion_counts.csv"
        )
        normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        pd.DataFrame(normalized, index=[f"true_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS]).to_csv(
            OUT / f"{model}_confusion_row_normalized.csv"
        )

    protocol = {
        "source": "CT-ADE-PT public test_frequencies split",
        "source_url": "https://huggingface.co/datasets/anthonyyazdaniml/CT-ADE-PT",
        "source_sha256": sha256(frequencies_file),
        "mapped_trial_groups": int(len(frame)),
        "exact_side_terms": int(len(side_matches)),
        "non_sider_nonzero_frequency_pairs": int(len(pairs)),
        "mapped_drugs_with_frequency_pairs": int(pairs.drug_index.nunique()),
        "aggregation": "at-risk weighted mean trial proportion by drug-side-effect pair",
        "class_bands": {"1": "<0.01%", "2": "0.01-0.1%", "3": "0.1-1%", "4": "1-10%", "5": ">=10%"},
        "claim_boundary": "external ordinal calibration diagnostic; no patient-specific or causal interpretation",
    }
    (OUT / "coverage_and_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    report = [
        "# CT-ADE external ordinal-frequency diagnostics",
        "",
        json.dumps(protocol, indent=2),
        "",
        "```text",
        pd.DataFrame(summary).to_string(index=False),
        "```",
    ]
    (OUT / "ct_ade_external_frequency_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
