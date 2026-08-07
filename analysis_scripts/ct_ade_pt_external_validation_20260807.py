"""Independent CT-ADE-PT validation using exact structure and MedDRA-PT mapping.

The benchmark is never used for training or coefficient selection. Trial-group
labels are aggregated by drug with logical OR, then SIDER-observed pairs are
excluded. The resulting task is discrimination of clinically significant
(>=1% with 95% confidence in CT-ADE) ADR labels, not ordinal calibration.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from rdkit import Chem
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data_external" / "ct_ade_pt_2025" / "test.csv"
OUT = ROOT / "analysis_outputs" / "scientific_gap_resolution_20260807" / "ct_ade_pt_external_validation"
SIDE_TERMS = ROOT / "analysis_outputs" / "side_semantic_features" / "side_terms.json"
SMILES = ROOT / "data" / "drug_SMILES_750.txt"
RAW_FILE = ROOT / "data" / "raw_frequency_750.mat"
MASK_FILE = ROOT / "data" / "blind_mask_mat_750.mat"
MODEL_FILES = {
    "CAFNet": ROOT / "result_ICS" / (
        "10ICS_CAFNet_knn=5_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_"
        "eps=0.5_DF=False_PCA=False_not-FC=False_cosine"
    ) / "blind_pred.csv",
    "CAFNet-D": ROOT / "result_ICS" / (
        "10cd3e100f10_CAFNetDecoupled_knn=5_wd=0.001_epoch=100_lamb=0.03_"
        "lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cross=True_"
        "fusion=gate_gate=new_fa=0.5_gatdrop=0.0_mix=0.3_aw=1.0_fw=1.0_"
        "rw=0.05_popw=0.1_biasw=1.0_listw=0.1_abw=1.0_arw=1.0_cosine"
    ) / "blind_pred.csv",
    "CAFNet-DG": ROOT / "result_ICS" / "10cafnet_dg_ensemble06_cafnetd04_cafnet" / "blind_pred.csv",
}
RNG_SEED = 20260807


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles)) if str(smiles).strip() else None
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_drugs() -> tuple[list[str], list[str | None]]:
    names, structures = [], []
    with SMILES.open(encoding="utf-8") as handle:
        for line in handle:
            name, smiles = line.rstrip("\n").split(",", 1)
            names.append(name)
            structures.append(canonical(smiles))
    if len(names) != 750:
        raise ValueError(f"Expected 750 current drugs, found {len(names)}")
    return names, structures


def reconstruct_oof(path: Path, masks: dict, shape: tuple[int, int]) -> np.ndarray:
    sequential = pd.read_csv(path, header=None).to_numpy(float)
    if sequential.shape != shape:
        raise ValueError(f"Unexpected prediction shape for {path}: {sequential.shape}, expected {shape}")
    full = np.full(shape, np.nan, dtype=float)
    offset = 0
    for fold in range(10):
        test_ids = np.flatnonzero(masks[f"mask{fold}"][:, 0] == 0)
        block = sequential[offset : offset + len(test_ids)]
        full[test_ids] = block
        offset += len(test_ids)
    if offset != shape[0] or not np.isfinite(full).all():
        raise ValueError("Cold OOF predictions could not be reconstructed")
    return full


def training_prevalence(raw: np.ndarray, masks: dict, drug_id: int) -> np.ndarray:
    matches = [fold for fold in range(10) if masks[f"mask{fold}"][drug_id, 0] == 0]
    if len(matches) != 1:
        raise ValueError(f"Drug {drug_id} belongs to {len(matches)} cold test folds")
    train_ids = np.flatnonzero(masks[f"mask{matches[0]}"][:, 0] != 0)
    return (raw[train_ids] > 0).mean(axis=0)


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return np.nan if np.unique(y).size < 2 else float(roc_auc_score(y, score))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    return np.nan if y.sum() == 0 else float(average_precision_score(y, score))


def holm(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def bootstrap_drugs(rows: pd.DataFrame, models: list[str], repeats: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    drugs = rows.drug_index.unique()
    output = []
    for model in models:
        values = []
        for _ in range(repeats):
            sampled = rng.choice(drugs, size=len(drugs), replace=True)
            pieces = [rows[(rows.drug_index == drug)] for drug in sampled]
            frame = pd.concat(pieces, ignore_index=True)
            values.append((safe_auc(frame.label.to_numpy(), frame[model].to_numpy()), safe_ap(frame.label.to_numpy(), frame[model].to_numpy())))
        array = np.asarray(values, dtype=float)
        for idx, metric in enumerate(["AUROC", "AUPR"]):
            output.append(
                {
                    "model": model,
                    "metric": metric,
                    "estimate": safe_auc(rows.label.to_numpy(), rows[model].to_numpy()) if metric == "AUROC" else safe_ap(rows.label.to_numpy(), rows[model].to_numpy()),
                    "ci_low": float(np.nanquantile(array[:, idx], 0.025)),
                    "ci_high": float(np.nanquantile(array[:, idx], 0.975)),
                    "bootstrap_unit": "drug",
                    "repeats": repeats,
                }
            )
    return pd.DataFrame(output)


def summarize_scope(frame: pd.DataFrame, models: list[str], scope: str) -> list[dict]:
    output = []
    for model in models:
        per_drug_auc, per_drug_ap = [], []
        for _, drug in frame.groupby("drug_index"):
            if drug.label.nunique() < 2:
                continue
            per_drug_auc.append(safe_auc(drug.label.to_numpy(), drug[model].to_numpy()))
            per_drug_ap.append(safe_ap(drug.label.to_numpy(), drug[model].to_numpy()))
        for metric, values in [("AUROC", per_drug_auc), ("AUPR", per_drug_ap)]:
            pooled = safe_auc(frame.label.to_numpy(), frame[model].to_numpy()) if metric == "AUROC" else safe_ap(frame.label.to_numpy(), frame[model].to_numpy())
            output.append(
                {
                    "scope": scope,
                    "model": model,
                    "metric": metric,
                    "pooled": pooled,
                    "per_drug_mean": float(np.nanmean(values)) if values else np.nan,
                    "per_drug_std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan,
                    "n_evaluable_drugs": len(values),
                    "n_pairs": len(frame),
                    "n_positive": int(frame.label.sum()),
                }
            )
    return output


def prevalence_matched_rows(pairs: pd.DataFrame, models: list[str], controls_per_positive: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select nearest-prevalence same-drug controls without replacement when possible."""
    records, concordance = [], []
    pair_id = 0
    for drug_id, frame in pairs.groupby("drug_index"):
        positives = frame[frame.label == 1].sort_values(["train_prevalence", "side_index"])
        negatives = frame[frame.label == 0].copy()
        available = set(negatives.index.tolist())
        for pos_index, positive in positives.iterrows():
            candidates = negatives.loc[list(available)] if available else negatives
            same = candidates[candidates.prevalence_stratum == positive.prevalence_stratum]
            if len(same) < controls_per_positive:
                same = candidates
            ordered = same.assign(distance=(same.train_prevalence - positive.train_prevalence).abs()).sort_values(
                ["distance", "side_index"], kind="stable"
            )
            selected = ordered.head(controls_per_positive)
            if selected.empty:
                continue
            available.difference_update(selected.index.tolist())
            pos_row = positive.to_dict()
            pos_row.update({"match_id": pair_id, "role": "positive", "label": 1})
            records.append(pos_row)
            for _, control in selected.iterrows():
                neg_row = control.to_dict()
                neg_row.update({"match_id": pair_id, "role": "control", "label": 0})
                records.append(neg_row)
            c_row = {
                "match_id": pair_id,
                "drug_index": int(drug_id),
                "positive_side_index": int(positive.side_index),
                "n_controls": int(len(selected)),
                "mean_abs_prevalence_difference": float((selected.train_prevalence - positive.train_prevalence).abs().mean()),
            }
            for model in models:
                control_scores = selected[model].to_numpy(float)
                positive_score = float(positive[model])
                c_row[f"{model}_concordance"] = float(np.mean((positive_score > control_scores) + 0.5 * (positive_score == control_scores)))
            concordance.append(c_row)
            pair_id += 1
    return pd.DataFrame(records), pd.DataFrame(concordance)


def paired_scope_tests(frame: pd.DataFrame, scope: str, comparators: list[str]) -> list[dict]:
    drug_rows = []
    for drug_id, drug in frame.groupby("drug_index"):
        if drug.label.nunique() < 2:
            continue
        row = {"drug_index": int(drug_id)}
        for model in ["CAFNet-DG"] + comparators:
            row[f"{model}_AUROC"] = safe_auc(drug.label.to_numpy(), drug[model].to_numpy())
            row[f"{model}_AUPR"] = safe_ap(drug.label.to_numpy(), drug[model].to_numpy())
        drug_rows.append(row)
    per_drug = pd.DataFrame(drug_rows)
    output = []
    for comparator in comparators:
        local = []
        for metric in ["AUROC", "AUPR"]:
            a = per_drug[f"CAFNet-DG_{metric}"]
            b = per_drug[f"{comparator}_{metric}"]
            valid = a.notna() & b.notna()
            delta = (a[valid] - b[valid]).to_numpy()
            if len(delta) == 0 or np.allclose(delta, 0):
                statistic, p = 0.0, 1.0
            else:
                test = stats.wilcoxon(delta, alternative="two-sided")
                statistic, p = float(test.statistic), float(test.pvalue)
            local.append(
                {
                    "scope": scope,
                    "comparison": f"CAFNet-DG vs {comparator}",
                    "metric": metric,
                    "n_drugs": int(len(delta)),
                    "mean_delta": float(np.mean(delta)) if len(delta) else np.nan,
                    "wilcoxon_statistic": statistic,
                    "p_raw": p,
                }
            )
        for row, adjusted in zip(local, holm([x["p_raw"] for x in local])):
            row["p_holm"] = adjusted
            row["significant_holm"] = bool(adjusted < 0.05)
            output.append(row)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    with SOURCE.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    label_by_norm: dict[str, str] = {}
    duplicates = []
    for column in header:
        if not column.startswith("label_"):
            continue
        key = norm(column[6:])
        if key in label_by_norm:
            duplicates.append(key)
        label_by_norm[key] = column
    if duplicates:
        raise ValueError(f"Normalized duplicate CT-ADE labels: {duplicates[:10]}")

    side_terms = json.loads(SIDE_TERMS.read_text(encoding="utf-8"))
    side_matches = [(idx, term, label_by_norm[norm(term)]) for idx, term in enumerate(side_terms) if norm(term) in label_by_norm]
    if len(side_matches) < 900:
        raise ValueError(f"Insufficient exact PT coverage: {len(side_matches)}/994")
    usecols = ["nctid", "group_id", "intervention_name", "smiles"] + [x[2] for x in side_matches]
    ct = pd.read_csv(SOURCE, usecols=usecols)

    drug_names, structures = current_drugs()
    structure_to_ids: dict[str, list[int]] = {}
    for idx, structure in enumerate(structures):
        if structure:
            structure_to_ids.setdefault(structure, []).append(idx)
    ct["canonical_smiles"] = [canonical(value) for value in ct.smiles.fillna("")]
    ct["drug_index"] = [
        structure_to_ids.get(value, [None])[0] if value and len(structure_to_ids.get(value, [])) == 1 else None
        for value in ct.canonical_smiles
    ]
    mapped = ct.dropna(subset=["drug_index"]).copy()
    mapped["drug_index"] = mapped.drug_index.astype(int)
    if mapped.drug_index.nunique() < 20:
        raise ValueError(f"Insufficient exact drug coverage: {mapped.drug_index.nunique()}")

    label_columns = [x[2] for x in side_matches]
    grouped = mapped.groupby("drug_index", sort=True)[label_columns].max().astype(int)
    raw = sio.loadmat(RAW_FILE)["R"].astype(float)
    masks = sio.loadmat(MASK_FILE)
    models = {name: reconstruct_oof(path, masks, raw.shape) for name, path in MODEL_FILES.items()}

    pair_rows = []
    stratum_counts = {"rare": 0, "middle": 0, "frequent": 0}
    for drug_id, labels in grouped.iterrows():
        prevalence = training_prevalence(raw, masks, int(drug_id))
        ordered = np.argsort(prevalence, kind="stable")
        rare, middle, frequent = np.array_split(ordered, 3)
        stratum = np.full(raw.shape[1], "frequent", dtype=object)
        stratum[rare] = "rare"
        stratum[middle] = "middle"
        for side_id, term, ct_column in side_matches:
            if raw[int(drug_id), side_id] > 0:
                continue
            label = int(labels[ct_column])
            row = {
                "drug_index": int(drug_id),
                "drug_name": drug_names[int(drug_id)],
                "side_index": side_id,
                "side_effect": term,
                "label": label,
                "train_prevalence": float(prevalence[side_id]),
                "prevalence_stratum": stratum[side_id],
            }
            for model, matrix in models.items():
                row[model] = float(matrix[int(drug_id), side_id])
            row["Global popularity"] = float(prevalence[side_id])
            pair_rows.append(row)
            if label:
                stratum_counts[stratum[side_id]] += 1
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT / "ct_ade_non_sider_pairs.csv", index=False)

    score_models = list(MODEL_FILES) + ["Global popularity"]
    per_drug = []
    for drug_id, frame in pairs.groupby("drug_index"):
        if frame.label.nunique() < 2:
            continue
        row = {
            "drug_index": int(drug_id),
            "drug_name": frame.drug_name.iloc[0],
            "n_pairs": len(frame),
            "n_positive": int(frame.label.sum()),
        }
        for model in score_models:
            row[f"{model}_AUROC"] = safe_auc(frame.label.to_numpy(), frame[model].to_numpy())
            row[f"{model}_AUPR"] = safe_ap(frame.label.to_numpy(), frame[model].to_numpy())
        per_drug.append(row)
    per_drug_df = pd.DataFrame(per_drug)
    per_drug_df.to_csv(OUT / "ct_ade_per_drug_metrics.csv", index=False)

    summary = []
    for model in score_models:
        for metric in ["AUROC", "AUPR"]:
            column = f"{model}_{metric}"
            summary.append(
                {
                    "model": model,
                    "metric": metric,
                    "pooled": safe_auc(pairs.label.to_numpy(), pairs[model].to_numpy()) if metric == "AUROC" else safe_ap(pairs.label.to_numpy(), pairs[model].to_numpy()),
                    "per_drug_mean": float(per_drug_df[column].mean()),
                    "per_drug_std": float(per_drug_df[column].std(ddof=1)),
                    "n_drugs": int(per_drug_df[column].notna().sum()),
                }
            )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "ct_ade_summary.csv", index=False)
    bootstrap_drugs(pairs, score_models).to_csv(OUT / "ct_ade_drug_bootstrap.csv", index=False)

    stratified = []
    stratified.extend(summarize_scope(pairs, score_models, "all_non_sider"))
    for stratum in ["rare", "middle", "frequent"]:
        stratified.extend(
            summarize_scope(pairs[pairs.prevalence_stratum == stratum], score_models, stratum)
        )
    stratified_df = pd.DataFrame(stratified)
    stratified_df.to_csv(OUT / "ct_ade_prevalence_stratified_summary.csv", index=False)

    matched, concordance = prevalence_matched_rows(pairs, score_models)
    matched.to_csv(OUT / "ct_ade_prevalence_matched_pairs.csv", index=False)
    concordance.to_csv(OUT / "ct_ade_prevalence_matched_concordance.csv", index=False)
    matched_summary = summarize_scope(matched, score_models, "prevalence_matched_1_to_5")
    for model in score_models:
        matched_summary.append(
            {
                "scope": "prevalence_matched_1_to_5",
                "model": model,
                "metric": "concordance",
                "pooled": float(concordance[f"{model}_concordance"].mean()),
                "per_drug_mean": float(concordance.groupby("drug_index")[f"{model}_concordance"].mean().mean()),
                "per_drug_std": float(concordance.groupby("drug_index")[f"{model}_concordance"].mean().std(ddof=1)),
                "n_evaluable_drugs": int(concordance.drug_index.nunique()),
                "n_pairs": int(len(matched)),
                "n_positive": int((matched.label == 1).sum()),
            }
        )
    matched_summary_df = pd.DataFrame(matched_summary)
    matched_summary_df.to_csv(OUT / "ct_ade_prevalence_matched_summary.csv", index=False)

    scope_tests = []
    for scope, frame in [
        ("all_non_sider", pairs),
        ("rare", pairs[pairs.prevalence_stratum == "rare"]),
        ("middle", pairs[pairs.prevalence_stratum == "middle"]),
        ("frequent", pairs[pairs.prevalence_stratum == "frequent"]),
        ("prevalence_matched_1_to_5", matched),
    ]:
        scope_tests.extend(paired_scope_tests(frame, scope, ["CAFNet-D", "CAFNet", "Global popularity"]))
    pd.DataFrame(scope_tests).to_csv(OUT / "ct_ade_stratified_paired_tests.csv", index=False)

    tests = []
    for comparator in ["CAFNet-D", "CAFNet", "Global popularity"]:
        local = []
        for metric in ["AUROC", "AUPR"]:
            a = per_drug_df[f"CAFNet-DG_{metric}"]
            b = per_drug_df[f"{comparator}_{metric}"]
            valid = a.notna() & b.notna()
            delta = (a[valid] - b[valid]).to_numpy()
            if len(delta) == 0 or np.allclose(delta, 0):
                statistic, p = 0.0, 1.0
            else:
                test = stats.wilcoxon(delta, alternative="two-sided")
                statistic, p = float(test.statistic), float(test.pvalue)
            local.append(
                {
                    "comparison": f"CAFNet-DG vs {comparator}",
                    "metric": metric,
                    "n_drugs": int(len(delta)),
                    "mean_delta": float(np.mean(delta)) if len(delta) else np.nan,
                    "wilcoxon_statistic": statistic,
                    "p_raw": p,
                }
            )
        for row, adjusted in zip(local, holm([x["p_raw"] for x in local])):
            row["p_holm"] = adjusted
            row["significant_holm"] = bool(adjusted < 0.05)
            tests.append(row)
    pd.DataFrame(tests).to_csv(OUT / "ct_ade_paired_tests.csv", index=False)

    mapping = mapped[["nctid", "group_id", "intervention_name", "smiles", "canonical_smiles", "drug_index"]].copy()
    mapping["current_drug_name"] = [drug_names[x] for x in mapping.drug_index]
    mapping.to_csv(OUT / "ct_ade_exact_drug_mapping.csv", index=False)
    pd.DataFrame(side_matches, columns=["side_index", "current_side_effect", "ct_ade_column"]).to_csv(
        OUT / "ct_ade_exact_side_mapping.csv", index=False
    )

    protocol = {
        "source": "CT-ADE-PT public test split from ClinicalTrials.gov monopharmacy results",
        "source_url": "https://huggingface.co/datasets/anthonyyazdaniml/CT-ADE-PT",
        "source_sha256": sha256(SOURCE),
        "ct_test_rows": int(len(ct)),
        "mapped_trial_groups": int(len(mapped)),
        "mapped_drugs": int(mapped.drug_index.nunique()),
        "exact_side_terms": int(len(side_matches)),
        "non_sider_pairs": int(len(pairs)),
        "non_sider_positives": int(pairs.label.sum()),
        "positive_drugs": int(pairs.loc[pairs.label == 1, "drug_index"].nunique()),
        "positive_prevalence_strata": stratum_counts,
        "matched_rows": int(len(matched)),
        "matched_positive_sets": int(len(concordance)),
        "matched_median_abs_prevalence_difference": float(concordance.mean_abs_prevalence_difference.median()),
        "label_definition": "CT-ADE label 1: 95% confidence that at least 1% of the trial group experiences the ADE",
        "aggregation": "logical OR over mapped trial groups for each drug",
        "exclusion": "all drug-side-effect pairs observed in the SIDER-derived training matrix",
        "prediction": "frozen cold-start out-of-fold model score; CT-ADE never used for training or selection",
        "claim_boundary": "independent controlled-trial ADR prioritization, not ordinal-frequency calibration, incidence estimation, or causal proof",
    }
    (OUT / "coverage_and_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    report = [
        "# CT-ADE-PT independent clinical-trial validation",
        "",
        f"- Exact structure mapping: {protocol['mapped_drugs']} drugs across {protocol['mapped_trial_groups']} trial groups.",
        f"- Exact MedDRA-PT mapping: {protocol['exact_side_terms']}/994 side effects.",
        f"- After SIDER exclusion: {protocol['non_sider_positives']} positives over {protocol['positive_drugs']} drugs.",
        "- The outcome is CT-ADE's clinically significant >=1% label, not the original 1--5 ordinal-frequency target.",
        "",
        "```text",
        summary_df.to_string(index=False),
        "```",
        "",
        "## Prevalence-controlled results",
        "",
        "```text",
        stratified_df.to_string(index=False),
        "```",
        "",
        "```text",
        matched_summary_df.to_string(index=False),
        "```",
        "",
        "## Interpretation boundary",
        "",
        "This analysis is an independent controlled-monopharmacy clinical-trial prioritization test. "
        "Because CAFNet-DG has no patient or regimen input, repeated trial groups are aggregated by drug. "
        "The result must not be described as patient-specific prediction, calibrated incidence, or causal confirmation.",
    ]
    (OUT / "ct_ade_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
