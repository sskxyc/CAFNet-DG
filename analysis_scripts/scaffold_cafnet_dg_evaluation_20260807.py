"""Evaluate CAFNet-D and fixed-rho CAFNet-DG on the frozen scaffold split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


ROOT = Path(r"D:\CAFNet-master-master")
OUT = ROOT / "analysis_outputs" / "scientific_gap_resolution_20260807" / "scaffold_cafnet_dg"
OUT.mkdir(parents=True, exist_ok=True)
RHO = 0.6

CAFNET_DIR = next((ROOT / "result_ICS").glob("10scaffold_CAFNet_knn=5_*"))
CAFNET_D_DIR = ROOT / "result_ICS" / "10SCFD8COLD_CAFNetDecoupled"


def load(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, header=None).to_numpy(float)


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    return np.nan if y.sum() in (0, len(y)) else float(average_precision_score(y, score))


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return np.nan if y.sum() in (0, len(y)) else float(roc_auc_score(y, score))


def drug_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if y.sum() == 0:
        return {"AP": np.nan, "AUROC": np.nan, "nDCG@10": np.nan, "P@1": np.nan, "P@15": np.nan, "R@15": np.nan}
    order = np.argsort(score)[::-1]
    top1 = order[:1]
    top15 = order[:15]
    return {
        "AP": safe_ap(y, score),
        "AUROC": safe_auc(y, score),
        "nDCG@10": float(ndcg_score(y[None, :], score[None, :], k=10)),
        "P@1": float(y[top1].mean()),
        "P@15": float(y[top15].mean()),
        "R@15": float(y[top15].sum() / y.sum()),
    }


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    n = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def split_rows(matrix: np.ndarray, masks: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    parts = []
    offset = 0
    for fold in range(10):
        test_ids = np.flatnonzero(masks[f"mask{fold}"][:, 0] == 0)
        n = len(test_ids)
        parts.append((test_ids, matrix[offset : offset + n]))
        offset += n
    if offset != len(matrix):
        raise ValueError(f"Consumed {offset} rows from matrix with {len(matrix)} rows")
    return parts


def prevalence_groups(raw: np.ndarray, train_ids: np.ndarray) -> dict[str, np.ndarray]:
    counts = (raw[train_ids] > 0).sum(axis=0)
    ordered = np.argsort(counts, kind="stable")
    rare, middle, frequent = np.array_split(ordered, 3)
    hot100 = np.argsort(counts)[::-1][:100]
    nonhot100 = np.setdiff1d(np.arange(raw.shape[1]), hot100, assume_unique=False)
    return {"rare": rare, "middle": middle, "frequent": frequent, "nonhot100": nonhot100}


def evaluate_model(model: str, parts: list[tuple[np.ndarray, np.ndarray]], raw: np.ndarray) -> pd.DataFrame:
    rows = []
    for fold, (test_ids, pred) in enumerate(parts):
        train_ids = np.setdiff1d(np.arange(raw.shape[0]), test_ids)
        groups = prevalence_groups(raw, train_ids)
        bucket: dict[str, list[float]] = {}
        y_all, s_all = [], []
        for local, drug_id in enumerate(test_ids):
            y = (raw[drug_id] > 0).astype(int)
            s = pred[local]
            y_all.append(y)
            s_all.append(s)
            for metric, value in drug_metrics(y, s).items():
                bucket.setdefault(metric, []).append(value)
            for group, cols in groups.items():
                value = drug_metrics(y[cols], s[cols])["AP"]
                bucket.setdefault(f"{group}_AP", []).append(value)
        row = {"model": model, "fold": fold}
        row.update({key: float(np.nanmean(values)) for key, values in bucket.items()})
        row["global_AUROC"] = safe_auc(np.concatenate(y_all), np.concatenate(s_all))
        row["global_AUPR"] = safe_ap(np.concatenate(y_all), np.concatenate(s_all))
        rows.append(row)
    return pd.DataFrame(rows)


def frequency_metrics(parts: list[tuple[np.ndarray, np.ndarray]], raw: np.ndarray) -> pd.DataFrame:
    rows = []
    for fold, (test_ids, pred) in enumerate(parts):
        labels = raw[test_ids]
        observed = labels > 0
        y = labels[observed]
        p = pred[observed]
        rows.append(
            {
                "model": "CAFNet-D frequency",
                "fold": fold,
                "Spearman": float(spearmanr(y, p).statistic),
                "RMSE": float(np.sqrt(np.mean((p - y) ** 2))),
                "MAE": float(np.mean(np.abs(p - y))),
            }
        )
    return pd.DataFrame(rows)


def similarity_audit(assignments: pd.DataFrame) -> pd.DataFrame:
    mols = [Chem.MolFromSmiles(str(x)) for x in assignments.smiles]
    if any(mol is None for mol in mols):
        raise ValueError("Invalid SMILES in scaffold assignment table")
    canonical = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in mols]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True) for mol in mols]
    rows = []
    for fold in range(10):
        test = assignments.loc[assignments.scaffold_fold == fold, "drug_index"].astype(int).to_numpy()
        train = assignments.loc[assignments.scaffold_fold != fold, "drug_index"].astype(int).to_numpy()
        for drug_id in test:
            similarities = DataStructs.BulkTanimotoSimilarity(fps[drug_id], [fps[x] for x in train])
            nearest_offset = int(np.argmax(similarities))
            nearest_id = int(train[nearest_offset])
            rows.append(
                {
                    "fold": fold,
                    "drug_index": drug_id,
                    "nearest_train_drug_index": nearest_id,
                    "max_train_tanimoto": float(similarities[nearest_offset]),
                    "exact_canonical_structure_match": bool(canonical[drug_id] == canonical[nearest_id]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    raw = sio.loadmat(ROOT / "data" / "raw_frequency_750.mat")["R"].astype(float)
    masks = sio.loadmat(ROOT / "data" / "scaffold_mask_mat_750.mat")
    assignments = pd.read_csv(ROOT / "analysis_outputs" / "scaffold_split_audit" / "drug_scaffold_assignments.csv")
    for fold in range(10):
        mask_ids = np.flatnonzero(masks[f"mask{fold}"][:, 0] == 0)
        assigned_ids = assignments.loc[assignments.scaffold_fold == fold, "drug_index"].astype(int).to_numpy()
        if not np.array_equal(np.sort(mask_ids), np.sort(assigned_ids)):
            raise ValueError(f"Scaffold assignment mismatch in fold {fold}")
        train_groups = set(assignments.loc[assignments.scaffold_fold != fold, "split_group"])
        test_groups = set(assignments.loc[assignments.scaffold_fold == fold, "split_group"])
        if train_groups & test_groups:
            raise ValueError(f"Scaffold leakage in fold {fold}")

    cafnet = load(CAFNET_DIR / "blind_pred.csv")
    cafnet_raw = load(CAFNET_DIR / "blind_raw.csv")
    cafnet_d = load(CAFNET_D_DIR / "blind_pred.csv")
    cafnet_d_freq = load(CAFNET_D_DIR / "blind_freq_pred.csv")
    cafnet_d_raw = load(CAFNET_D_DIR / "blind_raw.csv")
    if not np.array_equal(cafnet_raw, cafnet_d_raw):
        raise ValueError("CAFNet and CAFNet-D scaffold row labels differ")

    cafnet_parts = split_rows(cafnet, masks)
    cafnet_d_parts = split_rows(cafnet_d, masks)
    freq_parts = split_rows(cafnet_d_freq, masks)
    dg_parts = [
        (ids_c, RHO * pred_d + (1.0 - RHO) * pred_c)
        for (ids_c, pred_c), (ids_d, pred_d) in zip(cafnet_parts, cafnet_d_parts)
        if np.array_equal(ids_c, ids_d)
    ]
    if len(dg_parts) != 10:
        raise ValueError("Failed to align all scaffold folds for fusion")

    rank = pd.concat(
        [
            evaluate_model("CAFNet", cafnet_parts, raw),
            evaluate_model("CAFNet-D", cafnet_d_parts, raw),
            evaluate_model("CAFNet-DG", dg_parts, raw),
        ],
        ignore_index=True,
    )
    rank.to_csv(OUT / "scaffold_ranking_by_fold.csv", index=False)
    metric_cols = [c for c in rank.columns if c not in {"model", "fold"}]
    rank.groupby("model")[metric_cols].agg(["mean", "std"]).to_csv(OUT / "scaffold_ranking_summary.csv")

    tests = []
    for comparator in ["CAFNet", "CAFNet-D"]:
        a = rank[rank.model == "CAFNet-DG"].sort_values("fold")
        b = rank[rank.model == comparator].sort_values("fold")
        local = []
        for metric in metric_cols:
            delta = a[metric].to_numpy() - b[metric].to_numpy()
            try:
                p = float(wilcoxon(a[metric], b[metric], alternative="two-sided").pvalue)
            except ValueError:
                p = 1.0
            local.append({"comparison": f"CAFNet-DG vs {comparator}", "metric": metric, "mean_delta": float(np.nanmean(delta)), "p_raw": p})
        adjusted = holm([row["p_raw"] for row in local])
        for row, p_adj in zip(local, adjusted):
            row["p_holm"] = p_adj
            row["significant_holm"] = bool(p_adj < 0.05)
            tests.append(row)
    pd.DataFrame(tests).to_csv(OUT / "scaffold_ranking_paired_tests.csv", index=False)

    freq = frequency_metrics(freq_parts, raw)
    freq.to_csv(OUT / "scaffold_frequency_by_fold.csv", index=False)
    freq.groupby("model")[["Spearman", "RMSE", "MAE"]].agg(["mean", "std"]).to_csv(OUT / "scaffold_frequency_summary.csv")

    sim = similarity_audit(assignments)
    if sim["exact_canonical_structure_match"].any():
        raise ValueError("Exact canonical drug structure occurs across scaffold train/test folds")
    sim.to_csv(OUT / "scaffold_train_test_similarity_by_drug.csv", index=False)
    sim.groupby("fold")["max_train_tanimoto"].agg(["mean", "std", "median", "max"]).to_csv(
        OUT / "scaffold_train_test_similarity_summary.csv"
    )

    key = rank.groupby("model")[["AP", "global_AUROC", "global_AUPR", "nDCG@10", "rare_AP", "middle_AP", "frequent_AP", "nonhot100_AP"]].agg(["mean", "std"])
    report = [
        "# CAFNet-DG Scaffold-Disjoint Evaluation",
        "",
        "All 10 folds have zero overlap in the frozen split_group field. CAFNet-DG uses fixed rho=0.6 without scaffold-test tuning.",
        "",
        "```text",
        key.to_string(),
        "```",
        "",
        "The train-test similarity audit uses 2048-bit radius-2 Morgan fingerprints with chirality and is reported separately from the Bemis-Murcko split definition. Exact canonical-structure matches are also audited.",
    ]
    (OUT / "scaffold_cafnet_dg_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
