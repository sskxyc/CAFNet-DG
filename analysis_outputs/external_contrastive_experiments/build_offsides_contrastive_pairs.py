from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_external" / "external_pairs"
ANALYSIS_OUT = ROOT / "analysis_outputs" / "external_contrastive_experiments"
RAW_FILE = ROOT / "data" / "raw_frequency_750.mat"
MASK_FILE = ROOT / "data" / "blind_mask_mat_750.mat"
OFFSIDES_FILE = ROOT / "baselines" / "DSGAT-master" / "DSGAT-master" / "original_data" / "Supplementary Data 2.txt"


def mat_string(x):
    while isinstance(x, np.ndarray):
        if x.size == 1:
            x = x.item()
        else:
            x = x.flat[0]
    return str(x)


def norm(x: str) -> str:
    return str(x).replace(".", " ").replace("_", " ").replace("-", " ").lower().strip()


def load_vocab():
    raw = sio.loadmat(RAW_FILE)
    drugs = [mat_string(x) for x in raw["drugs"].flatten()]
    sides = [mat_string(x) for x in raw["sideeffects"].flatten()]
    drug_map = {norm(v): i for i, v in enumerate(drugs)}
    side_map = {norm(v): i for i, v in enumerate(sides)}
    return raw["R"].astype(float), drugs, sides, drug_map, side_map


def load_external_pairs(R, drugs, sides, drug_map, side_map):
    rows = []
    raw_rows = []
    with OFFSIDES_FILE.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            drug, side, source = row[0].strip(), row[1].strip(), row[2].strip()
            raw_rows.append({"drug": drug, "side_effect": side, "source": source})
            if source.upper() != "OFFSIDES":
                continue
            di = drug_map.get(norm(drug))
            si = side_map.get(norm(side))
            if di is None or si is None:
                continue
            rows.append(
                {
                    "drug_index": di,
                    "side_index": si,
                    "drug_name_external": drug,
                    "side_effect_external": side,
                    "drug_name_sider": drugs[di],
                    "side_effect_sider": sides[si],
                    "source": "OFFSIDES",
                    "is_sider_overlap": int(R[di, si] != 0),
                }
            )
    pairs = pd.DataFrame(rows).drop_duplicates(["drug_index", "side_index", "source"])
    raw_df = pd.DataFrame(raw_rows)
    return pairs, raw_df


def prevalence_bins(prevalence, n_bins=10):
    rank = pd.Series(prevalence).rank(method="first")
    return pd.qcut(rank, q=n_bins, labels=False, duplicates="drop").to_numpy()


def build_fold_contrastive(R, external_pairs, fold, n_neg=5, seed=20260630):
    rng = np.random.default_rng(seed + fold)
    masks = sio.loadmat(MASK_FILE)
    mask = masks[f"mask{fold}"].astype(float)
    train_bool = mask[:, 0] > 0
    train_idx = np.where(train_bool)[0]
    global_to_local = {int(g): i for i, g in enumerate(train_idx)}
    prevalence = (R[train_bool] > 0).mean(axis=0)
    bins = prevalence_bins(prevalence)

    ext_by_drug = {
        int(di): set(chunk["side_index"].astype(int).tolist())
        for di, chunk in external_pairs.groupby("drug_index")
    }
    fold_pos = external_pairs[external_pairs["drug_index"].isin(train_idx)].copy()
    rows = []
    for _, pair in fold_pos.iterrows():
        di = int(pair["drug_index"])
        si = int(pair["side_index"])
        local_di = global_to_local[di]
        excluded = set(np.where(R[di] != 0)[0].tolist())
        excluded.update(ext_by_drug.get(di, set()))
        pool = np.where((R[di] == 0) & (bins == bins[si]))[0]
        pool = np.array([x for x in pool if int(x) not in excluded], dtype=int)
        if len(pool) == 0:
            pool = np.where(R[di] == 0)[0]
            pool = np.array([x for x in pool if int(x) not in excluded], dtype=int)
        if len(pool) == 0:
            continue
        negs = rng.choice(pool, size=min(n_neg, len(pool)), replace=False)
        for neg in negs:
            rows.append(
                {
                    "fold": fold,
                    "drug_index": di,
                    "train_local_index": local_di,
                    "pos_side_index": si,
                    "neg_side_index": int(neg),
                    "pos_prevalence": float(prevalence[si]),
                    "neg_prevalence": float(prevalence[int(neg)]),
                    "source": pair["source"],
                    "is_sider_overlap": int(pair["is_sider_overlap"]),
                }
            )
    df = pd.DataFrame(rows)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    R, drugs, sides, drug_map, side_map = load_vocab()
    external_pairs, raw_df = load_external_pairs(R, drugs, sides, drug_map, side_map)
    external_pairs.to_csv(OUT / "offsides_positive_pairs_all.csv", index=False)
    external_pairs[external_pairs["is_sider_overlap"] == 0].to_csv(
        OUT / "offsides_positive_pairs_non_sider.csv", index=False
    )
    coverage = {
        "offsides_raw_rows": int((raw_df["source"].str.upper() == "OFFSIDES").sum()),
        "mapped_unique_offsides_pairs": int(len(external_pairs)),
        "mapped_drugs": int(external_pairs["drug_index"].nunique()),
        "mapped_side_effects": int(external_pairs["side_index"].nunique()),
        "sider_overlap_pairs": int(external_pairs["is_sider_overlap"].sum()),
        "non_sider_pairs": int((external_pairs["is_sider_overlap"] == 0).sum()),
    }
    pd.DataFrame([coverage]).to_csv(ANALYSIS_OUT / "offsides_mapping_coverage.csv", index=False)

    all_fold_rows = []
    for fold in range(10):
        df = build_fold_contrastive(R, external_pairs, fold)
        df.to_csv(OUT / f"offsides_contrastive_pairs_fold{fold}.csv", index=False)
        np.savez_compressed(
            OUT / f"offsides_contrastive_pairs_fold{fold}.npz",
            train_local_index=df["train_local_index"].to_numpy(dtype=np.int64),
            pos_side_index=df["pos_side_index"].to_numpy(dtype=np.int64),
            neg_side_index=df["neg_side_index"].to_numpy(dtype=np.int64),
            is_sider_overlap=df["is_sider_overlap"].to_numpy(dtype=np.int64),
        )
        all_fold_rows.append(df)
    all_pairs = pd.concat(all_fold_rows, ignore_index=True)
    all_pairs.to_csv(OUT / "offsides_contrastive_pairs_all_folds.csv", index=False)
    fold_summary = all_pairs.groupby("fold").agg(
        contrastive_pairs=("pos_side_index", "size"),
        unique_train_drugs=("drug_index", "nunique"),
        unique_positive_pairs=("pos_side_index", lambda x: len(set(zip(all_pairs.loc[x.index, "drug_index"], x)))),
        sider_overlap_rows=("is_sider_overlap", "sum"),
    )
    fold_summary.to_csv(ANALYSIS_OUT / "offsides_contrastive_fold_summary.csv")
    print("Coverage:", coverage)
    print(fold_summary)


if __name__ == "__main__":
    main()
