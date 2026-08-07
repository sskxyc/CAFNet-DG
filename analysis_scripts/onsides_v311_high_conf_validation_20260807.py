from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from fda_aems_signal_validation_20260807 import (
    MODEL_FILES,
    ROOT,
    drug_bootstrap,
    labels,
    matched_controls,
    normalize,
    paired_tests,
    read_predictions,
    score_controls,
    summaries,
)


DATA = ROOT / "data_external" / "onsides_v3.1.1" / "release_data" / "extracted" / "csv"
OUT = ROOT / "analysis_outputs" / "scientific_gap_resolution_20260807" / "onsides_v311_high_confidence"


def build_exact_mapping() -> pd.DataFrame:
    drug_names, side_names = labels()
    drug_lookup = {normalize(name): index for index, name in enumerate(drug_names)}
    side_lookup = {normalize(name): index for index, name in enumerate(side_names)}

    ingredients = pd.read_csv(DATA / "vocab_rxnorm_ingredient.csv", dtype=str)
    ingredients["drug_index"] = ingredients.rxnorm_name.map(lambda value: drug_lookup.get(normalize(value)))
    ingredients = ingredients.dropna(subset=["drug_index"]).copy()
    ingredients["drug_index"] = ingredients.drug_index.astype(int)

    meddra = pd.read_csv(DATA / "vocab_meddra_adverse_effect.csv", dtype=str)
    meddra = meddra[meddra.meddra_term_type == "PT"].copy()
    meddra["side_index"] = meddra.meddra_name.map(lambda value: side_lookup.get(normalize(value)))
    meddra = meddra.dropna(subset=["side_index"]).copy()
    meddra["side_index"] = meddra.side_index.astype(int)

    high_confidence = pd.read_csv(DATA / "high_confidence.csv", dtype=str)
    mapped = high_confidence.merge(
        ingredients[["rxnorm_id", "rxnorm_name", "drug_index"]],
        left_on="ingredient_id",
        right_on="rxnorm_id",
        validate="many_to_one",
    ).merge(
        meddra[["meddra_id", "meddra_name", "side_index"]],
        left_on="effect_meddra_id",
        right_on="meddra_id",
        validate="many_to_one",
    )
    mapped["drug_name_benchmark"] = mapped.drug_index.map(dict(enumerate(drug_names)))
    mapped["side_effect_benchmark"] = mapped.side_index.map(dict(enumerate(side_names)))
    mapped["mapping_rule"] = "exact normalized RxNorm ingredient and exact normalized MedDRA PT"
    return mapped.drop_duplicates(["drug_index", "side_index"])


def annotate_prevalence_conditions(controls: pd.DataFrame, raw: np.ndarray, masks: dict) -> pd.DataFrame:
    labels_binary = (raw != 0).astype(int)
    annotated = controls.copy()
    annotated["positive_prevalence_group"] = ""
    annotated["positive_is_top100"] = False
    annotated["control_is_top100"] = False
    for fold in range(10):
        index = annotated.index[annotated.fold == fold]
        if len(index) == 0:
            continue
        train = masks[f"mask{fold}"][:, 0] != 0
        prevalence = labels_binary[train].sum(axis=0)
        rare, middle, frequent = np.array_split(np.argsort(prevalence, kind="stable"), 3)
        group = np.empty(raw.shape[1], dtype=object)
        group[rare], group[middle], group[frequent] = "rare", "middle", "frequent"
        hot100 = set(np.argsort(-prevalence, kind="stable")[:100].tolist())
        annotated.loc[index, "positive_prevalence_group"] = annotated.loc[index, "positive_side_index"].map(
            lambda value: group[int(value)]
        )
        annotated.loc[index, "positive_is_top100"] = annotated.loc[index, "positive_side_index"].map(
            lambda value: int(value) in hot100
        )
        annotated.loc[index, "control_is_top100"] = annotated.loc[index, "control_side_index"].map(
            lambda value: int(value) in hot100
        )
    return annotated


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mapped = build_exact_mapping()
    raw = sio.loadmat(ROOT / "data" / "raw_frequency_750.mat")["R"].astype(float)
    mapped["known_in_sider_matrix"] = raw[mapped.drug_index, mapped.side_index] != 0
    mapped.to_csv(OUT / "exact_mapping_audit_with_sider_overlap.csv", index=False)
    positives = mapped[~mapped.known_in_sider_matrix].copy()
    positives.to_csv(OUT / "high_confidence_positive_pairs_non_sider.csv", index=False)

    masks = sio.loadmat(ROOT / "data" / "blind_mask_mat_750.mat")
    controls = annotate_prevalence_conditions(matched_controls(positives, raw, masks), raw, masks)
    controls.to_csv(OUT / "prevalence_matched_controls.csv", index=False)
    predictions = {name: read_predictions(path, masks) for name, path in MODEL_FILES.items()}
    scored = score_controls(controls, predictions, raw, masks)
    scored.to_csv(OUT / "scored_positive_control_rows.csv", index=False)

    global_summary, per_drug = summaries(scored)
    global_summary.to_csv(OUT / "global_summary.csv", index=False)
    per_drug.to_csv(OUT / "per_drug_summary.csv", index=False)
    paired_tests(per_drug).to_csv(OUT / "paired_tests.csv", index=False)
    drug_bootstrap(scored).to_csv(OUT / "drug_bootstrap_ci.csv", index=False)
    condition_rows = []
    conditions = {
        "all": np.ones(len(scored), dtype=bool),
        "rare": scored.positive_prevalence_group == "rare",
        "middle": scored.positive_prevalence_group == "middle",
        "frequent": scored.positive_prevalence_group == "frequent",
        "nonhot100": (~scored.positive_is_top100) & (~scored.control_is_top100),
    }
    for condition, keep in conditions.items():
        subset = scored.loc[keep].copy()
        if subset.empty:
            continue
        condition_summary, _ = summaries(subset)
        condition_summary.insert(0, "condition", condition)
        condition_rows.append(condition_summary)
    pd.concat(condition_rows, ignore_index=True).to_csv(OUT / "prevalence_condition_summary.csv", index=False)

    metadata = {
        "source": "OnSIDES v3.1.1 high_confidence.csv",
        "release_date": "2026-04-22",
        "release_asset_md5": "ab934e47108f2f53b3d7107405fa93d3",
        "high_confidence_definition": "ingredient-effect pair observed across all four label sources (US, UK, EU, JP)",
        "mapping": "exact normalized RxNorm ingredient and exact normalized MedDRA PT",
        "mapped_pair_count_before_sider_exclusion": int(len(mapped)),
        "non_sider_external_positive_count": int(len(positives)),
        "mapped_drug_count": int(positives.drug_index.nunique()),
        "matched_control_row_count": int(len(controls)),
        "controls_per_positive": 5,
        "interpretation": "independent multi-regulatory-label robustness validation; not incidence or causal validation",
    }
    (OUT / "coverage_and_protocol.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(global_summary.to_string(index=False))


if __name__ == "__main__":
    main()
