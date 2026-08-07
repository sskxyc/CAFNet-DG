from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis_outputs" / "scientific_gap_resolution_20260807"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check(condition: bool, evidence: str, detail: str) -> dict:
    return {"verified": bool(condition), "evidence": evidence, "detail": detail}


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def main() -> None:
    result: dict[str, dict] = {}

    snapshot = ROOT / "docs" / "reproducibility" / "code_SHA256.csv"
    runtime = ROOT / "docs" / "reproducibility" / "runtime_manifest.json"
    result["frozen_reproducibility_inputs"] = check(
        snapshot.exists() and runtime.exists(), relative(snapshot), "Code hashes and runtime manifest must both exist."
    )

    ordinal = BASE / "ordinal_diagnostics"
    ordinal_ok = True
    ordinal_detail = []
    for scenario in ["cold", "warm"]:
        path = ordinal / f"ordinal_{scenario}_per_fold.csv"
        ok = path.exists() and len(pd.read_csv(path)) == 10
        ordinal_ok &= ok
        ordinal_detail.append(f"{scenario}_10fold={ok}")
        ordinal_ok &= (ordinal / f"ordinal_{scenario}_confusion_counts.csv").exists()
        ordinal_ok &= (ordinal / f"ordinal_{scenario}_confusion_row_normalized.csv").exists()
    result["ordinal_diagnostics"] = check(
        ordinal_ok, relative(ordinal), ", ".join(ordinal_detail) + "; QWK/within-one/confusion files required."
    )

    scaffold = BASE / "scaffold_cafnet_dg"
    scaffold_rank = scaffold / "scaffold_ranking_by_fold.csv"
    scaffold_ok = False
    if scaffold_rank.exists():
        frame = pd.read_csv(scaffold_rank)
        scaffold_ok = len(frame) == 30 and frame.groupby("model").fold.nunique().eq(10).all()
        scaffold_ok &= (scaffold / "scaffold_train_test_similarity_by_drug.csv").exists()
        scaffold_ok &= (scaffold / "scaffold_ranking_paired_tests.csv").exists()
    result["scaffold_disjoint"] = check(
        scaffold_ok, relative(scaffold), "Requires 3 models x 10 folds, similarity audit, and paired tests."
    )

    determinism = BASE / "determinism_audit"
    det_summary = determinism / "determinism_summary.json"
    strict_status = BASE / "determinism_strict_status.json"
    det_ok = det_summary.exists() and strict_status.exists()
    detail = "Strict probe and three repeats are missing."
    if det_ok:
        det = read_json(det_summary)
        strict = read_json(strict_status)
        detail = (
            f"strict_supported={strict.get('strict_supported')}; "
            f"bitwise_equal={det.get('all_prediction_arrays_bitwise_equal')}; "
            f"max_abs_difference={det.get('max_abs_difference')}"
        )
    result["determinism_audit"] = check(det_ok, relative(determinism), detail)

    rare = BASE / "rare_e2e_screen"
    decision_path = rare / "screen_promotion_decision.csv"
    rare_ok = decision_path.exists() and (rare / "screen_paired_tests.csv").exists()
    rare_detail = "Full 10-fold ablation decision is missing."
    if rare_ok:
        decision = pd.read_csv(decision_path)
        promoted = decision.loc[decision.promote_to_10fold.astype(bool), "model"].tolist()
        rare_detail = f"all_variants_reported={len(decision) == 4}; promotable_variants={promoted}"
        rare_ok &= len(decision) == 4
    result["rare_adr_and_end_to_end_fusion"] = check(rare_ok, relative(rare), rare_detail)

    onsides = BASE / "onsides_v311_high_confidence" / "coverage_and_protocol.json"
    aems = BASE / "fda_aems_signal_validation" / "coverage_and_protocol.json"
    ct_ade = BASE / "ct_ade_pt_external_validation" / "coverage_and_protocol.json"
    ct_frequency = BASE / "ct_ade_frequency_validation" / "coverage_and_protocol.json"
    external_ok = onsides.exists() and aems.exists() and ct_ade.exists() and ct_frequency.exists()
    external_detail = "CT-ADE association/frequency, OnSIDES, and FDA AEMS coverage metadata are all required."
    if external_ok:
        on = read_json(onsides)
        fd = read_json(aems)
        ct = read_json(ct_ade)
        ct_freq = read_json(ct_frequency)
        external_ok &= on.get("non_sider_external_positive_count", 0) > 0
        external_ok &= fd.get("non_sider_external_positive_count", 0) > 0
        external_ok &= ct.get("non_sider_positives", 0) > 0
        external_ok &= ct.get("mapped_drugs", 0) >= 20
        external_ok &= ct_freq.get("non_sider_nonzero_frequency_pairs", 0) >= 100
        external_detail = (
            f"CT-ADE positives={ct.get('non_sider_positives')} over {ct.get('mapped_drugs')} drugs, "
            f"OnSIDES positives={on.get('non_sider_external_positive_count')}, "
            f"FDA signals={fd.get('non_sider_external_positive_count')}; "
            f"CT-ADE frequency pairs={ct_freq.get('non_sider_nonzero_frequency_pairs')}; "
            "controlled-trial prioritization but failed external ordinal calibration and no EHR/incidence/causal claim."
        )
    result["independent_external_validation"] = check(
        external_ok,
        "; ".join(relative(path) for path in [ct_ade, ct_frequency, onsides, aems]),
        external_detail,
    )

    all_verified = all(item["verified"] for item in result.values())
    payload = {"all_experimental_work_packages_verified": all_verified, "requirements": result}
    (BASE / "COMPLETION_AUDIT.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# Scientific-Gap Completion Audit", "", f"All work packages verified: **{all_verified}**", ""]
    for name, item in result.items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Verified: `{item['verified']}`",
                f"- Evidence: `{item['evidence']}`",
                f"- Detail: {item['detail']}",
                "",
            ]
        )
    (BASE / "COMPLETION_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
