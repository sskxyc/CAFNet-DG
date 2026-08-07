from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import scipy.io as sio
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "scientific_gap_resolution_20260807" / "fda_aems_signal_validation"
CACHE = ROOT / "data_external" / "fda_aems_signals_20260807"
INDEX_URL = (
    "https://www.fda.gov/drugs/fda-adverse-event-monitoring-system-aems/"
    "new-safety-information-or-potential-signals-serious-risks-identified-fda-adverse-event-monitoring"
)
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


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self.links: list[str] = []
        self.table: list[list[str]] | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "a" and attr.get("href"):
            self.links.append(str(attr["href"]))
        elif tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in {"th", "td"} and self.row is not None:
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self.cell is not None and self.row is not None:
            self.row.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None and self.table is not None:
            if any(self.row):
                self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            if self.table:
                self.tables.append(self.table)
            self.table = None


def fetch(url: str) -> tuple[str, str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    path = CACHE / f"{key}.html"
    if path.exists():
        return path.read_text(encoding="utf-8"), str(path)
    response = requests.get(url, timeout=45, headers={"User-Agent": "CAFNet-DG reproducibility audit/1.0"})
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return response.text, str(path)


def normalize(text: str) -> str:
    text = text.lower().replace(".", " ").replace("-", " ")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def match_term(term: str, text: str, allow_token_reorder: bool) -> bool:
    term_norm = normalize(term)
    text_norm = normalize(text)
    if not term_norm:
        return False
    if re.search(rf"(?<![a-z0-9]){re.escape(term_norm)}(?![a-z0-9])", text_norm):
        return True
    term_tokens = term_norm.split()
    text_tokens = set(text_norm.split())
    return allow_token_reorder and len(term_tokens) >= 2 and set(term_tokens).issubset(text_tokens)


SALT_AND_FORM_TOKENS = {
    "anhydrous", "acetate", "besylate", "bromide", "calcium", "capsule", "capsules",
    "chloride", "citrate", "dihydrochloride", "disodium", "fumarate", "gluconate",
    "hcl", "hydrobromide", "hydrochloride", "injection", "magnesium", "maleate",
    "mesylate", "monohydrate", "phosphate", "potassium", "sodium", "solution", "sulfate",
    "tablet", "tablets", "tartrate",
}


def match_drug(term: str, product_text: str) -> bool:
    """Accept monotherapy mentions; reject incidental co-ingredients in combinations."""
    term_norm = normalize(term)
    for segment in re.findall(r"\(([^()]*)\)", product_text):
        segment_norm = normalize(segment)
        match = re.search(rf"(?<![a-z0-9]){re.escape(term_norm)}(?![a-z0-9])", segment_norm)
        if not match:
            continue
        before = segment_norm[: match.start()].split()
        after = segment_norm[match.end() :].split()
        if set(before + after).issubset(SALT_AND_FORM_TOKENS):
            return True

    product_norm = normalize(product_text)
    if product_norm.startswith(term_norm + " ") or product_norm == term_norm:
        return True
    return f"{term_norm} containing products" in product_norm


def quarterly_pages() -> list[str]:
    html, _ = fetch(INDEX_URL)
    parser = TableParser()
    parser.feed(html)
    pages = []
    for href in parser.links:
        value = href.lower()
        if not any(str(year) in value for year in range(2023, 2027)):
            continue
        if "safety-information" not in value and "potential-signals" not in value:
            continue
        pages.append(urljoin(INDEX_URL, href))
    return sorted(set(pages))


def scrape_entries() -> tuple[pd.DataFrame, pd.DataFrame]:
    entries = []
    sources = []
    for url in quarterly_pages():
        html, cache_path = fetch(url)
        parser = TableParser()
        parser.feed(html)
        n_rows = 0
        for table in parser.tables:
            for row in table:
                if len(row) < 2:
                    continue
                product, signal = row[0].strip(), row[1].strip()
                header = normalize(product + " " + signal)
                if "product name" in header or "potential signal" in header:
                    continue
                if not product or not signal:
                    continue
                entries.append({"source_url": url, "product_text": product, "signal_text": signal})
                n_rows += 1
        sources.append({"source_url": url, "cache_path": cache_path, "n_rows": n_rows})
    return pd.DataFrame(entries).drop_duplicates(), pd.DataFrame(sources)


def labels() -> tuple[list[str], list[str]]:
    drugs = []
    with (ROOT / "data" / "drug_SMILES_750.txt").open(encoding="utf-8") as handle:
        for line in handle:
            drugs.append(line.split(",", 1)[0].strip().replace(".", " "))
    mat = sio.loadmat(ROOT / "data" / "side_effect_label_750.mat")
    sides = [str(value[0]) for value in mat["side_effect"].ravel()]
    return drugs, sides


def map_entries(entries: pd.DataFrame, drugs: list[str], sides: list[str]) -> pd.DataFrame:
    mapped = []
    for entry_index, row in entries.iterrows():
        drug_hits = [i for i, name in enumerate(drugs) if match_drug(name, row.product_text)]
        side_hits = [i for i, name in enumerate(sides) if match_term(name, row.signal_text, True)]
        side_token_sets = {i: set(normalize(sides[i]).split()) for i in side_hits}
        side_hits = [
            i for i in side_hits
            if not any(side_token_sets[i] < side_token_sets[j] for j in side_hits if i != j)
        ]
        for drug_index in drug_hits:
            for side_index in side_hits:
                mapped.append(
                    {
                        "entry_index": int(entry_index),
                        "source_url": row.source_url,
                        "product_text": row.product_text,
                        "signal_text": row.signal_text,
                        "drug_index": drug_index,
                        "drug_name": drugs[drug_index],
                        "side_index": side_index,
                        "side_effect": sides[side_index],
                        "mapping_rule": "monotherapy/identical-combination drug; maximal-specific exact or reordered multi-token ADR",
                    }
                )
    return pd.DataFrame(mapped).drop_duplicates(["drug_index", "side_index"])


def primary_mapping_filter(mapped: pd.DataFrame) -> pd.DataFrame:
    """Pre-score semantic exclusions for mappings that are not single-drug ADR targets."""
    reviewed = mapped.copy()
    reviewed["primary_mapping_accepted"] = True
    reviewed["mapping_exclusion_reason"] = ""
    rules = [
        (
            reviewed.signal_text.str.contains("drug-drug interaction", case=False, regex=False),
            "interaction signal cannot be attributed to a single drug",
        ),
        (
            (reviewed.side_effect == "infection")
            & reviewed.signal_text.str.contains("central nervous system infection", case=False, regex=False),
            "generic infection label is broader than the regulatory signal",
        ),
        (
            (reviewed.side_effect == "oedema")
            & reviewed.signal_text.str.contains("corneal oedema|macular oedema", case=False, regex=True),
            "generic oedema label is broader than the organ-specific regulatory signal",
        ),
    ]
    for condition, reason in rules:
        reviewed.loc[condition, "primary_mapping_accepted"] = False
        reviewed.loc[condition, "mapping_exclusion_reason"] = reason
    return reviewed


def read_predictions(path: Path, masks: dict) -> np.ndarray:
    matrix = pd.read_csv(path, header=None).to_numpy(dtype=float)
    if matrix.shape != (750, 994):
        raise ValueError(f"Unexpected cold prediction shape {matrix.shape}: {path}")
    full = np.empty_like(matrix)
    start = 0
    for fold in range(10):
        ids = np.flatnonzero(masks[f"mask{fold}"][:, 0] == 0)
        full[ids] = matrix[start : start + len(ids)]
        start += len(ids)
    if start != len(matrix):
        raise ValueError("Cold prediction matrix could not be consumed by the fixed masks")
    return full


def matched_controls(positives: pd.DataFrame, raw: np.ndarray, masks: dict) -> pd.DataFrame:
    labels = (raw != 0).astype(int)
    positive_by_drug = positives.groupby("drug_index")["side_index"].apply(set).to_dict()
    fold_by_drug = {}
    for fold in range(10):
        for drug in np.flatnonzero(masks[f"mask{fold}"][:, 0] == 0):
            fold_by_drug[int(drug)] = fold
    rows = []
    for row in positives.itertuples(index=False):
        fold = fold_by_drug[int(row.drug_index)]
        train = masks[f"mask{fold}"][:, 0] != 0
        prevalence = labels[train].sum(axis=0).astype(float)
        ordered = np.argsort(prevalence, kind="stable")
        bin_id = np.empty(raw.shape[1], dtype=int)
        for index, cols in enumerate(np.array_split(ordered, 5)):
            bin_id[cols] = index
        candidates = np.flatnonzero(
            (labels[int(row.drug_index)] == 0)
            & (bin_id == bin_id[int(row.side_index)])
        )
        excluded = positive_by_drug.get(int(row.drug_index), set())
        candidates = np.asarray([value for value in candidates if int(value) not in excluded], dtype=int)
        if len(candidates) == 0:
            continue
        distance = np.abs(prevalence[candidates] - prevalence[int(row.side_index)])
        selected = candidates[np.argsort(distance, kind="stable")[:5]]
        for control in selected:
            rows.append(
                {
                    "drug_index": int(row.drug_index),
                    "positive_side_index": int(row.side_index),
                    "control_side_index": int(control),
                    "fold": fold,
                    "positive_prevalence": float(prevalence[int(row.side_index)]),
                    "control_prevalence": float(prevalence[int(control)]),
                }
            )
    return pd.DataFrame(rows)


def score_controls(controls: pd.DataFrame, predictions: dict[str, np.ndarray], raw: np.ndarray, masks: dict) -> pd.DataFrame:
    labels = (raw != 0).astype(int)
    rows = []
    for pair in controls.itertuples(index=False):
        train = masks[f"mask{pair.fold}"][:, 0] != 0
        prevalence = labels[train].sum(axis=0).astype(float)
        model_scores = {**predictions, "Global popularity": np.tile(prevalence, (1, 1))}
        for model, score in model_scores.items():
            vector = score[0] if model == "Global popularity" else score[int(pair.drug_index)]
            rows.append(
                {
                    **pair._asdict(),
                    "model": model,
                    "positive_score": float(vector[int(pair.positive_side_index)]),
                    "control_score": float(vector[int(pair.control_side_index)]),
                }
            )
    return pd.DataFrame(rows)


def summaries(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rows = []
    per_drug_rows = []
    for model, group in scored.groupby("model", sort=False):
        difference = group.positive_score.to_numpy() - group.control_score.to_numpy()
        strict_win = difference > 0
        ties = np.isclose(difference, 0.0, rtol=0.0, atol=1e-12)
        y = np.r_[np.ones(len(group)), np.zeros(len(group))]
        s = np.r_[group.positive_score.to_numpy(), group.control_score.to_numpy()]
        global_rows.append(
            {
                "model": model,
                "n_positive_control_rows": len(group),
                "n_unique_regulatory_signals": group[["drug_index", "positive_side_index"]].drop_duplicates().shape[0],
                "n_drugs": group.drug_index.nunique(),
                "AUROC": roc_auc_score(y, s),
                "AUPR": average_precision_score(y, s),
                "strict_positive_outranks_control": float(strict_win.mean()),
                "tie_rate": float(ties.mean()),
                "tie_aware_concordance": float((strict_win.astype(float) + 0.5 * ties.astype(float)).mean()),
                "mean_score_difference": float(difference.mean()),
            }
        )
        for drug, drug_group in group.groupby("drug_index"):
            drug_difference = drug_group.positive_score.to_numpy() - drug_group.control_score.to_numpy()
            drug_win = drug_difference > 0
            drug_tie = np.isclose(drug_difference, 0.0, rtol=0.0, atol=1e-12)
            per_drug_rows.append(
                {
                    "model": model,
                    "drug_index": int(drug),
                    "n_rows": len(drug_group),
                    "strict_positive_outranks_control": float(drug_win.mean()),
                    "tie_rate": float(drug_tie.mean()),
                    "tie_aware_concordance": float((drug_win.astype(float) + 0.5 * drug_tie.astype(float)).mean()),
                    "mean_score_difference": float(drug_difference.mean()),
                }
            )
    return pd.DataFrame(global_rows), pd.DataFrame(per_drug_rows)


def paired_tests(per_drug: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = per_drug[per_drug.model == "CAFNet-DG"]
    for baseline_name in ["CAFNet-D", "CAFNet", "Global popularity"]:
        baseline = per_drug[per_drug.model == baseline_name]
        merged = target.merge(baseline, on="drug_index", suffixes=("_target", "_baseline"))
        for metric in ["tie_aware_concordance", "mean_score_difference"]:
            delta = merged[f"{metric}_target"] - merged[f"{metric}_baseline"]
            if np.allclose(delta, 0):
                statistic, p_value = 0.0, 1.0
            else:
                test = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            rows.append(
                {
                    "comparison": f"CAFNet-DG vs {baseline_name}",
                    "metric": metric,
                    "n_drugs": len(delta),
                    "mean_delta": float(delta.mean()),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p": p_value,
                }
            )
    result = pd.DataFrame(rows)
    for comparison, index in result.groupby("comparison").groups.items():
        p = result.loc[index, "wilcoxon_p"].to_numpy(dtype=float)
        order = np.argsort(p)
        adjusted = np.empty_like(p)
        running = 0.0
        for rank, position in enumerate(order):
            running = max(running, (len(p) - rank) * p[position])
            adjusted[position] = min(running, 1.0)
        result.loc[index, "wilcoxon_p_holm"] = adjusted
    result["holm_significant"] = result["wilcoxon_p_holm"] < 0.05
    return result


def drug_bootstrap(scored: pd.DataFrame, n_bootstrap: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for model, group in scored.groupby("model", sort=False):
        drugs = group.drug_index.unique()
        estimates = {"AUROC": [], "AUPR": [], "tie_aware_concordance": []}
        grouped = {drug: group[group.drug_index == drug] for drug in drugs}
        for _ in range(n_bootstrap):
            sampled = rng.choice(drugs, size=len(drugs), replace=True)
            boot = pd.concat([grouped[drug] for drug in sampled], ignore_index=True)
            y = np.r_[np.ones(len(boot)), np.zeros(len(boot))]
            s = np.r_[boot.positive_score.to_numpy(), boot.control_score.to_numpy()]
            estimates["AUROC"].append(roc_auc_score(y, s))
            estimates["AUPR"].append(average_precision_score(y, s))
            difference = boot.positive_score.to_numpy() - boot.control_score.to_numpy()
            wins = difference > 0
            ties = np.isclose(difference, 0.0, rtol=0.0, atol=1e-12)
            estimates["tie_aware_concordance"].append(
                float((wins.astype(float) + 0.5 * ties.astype(float)).mean())
            )
        for metric, values in estimates.items():
            values = np.asarray(values, dtype=float)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "bootstrap_n": n_bootstrap,
                    "estimate_mean": float(values.mean()),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entries, sources = scrape_entries()
    entries.to_csv(OUT / "fda_quarterly_entries_raw.csv", index=False)
    sources.to_csv(OUT / "source_pages.csv", index=False)
    drugs, sides = labels()
    mapped = primary_mapping_filter(map_entries(entries, drugs, sides))
    mapped.to_csv(OUT / "strict_mapping_audit_all.csv", index=False)

    raw = sio.loadmat(ROOT / "data" / "raw_frequency_750.mat")["R"].astype(float)
    if mapped.empty:
        raise RuntimeError("Strict mapping produced no FDA signal pairs; manual mapping review is required")
    mapped["known_in_sider_matrix"] = raw[mapped.drug_index, mapped.side_index] != 0
    mapped.to_csv(OUT / "strict_mapping_audit_with_sider_overlap.csv", index=False)
    positives = mapped[mapped.primary_mapping_accepted & ~mapped.known_in_sider_matrix].copy()
    positives.to_csv(OUT / "temporal_external_positive_pairs_non_sider.csv", index=False)

    masks = sio.loadmat(ROOT / "data" / "blind_mask_mat_750.mat")
    controls = matched_controls(positives, raw, masks)
    controls.to_csv(OUT / "prevalence_matched_controls.csv", index=False)
    predictions = {name: read_predictions(path, masks) for name, path in MODEL_FILES.items()}
    scored = score_controls(controls, predictions, raw, masks)
    scored.to_csv(OUT / "scored_positive_control_rows.csv", index=False)
    global_summary, per_drug = summaries(scored)
    global_summary.to_csv(OUT / "global_summary.csv", index=False)
    per_drug.to_csv(OUT / "per_drug_summary.csv", index=False)
    paired_tests(per_drug).to_csv(OUT / "paired_tests.csv", index=False)
    drug_bootstrap(scored).to_csv(OUT / "drug_bootstrap_ci.csv", index=False)
    metadata = {
        "source": "FDA AEMS/FAERS quarterly potential-signal and new-safety-information tables",
        "years": "2023-2026",
        "source_page_count": int(len(sources)),
        "raw_entry_count": int(len(entries)),
        "strict_mapped_pair_count_before_semantic_exclusions": int(len(mapped)),
        "semantic_exclusion_count": int((~mapped.primary_mapping_accepted).sum()),
        "non_sider_external_positive_count": int(len(positives)),
        "mapped_drug_count": int(positives.drug_index.nunique()),
        "matched_control_row_count": int(len(controls)),
        "interpretation": "regulatory pharmacovigilance signal validation; not incidence or causal validation",
    }
    (OUT / "coverage_and_protocol.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(global_summary.to_string(index=False))


if __name__ == "__main__":
    main()
