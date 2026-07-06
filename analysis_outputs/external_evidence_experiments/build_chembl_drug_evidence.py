import csv
import argparse
import json
import time
import urllib.parse
from pathlib import Path

import numpy as np
import requests
import scipy.io as sio


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "data_external"
CACHE_DIR = OUT_DIR / "chembl_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _mat_string(x):
    while isinstance(x, np.ndarray):
        if x.size == 1:
            x = x.item()
        else:
            x = x.flat[0]
    return str(x)


def normalize_name(name):
    return name.replace(".", " ").replace("-", " ").lower().strip()


def cache_get_json(url, cache_name, sleep_s=0.05, retries=5):
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    last_error = None
    for attempt in range(int(retries)):
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            data = response.json()
            cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            time.sleep(sleep_s)
            return data
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(10.0, 0.5 * (2 ** attempt)))
    raise last_error


def chembl_search(name):
    query = urllib.parse.quote(normalize_name(name))
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={query}&limit=20"
    return cache_get_json(url, f"search_{query}.json").get("molecules", [])


def chembl_mechanisms(molecule_chembl_id):
    url = (
        "https://www.ebi.ac.uk/chembl/api/data/mechanism.json?"
        f"molecule_chembl_id={urllib.parse.quote(molecule_chembl_id)}&limit=1000"
    )
    try:
        return cache_get_json(url, f"mechanism_{molecule_chembl_id}.json").get("mechanisms", [])
    except requests.RequestException as exc:
        fail_path = CACHE_DIR / "mechanism_failures.tsv"
        with fail_path.open("a", encoding="utf-8") as f:
            f.write(f"{molecule_chembl_id}\t{type(exc).__name__}\t{exc}\n")
        return []


def score_candidate(query_name, mol):
    query_norm = normalize_name(query_name)
    pref = normalize_name(mol.get("pref_name") or "")
    synonyms = [normalize_name(syn.get("molecule_synonym", "")) for syn in mol.get("molecule_synonyms", [])]
    atc = mol.get("atc_classifications") or []
    max_phase = mol.get("max_phase")
    exact = int(query_norm == pref or query_norm in synonyms)
    approved = int(max_phase is not None and float(max_phase) >= 4)
    has_atc = int(len(atc) > 0)
    return 100 * exact + 10 * approved + 3 * has_atc


def select_candidate(name, candidates):
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda m: score_candidate(name, m), reverse=True)
    return ranked[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_targets", action="store_true", default=False)
    parser.add_argument("--output_prefix", default="chembl_atc")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = sio.loadmat(DATA_DIR / "raw_frequency_750.mat")
    drugs = [_mat_string(x) for x in raw["drugs"].flatten()]

    rows = []
    atc_terms = set()
    target_terms = set()
    moa_terms = set()
    for idx, name in enumerate(drugs):
        candidates = chembl_search(name)
        selected = select_candidate(name, candidates)
        row = {
            "drug_index": idx,
            "sider_name": name,
            "query_name": normalize_name(name),
            "selected_chembl_id": "",
            "selected_pref_name": "",
            "max_phase": "",
            "candidate_count": len(candidates),
            "selection_score": "",
            "atc_codes": "",
            "atc_l1": "",
            "atc_l2": "",
            "atc_l3": "",
            "target_chembl_ids": "",
            "mechanisms": "",
        }
        if selected:
            chembl_id = selected.get("molecule_chembl_id") or ""
            atc_codes = selected.get("atc_classifications") or []
            mechanisms = chembl_mechanisms(chembl_id) if (args.include_targets and chembl_id) else []
            target_ids = sorted(
                {
                    mech.get("target_chembl_id")
                    for mech in mechanisms
                    if mech.get("target_chembl_id")
                }
            )
            moas = sorted(
                {
                    mech.get("mechanism_of_action")
                    for mech in mechanisms
                    if mech.get("mechanism_of_action")
                }
            )
            atc_l1 = sorted({code[:1] for code in atc_codes if len(code) >= 1})
            atc_l2 = sorted({code[:3] for code in atc_codes if len(code) >= 3})
            atc_l3 = sorted({code[:4] for code in atc_codes if len(code) >= 4})
            atc_terms.update(atc_l2)
            atc_terms.update(atc_l3)
            target_terms.update(target_ids)
            moa_terms.update(moas)
            row.update(
                {
                    "selected_chembl_id": chembl_id,
                    "selected_pref_name": selected.get("pref_name") or "",
                    "max_phase": selected.get("max_phase"),
                    "selection_score": score_candidate(name, selected),
                    "atc_codes": "|".join(atc_codes),
                    "atc_l1": "|".join(atc_l1),
                    "atc_l2": "|".join(atc_l2),
                    "atc_l3": "|".join(atc_l3),
                    "target_chembl_ids": "|".join(target_ids),
                    "mechanisms": "|".join(moas),
                }
            )
        rows.append(row)
        if (idx + 1) % 50 == 0:
            print(f"mapped {idx + 1}/{len(drugs)}")

    mapping_csv = OUT_DIR / f"{args.output_prefix}_drug_evidence_mapping.csv"
    with mapping_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    atc_terms = sorted(atc_terms)
    target_terms = sorted(target_terms)
    feature_terms = [f"ATC:{term}" for term in atc_terms] + [f"TARGET:{term}" for term in target_terms]
    feature_index = {term: i for i, term in enumerate(feature_terms)}
    x = np.zeros((len(rows), len(feature_terms)), dtype=np.float32)
    for row in rows:
        i = int(row["drug_index"])
        for term in (row["atc_l2"].split("|") + row["atc_l3"].split("|")):
            if term:
                x[i, feature_index[f"ATC:{term}"]] = 1.0
        for term in row["target_chembl_ids"].split("|"):
            if term:
                x[i, feature_index[f"TARGET:{term}"]] = 1.0

    np.save(OUT_DIR / f"{args.output_prefix}_features.npy", x)
    (OUT_DIR / f"{args.output_prefix}_feature_terms.txt").write_text(
        "\n".join(feature_terms), encoding="utf-8"
    )

    coverage = {
        "n_drugs": len(rows),
        "mapped_chembl": sum(bool(r["selected_chembl_id"]) for r in rows),
        "with_atc": sum(bool(r["atc_codes"]) for r in rows),
        "with_target": sum(bool(r["target_chembl_ids"]) for r in rows),
        "with_any_feature": int((x.sum(axis=1) > 0).sum()),
        "n_features": x.shape[1],
        "n_atc_features": len(atc_terms),
        "n_target_features": len(target_terms),
    }
    with (OUT_DIR / f"{args.output_prefix}_coverage.json").open("w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2)
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
