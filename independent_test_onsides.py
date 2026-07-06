import os
import csv
import argparse
import numpy as np
import re
import scipy.io
from scipy import stats
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
from sklearn.neighbors import kneighbors_graph
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

from Net import CAFNet, A3_Net
from utils import myDataset, rmse, MAE, pearson, spearman
from vector import convert2graph


def read_smiles_list_csv(path):
    smiles = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            smiles.append(row[1].strip())
    return smiles


def read_onsides_smiles_ordered(drug_list_path, drug_smiles_path):
    id_to_smiles = {}
    with open(drug_smiles_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            drug_id = row[0].strip()
            id_to_smiles[drug_id] = row[1].strip()
    ordered = []
    with open(drug_list_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            drug_id = line.strip().split("\t")[0]
            if not drug_id:
                continue
            if drug_id not in id_to_smiles:
                ordered.append(None)
            else:
                ordered.append(id_to_smiles[drug_id])
    return ordered


def normalize_text(s):
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_set(s):
    if not s:
        return set()
    return set(s.split(" "))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def read_train_side_effects(mat_path):
    mat = scipy.io.loadmat(mat_path)
    side_effect = mat["side_effect"]
    side_effect = [str(x[0]) for x in side_effect.tolist()]
    return side_effect


def read_onsides_side_effects(list_path):
    se_list = []
    with open(list_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            name = line.strip().split("\t")[0]
            if name:
                se_list.append(name)
    return se_list


def build_overlap_mappings(
    train_smiles,
    onsides_smiles_ordered,
    train_se_list,
    onsides_se_list,
    fuzzy_thr=0.88,
    jaccard_thr=0.8,
):
    train_smile_to_idx = {}
    for i, smi in enumerate(train_smiles):
        if smi not in train_smile_to_idx:
            train_smile_to_idx[smi] = i

    onsides_smile_to_idx = {}
    for i, smi in enumerate(onsides_smiles_ordered):
        if smi is None:
            continue
        if smi not in onsides_smile_to_idx:
            onsides_smile_to_idx[smi] = i

    overlap_drug_pairs = []
    for i, smi in enumerate(train_smiles):
        if smi in onsides_smile_to_idx:
            overlap_drug_pairs.append((i, onsides_smile_to_idx[smi]))

    from difflib import SequenceMatcher

    train_se_norm = [normalize_text(s) for s in train_se_list]
    onsides_se_norm = [normalize_text(s) for s in onsides_se_list]
    onsides_tokens = [token_set(s) for s in onsides_se_norm]

    train_to_onsides_se = np.full(len(train_se_norm), -1, dtype=int)
    mapping_rows = []

    for i, s in enumerate(train_se_norm):
        if not s:
            continue
        best_idx = -1
        best_score = 0.0
        s_tokens = token_set(s)
        for j, t in enumerate(onsides_se_norm):
            if not t:
                continue
            fz = SequenceMatcher(None, s, t).ratio()
            jac = jaccard(s_tokens, onsides_tokens[j])
            if fz >= fuzzy_thr and jac >= jaccard_thr:
                score = 0.5 * fz + 0.5 * jac
                if score > best_score:
                    best_score = score
                    best_idx = j
        if best_idx >= 0:
            train_to_onsides_se[i] = best_idx
            mapping_rows.append((
                train_se_list[i],
                onsides_se_list[best_idx],
                best_score,
            ))

    return overlap_drug_pairs, train_to_onsides_se, mapping_rows


def build_side_effect_graph(train_frequency, knn=10, metric="cosine", use_pca=False):
    freq_t = train_frequency.T
    if use_pca:
        from sklearn.decomposition import PCA
        pca_ = PCA(n_components=256)
        freq_t = pca_.fit_transform(freq_t)
    A = kneighbors_graph(freq_t, knn, mode="connectivity", metric=metric, include_self=False)
    edges = np.array(A.nonzero())
    edges = torch.tensor(edges, dtype=torch.long)
    return edges


def predict_matrix(model, loader, device, side_effect_graph, DF, not_FC):
    model.eval()
    preds = []
    with torch.no_grad():
        side_effect_graph = side_effect_graph.to(device)
        for data in loader:
            data = data.to(device)
            out, _, _ = model(data, side_effect_graph, DF, not_FC)
            preds.append(out.cpu())
    return torch.cat(preds, dim=0).numpy()


def compute_metrics(y_true, y_pred):
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": MAE(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mse": np.mean((y_true - y_pred) ** 2),
    }
    # Classification-style metrics from regression scores
    y_bin = (y_true > 0).astype(int)
    if np.unique(y_bin).size == 2:
        metrics["auc_roc"] = roc_auc_score(y_bin, y_pred)
        metrics["aupr"] = average_precision_score(y_bin, y_pred)
    else:
        metrics["auc_roc"] = np.nan
        metrics["aupr"] = np.nan
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cafnet", default=r"G:\studyPj\MIP-ASF\A-3Net-master-master\result_WS\10WS_CAFNet_knn=10_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cosine_abl=CA_gate_new_loss=focal\checkpoints\10MF_CAFNet_epoch=100.model")
    parser.add_argument("--a3net", default=r"G:\studyPj\MIP-ASF\A-3Net-master-master\result_WS\10WS_A3_Net_knn=10_wd=0.001_epoch=100_lamb=0.03_lr0.0004_dim=200_eps=0.5_DF=False_PCA=False_not-FC=False_cosine\checkpoints\10MF_A3_Net_epoch=100.model")
    parser.add_argument("--outdir", default="plots/independent_test")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_data_dir = os.path.join(base_dir, "data")
    onsides_dir = r"G:\studyPj\MIP-ASF\CAFNet-Onsides\data\3to1"

    train_smiles = read_smiles_list_csv(os.path.join(train_data_dir, "drug_SMILES_750.txt"))
    onsides_smiles_ordered = read_onsides_smiles_ordered(
        os.path.join(onsides_dir, "drug_list_onsides_balanced_3to1.txt"),
        os.path.join(onsides_dir, "drug_SMILES_onsides_balanced_3to1.txt"),
    )

    train_se_list = read_train_side_effects(os.path.join(train_data_dir, "side_effect_label_750.mat"))
    onsides_se_list = read_onsides_side_effects(os.path.join(onsides_dir, "side_effect_list_onsides_balanced_3to1.txt"))

    overlap_drug_pairs, train_to_onsides_se, mapping_rows = build_overlap_mappings(
        train_smiles, onsides_smiles_ordered, train_se_list, onsides_se_list,
        fuzzy_thr=0.88, jaccard_thr=0.8
    )

    if not overlap_drug_pairs:
        raise RuntimeError("No overlapping drugs found between train and OnSides by SMILES.")

    overlap_train_idx = [p[0] for p in overlap_drug_pairs]
    overlap_onsides_idx = [p[1] for p in overlap_drug_pairs]
    overlap_smiles = [train_smiles[i] for i in overlap_train_idx]

    train_freq = scipy.io.loadmat(os.path.join(train_data_dir, "raw_frequency_750.mat"))["R"]
    onsides_freq = scipy.io.loadmat(os.path.join(onsides_dir, "raw_frequency_onsides_balanced_3to1.mat"))["R"]

    onsides_subset = onsides_freq[overlap_onsides_idx, :]
    n_train_se = len(train_se_list)
    mapped = np.zeros((len(overlap_smiles), n_train_se), dtype=onsides_subset.dtype)
    mask = np.zeros_like(mapped, dtype=int)
    valid_cols = np.where(train_to_onsides_se >= 0)[0]
    mapped[:, valid_cols] = onsides_subset[:, train_to_onsides_se[valid_cols]]
    mask[:, valid_cols] = 1

    print(f"Overlap drugs: {len(overlap_smiles)}")
    print(f"Overlap side-effects: {len(valid_cols)} / {n_train_se}")
    if len(valid_cols) == 0:
        raise RuntimeError(
            "No overlapping side-effects found. "
            "Check name normalization or use a mapping table."
        )

    simle_graph = convert2graph(overlap_smiles)
    dataset_root = os.path.join(base_dir, "data_WS_onsides_3to1")
    dataset_name = "onsides_independent_3to1"
    test_data = myDataset(
        root=dataset_root,
        dataset=dataset_name,
        drug_simles=overlap_smiles,
        frequencyMat=mapped,
        simle_graph=simle_graph,
    )
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    side_effect_edges = build_side_effect_graph(train_freq, knn=10, metric="cosine", use_pca=False)
    node_label = scipy.io.loadmat(os.path.join(train_data_dir, "side_effect_label_750.mat"))["node_label"]
    side_effect_graph = Data(x=torch.tensor(node_label, dtype=torch.float), edge_index=side_effect_edges)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(base_dir, args.outdir)
    os.makedirs(out_dir, exist_ok=True)

    if mapping_rows:
        mapping_rows = sorted(mapping_rows, key=lambda x: x[2], reverse=True)
        with open(os.path.join(out_dir, "side_effect_mapping.csv"), "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["train_side_effect", "onsides_side_effect", "score"])
            for row in mapping_rows:
                writer.writerow(row)

    results = []
    for name, model_path in [("CAFNet", args.cafnet), ("A3Net", args.a3net)]:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        if name == "CAFNet":
            model = CAFNet(input_dim=109, input_dim_e=243, output_dim=200, use_cross_attn=True, fusion_mode="gate", gate_mode="new").to(device)
        else:
            model = A3_Net(input_dim=109, input_dim_e=243, output_dim=200).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=True)

        preds = predict_matrix(model, test_loader, device, side_effect_graph, DF=False, not_FC=False)
        y_true = mapped[mask == 1].astype(float)
        y_pred = preds[mask == 1].astype(float)

        if y_true.size == 0:
            raise RuntimeError("No valid evaluation samples after masking.")

        metrics = compute_metrics(y_true, y_pred)
        results.append((name, metrics))

        np.save(os.path.join(base_dir, args.outdir, f"indep_onsides_{name}_pred.npy"), preds)
        np.save(os.path.join(base_dir, args.outdir, f"indep_onsides_true.npy"), mapped)
        np.save(os.path.join(base_dir, args.outdir, f"indep_onsides_mask.npy"), mask)

    with open(os.path.join(base_dir, args.outdir, "metrics_summary.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "rmse", "mae", "pearson", "spearman", "r2", "mse", "auc_roc", "aupr"])
        for name, m in results:
            writer.writerow([
                name, m["rmse"], m["mae"], m["pearson"], m["spearman"], m["r2"], m["mse"],
                m["auc_roc"], m["aupr"]
            ])


if __name__ == "__main__":
    main()
