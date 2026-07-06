# 十折交叉训练， 一次mask73行
import argparse
import csv
import datetime
import random
import shutil

import networkx as nx
import pandas as pd
import scipy
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data, DataLoader

from Net import *
from vector import load_drug_smile, convert2graph
from utils import *

raw_file = 'data/raw_frequency_750.mat'
SMILES_file = 'data/drug_SMILES_750.txt'
blind_mask_mat_file = 'data/blind_mask_mat_750.mat'
side_effect_label = "data/side_effect_label_750.mat"
dataset = 'drug_sideEffect'
input_dim = 109


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def loss_fun(output, label, lam, eps):
    x0 = torch.where(label == 0)
    x1 = torch.where(label != 0)
    loss = torch.sum((output[x1] - label[x1]) ** 2) + lam * torch.sum((output[x0] - eps) ** 2)
    return loss


def bpr_loss(logits, labels, samples_per_row=32, side_weights=None):
    labels = labels.to(device=logits.device)
    if side_weights is not None:
        side_weights = side_weights.to(device=logits.device, dtype=logits.dtype)
    losses = []
    for row_idx in range(logits.size(0)):
        pos_idx = torch.where(labels[row_idx] != 0)[0]
        neg_idx = torch.where(labels[row_idx] == 0)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        n = min(int(samples_per_row), pos_idx.numel(), neg_idx.numel())
        if side_weights is None:
            pos_perm = torch.randperm(pos_idx.numel(), device=logits.device)[:n]
        else:
            pos_w = side_weights[pos_idx].clamp_min(1e-6)
            pos_perm = torch.multinomial(pos_w / pos_w.sum(), num_samples=n, replacement=False)
        neg_perm = torch.randperm(neg_idx.numel(), device=logits.device)[:n]
        pos_scores = logits[row_idx, pos_idx[pos_perm]]
        neg_scores = logits[row_idx, neg_idx[neg_perm]]
        losses.append(-F.logsigmoid(pos_scores - neg_scores).mean())
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def matched_bpr_loss(logits, labels, side_prevalence, samples_per_row=32, n_bins=10, side_weights=None):
    labels = labels.to(device=logits.device)
    side_prevalence = side_prevalence.to(device=logits.device, dtype=logits.dtype)
    side_weights = side_weights.to(device=logits.device, dtype=logits.dtype) if side_weights is not None else None
    ranks = torch.argsort(torch.argsort(side_prevalence))
    bins = torch.clamp((ranks.float() * int(n_bins) / max(1, side_prevalence.numel())).long(), max=int(n_bins) - 1)
    losses = []
    for row_idx in range(logits.size(0)):
        pos_idx = torch.where(labels[row_idx] != 0)[0]
        neg_idx_all = torch.where(labels[row_idx] == 0)[0]
        if pos_idx.numel() == 0 or neg_idx_all.numel() == 0:
            continue
        n = min(int(samples_per_row), pos_idx.numel())
        if side_weights is None:
            pos_perm = torch.randperm(pos_idx.numel(), device=logits.device)[:n]
        else:
            pos_w = side_weights[pos_idx].clamp_min(1e-6)
            pos_perm = torch.multinomial(pos_w / pos_w.sum(), num_samples=n, replacement=False)
        sampled_pos = pos_idx[pos_perm]
        row_losses = []
        for pos_col in sampled_pos:
            same_bin_neg = neg_idx_all[bins[neg_idx_all] == bins[pos_col]]
            if same_bin_neg.numel() == 0:
                same_bin_neg = neg_idx_all
            neg_col = same_bin_neg[torch.randint(same_bin_neg.numel(), (1,), device=logits.device)]
            row_losses.append(-F.logsigmoid(logits[row_idx, pos_col] - logits[row_idx, neg_col.squeeze(0)]))
        if row_losses:
            losses.append(torch.stack(row_losses).mean())
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def external_contrastive_loss(logits, batch_drug_ids, ext_drug_ids, ext_pos_idx, ext_neg_idx,
                              samples_per_drug=64):
    if ext_drug_ids is None or ext_pos_idx is None or ext_neg_idx is None:
        return logits.new_tensor(0.0)
    ext_drug_ids = ext_drug_ids.to(device=logits.device)
    ext_pos_idx = ext_pos_idx.to(device=logits.device)
    ext_neg_idx = ext_neg_idx.to(device=logits.device)
    batch_drug_ids = batch_drug_ids.to(device=logits.device)
    losses = []
    for row_idx, drug_id in enumerate(batch_drug_ids):
        pair_idx = torch.where(ext_drug_ids == drug_id)[0]
        if pair_idx.numel() == 0:
            continue
        n = min(int(samples_per_drug), pair_idx.numel())
        if n <= 0:
            continue
        sampled = pair_idx[torch.randperm(pair_idx.numel(), device=logits.device)[:n]]
        pos_scores = logits[row_idx, ext_pos_idx[sampled]]
        neg_scores = logits[row_idx, ext_neg_idx[sampled]]
        losses.append(-F.logsigmoid(pos_scores - neg_scores).mean())
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def listnet_loss(logits, labels, side_weights=None):
    target = (labels != 0).to(dtype=logits.dtype, device=logits.device)
    if side_weights is not None:
        side_weights = side_weights.to(device=logits.device, dtype=logits.dtype)
        target = target * side_weights.view(1, -1)
    row_has_pos = target.sum(dim=1) > 0
    if not torch.any(row_has_pos):
        return logits.new_tensor(0.0)
    target = target[row_has_pos]
    logits = logits[row_has_pos]
    target_prob = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
    log_prob = F.log_softmax(logits, dim=1)
    return -(target_prob * log_prob).sum(dim=1).mean()


def cafnet_v2_loss(model, output, label, lam, eps, assoc_weight, freq_weight, rank_weight, bpr_samples,
                   list_weight=0.0, side_weights=None, matched_bpr=False, side_prevalence=None,
                   prevalence_bins=10, batch_drug_ids=None, ext_drug_ids=None, ext_pos_idx=None,
                   ext_neg_idx=None, external_weight=0.0, external_samples_per_drug=64,
                   external_target="rank"):
    assoc_logits = getattr(model, "last_assoc_logits", None)
    freq_pred = getattr(model, "last_freq_pred", None)
    if assoc_logits is None or freq_pred is None:
        return loss_fun(output.flatten(), label.flatten(), lam, eps)

    target_assoc = (label != 0).to(dtype=assoc_logits.dtype, device=assoc_logits.device)
    label = label.to(device=assoc_logits.device)
    pos = label != 0
    neg = label == 0

    pos_count = target_assoc.sum().clamp_min(1.0)
    neg_count = (target_assoc.numel() - target_assoc.sum()).clamp_min(1.0)
    pos_weight = torch.clamp(neg_count / pos_count, max=20.0)
    if side_weights is None:
        assoc_loss = F.binary_cross_entropy_with_logits(
            assoc_logits, target_assoc, pos_weight=pos_weight
        )
    else:
        side_weights = side_weights.to(device=assoc_logits.device, dtype=assoc_logits.dtype)
        elem_loss = F.binary_cross_entropy_with_logits(
            assoc_logits, target_assoc, pos_weight=pos_weight, reduction="none"
        )
        elem_weight = torch.where(target_assoc > 0, side_weights.view(1, -1), torch.ones_like(elem_loss))
        assoc_loss = (elem_loss * elem_weight).sum() / elem_weight.sum().clamp_min(1.0)

    freq_loss = freq_pred.new_tensor(0.0)
    if pos.any():
        if side_weights is None:
            freq_loss = F.smooth_l1_loss(freq_pred[pos], label[pos])
        else:
            pos_cols = torch.where(pos)[1]
            elem = F.smooth_l1_loss(freq_pred[pos], label[pos], reduction="none")
            weights = side_weights.to(device=freq_pred.device, dtype=freq_pred.dtype)[pos_cols]
            freq_loss = (elem * weights).sum() / weights.sum().clamp_min(1.0)
    if neg.any():
        zero_loss = F.smooth_l1_loss(freq_pred[neg], torch.full_like(freq_pred[neg], eps))
        freq_loss = freq_loss + lam * zero_loss

    rank_source = output if model.__class__.__name__ == "CAFNetDecoupled" else assoc_logits
    if matched_bpr and side_prevalence is not None:
        rank_loss = matched_bpr_loss(
            rank_source, label, side_prevalence,
            samples_per_row=bpr_samples, n_bins=prevalence_bins, side_weights=side_weights
        )
    else:
        rank_loss = bpr_loss(rank_source, label, samples_per_row=bpr_samples, side_weights=side_weights)
    if list_weight > 0:
        rank_loss = rank_loss + list_weight * listnet_loss(rank_source, label, side_weights=side_weights)
    loss = assoc_weight * assoc_loss + freq_weight * freq_loss + rank_weight * rank_loss
    if external_weight > 0 and batch_drug_ids is not None:
        ext_source = assoc_logits if external_target == "assoc" else rank_source
        ext_loss = external_contrastive_loss(
            ext_source, batch_drug_ids, ext_drug_ids, ext_pos_idx, ext_neg_idx,
            samples_per_drug=external_samples_per_drug,
        )
        loss = loss + float(external_weight) * ext_loss
    return loss


def generateMat():
    """
    将矩阵按比例mask, 将被mask的部分分为10份，生成10份mask位置矩阵，保存在./data_ICS/processed/blind_mask_mat.mat
    :return:
    """
    # 每次加载都把之前的数据删除
    filenames = os.listdir('data')
    for s in filenames:
        os.remove('data' + s)

    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']

    # mask, get mask Mat
    index = np.arange(0, len(raw), 1)
    np.random.shuffle(index)
    x = []
    n = int(np.ceil(len(index) / 10))
    for i in range(10):
        if i == 9:
            x.append(index.tolist())
        x.append(index[0:n].tolist())
        index = index[n:]

    dic = {}
    for i in range(10):
        mask = np.ones(raw.shape)
        mask[x[i]] = 0
        dic['mask' + str(i)] = mask
    scipy.io.savemat(blind_mask_mat_file, dic)


def split_data(tenfold=False, mask_file=None, max_folds=None, drug_feature_file=None, dataset_suffix=''):
    """
    读取 data/blind_mask_mat.mat，根据原始频率矩阵生成10份被mask的频率矩阵并yield
    :return:
    """
    raw_frequency = scipy.io.loadmat(raw_file)
    print('******************')
    blind_mask_mat = scipy.io.loadmat(mask_file or blind_mask_mat_file)
    drug_dict, drug_smile = load_drug_smile(SMILES_file)
    print(len(drug_dict))
    drug_features = np.load(drug_feature_file).astype(np.float32) if drug_feature_file else None
    if drug_features is not None and drug_features.shape[0] != raw_frequency['R'].shape[0]:
        raise ValueError(
            'drug feature row count mismatch: {} vs {}'.format(
                drug_features.shape[0], raw_frequency['R'].shape[0]
            )
        )

    n_folds = 10 if max_folds is None else min(int(max_folds), 10)
    for idx in range(n_folds):
        raw = raw_frequency['R']
        mask = blind_mask_mat['mask' + str(idx)]
        drug_dict, drug_smile = load_drug_smile(SMILES_file)

        index = np.asarray(np.where(mask[:, 0].flatten() == 0)[0]).tolist()
        test_indices = list(index)

        frequencyMat = np.delete(raw, index, axis=0)
        train_drug_features = np.delete(drug_features, test_indices, axis=0) if drug_features is not None else None
        test_drug_features = drug_features[test_indices] if drug_features is not None else None
        print(len(frequencyMat))
        test_smiles = []
        test_label = []
        index.reverse()
        for i in index:
            smi = drug_smile.pop(i)
            test_smiles.append(smi)
            test_label.append(raw[i])
        train_smiles = drug_smile
        test_smiles.reverse()
        test_label.reverse()
        test_label = np.asarray(test_label)

        train_simle_graph = convert2graph(train_smiles)
        test_simle_graph = convert2graph(test_smiles)

        train_data = myDataset(root='data_ICS', dataset=dataset + dataset_suffix + '_blind_train' + str(idx),
                               drug_simles=train_smiles, frequencyMat=frequencyMat,
                               simle_graph=train_simle_graph, drug_features=train_drug_features)
        test_data = myDataset(root='data_ICS', dataset=dataset + dataset_suffix + '_blind_test' + str(idx),
                              drug_simles=test_smiles, frequencyMat=test_label,
                              simle_graph=test_simle_graph, drug_features=test_drug_features)
        yield idx, frequencyMat, mask

        if not tenfold and idx == 0:
            break


# training function at each epoch
def train(model, device, train_loader, optimizer, lamb, epoch, log_interval, sideEffectsGraph, raw, id, DF, not_FC, eps,
          dual_task=False, assoc_weight=1.0, freq_weight=0.2, rank_weight=0.1, bpr_samples=32, list_weight=0.0,
          side_weights=None, matched_bpr=False, side_prevalence=None, prevalence_bins=10,
          ext_drug_ids=None, ext_pos_idx=None, ext_neg_idx=None, external_weight=0.0,
          external_samples_per_drug=64, external_target="rank"):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()

    avg_loss = []
    for batch_idx, data in enumerate(train_loader):

        sideEffectsGraph = sideEffectsGraph.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        out, _, _ = model(data, sideEffectsGraph, DF, not_FC)

        raw_label = data.y

        pred = out.to(device)

        if dual_task:
            batch_drug_ids = model._drug_indices(data, pred.device) if hasattr(model, "_drug_indices") else None
            loss = cafnet_v2_loss(
                model, pred, raw_label, lamb, eps,
                assoc_weight=assoc_weight,
                freq_weight=freq_weight,
                rank_weight=rank_weight,
                bpr_samples=bpr_samples,
                list_weight=list_weight,
                side_weights=side_weights,
                matched_bpr=matched_bpr,
                side_prevalence=side_prevalence,
                prevalence_bins=prevalence_bins,
                batch_drug_ids=batch_drug_ids,
                ext_drug_ids=ext_drug_ids,
                ext_pos_idx=ext_pos_idx,
                ext_neg_idx=ext_neg_idx,
                external_weight=external_weight,
                external_samples_per_drug=external_samples_per_drug,
                external_target=external_target,
            )
        else:
            loss = loss_fun(pred.flatten(), raw_label.flatten(), lamb, eps)

        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if (batch_idx + 1) % log_interval == 0:
            print('{} Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(id, epoch,
                                                                              (batch_idx + 1) * len(data.y),
                                                                              len(train_loader.dataset),
                                                                              100. * (batch_idx + 1) / len(
                                                                                  train_loader),



                                                                            loss.item()))
    return sum(avg_loss) / len(avg_loss)


def predict(model, device, loader, sideEffectsGraph, DF, not_FC):
    model.eval()
    torch.cuda.manual_seed(42)
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    # 对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用with torch.no_grad():来强制之后的内容不进行计算图构建
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for batch_idx, data in enumerate(loader):
            raw_label = torch.FloatTensor(data.y)
            data = data.to(device)
            out, _, _ = model(data, sideEffectsGraph, DF, not_FC)
            if model.__class__.__name__ == "CAFNetDecoupled":
                out = model.last_freq_pred

            location = torch.where(raw_label != 0)
            pred = out[location]
            label = raw_label[location]

            total_preds = torch.cat((total_preds, pred.cpu()), 0)
            total_labels = torch.cat((total_labels, label.cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def evaluate(model, device, loader, sideEffectsGraph, DF, not_FC, result_folder, id):
    total_preds = torch.Tensor()
    total_assoc_preds = torch.Tensor()
    total_label = torch.Tensor()
    singleDrug_auc = []
    singleDrug_aupr = []
    model.eval()
    torch.cuda.manual_seed(42)
    # 对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用with torch.no_grad():来强制之后的内容不进行计算图构建
    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for data in loader:
            # 查找被mask的数据
            label = data.y
            data = data.to(device)
            output, _, _ = model(data, sideEffectsGraph, DF, not_FC)
            pred = output.cpu()
            assoc_pred = model.last_assoc_logits.detach().cpu() if model.__class__.__name__ == "CAFNetDecoupled" else pred

            total_preds = torch.cat((total_preds, pred), 0)
            total_assoc_preds = torch.cat((total_assoc_preds, assoc_pred), 0)
            total_label = torch.cat((total_label, label), 0)
            # test batch size must be 1
            pred = assoc_pred.numpy().flatten()
            label = (label.numpy().flatten() != 0).astype(int)

            singleDrug_auc.append(roc_auc_score(label, pred))
            singleDrug_aupr.append(average_precision_score(label, pred))
    if id == 1:
        pred_result = pd.read_csv(result_folder + '/blind_pred.csv', header=0, index_col=None).values
        raw_result = pd.read_csv(result_folder + '/blind_raw.csv', header=0, index_col=None).values
    else:
        pred_result = pd.read_csv(result_folder + '/blind_pred.csv', header=None, index_col=None).values
        raw_result = pd.read_csv(result_folder + '/blind_raw.csv', header=None, index_col=None).values
    print(pred_result.shape)


    pred_result = pd.DataFrame(np.vstack((pred_result, total_preds.numpy())))
    raw_result = pd.DataFrame(np.vstack((raw_result, total_label.numpy())))
    pred_result.to_csv(result_folder + '/blind_pred.csv', header=False, index=False)
    raw_result.to_csv(result_folder + '/blind_raw.csv', header=False, index=False)


    drugAUC = sum(singleDrug_auc) / len(singleDrug_auc)
    drugAUPR = sum(singleDrug_aupr) / len(singleDrug_aupr)
    total_preds = total_preds.numpy()
    total_assoc_preds = total_assoc_preds.numpy()
    total_label = total_label.numpy()

    pos = total_assoc_preds[np.where(total_label)]
    pos_label = np.ones(len(pos))

    neg = total_assoc_preds[np.where(total_label == 0)]
    neg_label = np.zeros(len(neg))

    y = np.hstack((pos, neg))
    y_true = np.hstack((pos_label, neg_label))
    auc_all = roc_auc_score(y_true, y)
    aupr_all = average_precision_score(y_true, y)

    # others

    Te = {}
    Te_all = {}
    Te_pairs = np.where(total_label)
    Te_pairs = np.array(Te_pairs).transpose()

    for pair in Te_pairs:
        drug_id = pair[0]
        SE_id = pair[1]
        if drug_id not in Te:
            Te[drug_id] = [SE_id]
        else:
            Te[drug_id].append(SE_id)
    shape = total_label.shape
    for i in range(shape[0]):
        Te_all[i] = [i for i in range(shape[1])]

    positions = [1, 5, 10, 15]
    map_value, auc_value, ndcg, prec, rec = evaluate_others(total_preds, Te_all, Te, positions)

    p1, p5, p10, p15 = prec[0], prec[1], prec[2], prec[3]
    r1, r5, r10, r15 = rec[0], rec[1], rec[2], rec[3]
    return auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15


def evaluate_metrics(model, device, loader, sideEffectsGraph, DF, not_FC):
    total_preds = torch.Tensor()
    total_assoc_preds = torch.Tensor()
    total_label = torch.Tensor()
    singleDrug_auc = []
    singleDrug_aupr = []
    model.eval()
    torch.cuda.manual_seed(42)
    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for data in loader:
            label = data.y
            data = data.to(device)
            output, _, _ = model(data, sideEffectsGraph, DF, not_FC)
            pred = output.cpu()
            assoc_pred = model.last_assoc_logits.detach().cpu() if model.__class__.__name__ == "CAFNetDecoupled" else pred

            total_preds = torch.cat((total_preds, pred), 0)
            total_assoc_preds = torch.cat((total_assoc_preds, assoc_pred), 0)
            total_label = torch.cat((total_label, label), 0)

            pred = assoc_pred.numpy().flatten()
            label = (label.numpy().flatten() != 0).astype(int)
            singleDrug_auc.append(roc_auc_score(label, pred))
            singleDrug_aupr.append(average_precision_score(label, pred))

    drugAUC = sum(singleDrug_auc) / len(singleDrug_auc)
    drugAUPR = sum(singleDrug_aupr) / len(singleDrug_aupr)
    total_preds = total_preds.numpy()
    total_assoc_preds = total_assoc_preds.numpy()
    total_label = total_label.numpy()

    pos = total_assoc_preds[np.where(total_label)]
    pos_label = np.ones(len(pos))
    neg = total_assoc_preds[np.where(total_label == 0)]
    neg_label = np.zeros(len(neg))

    y = np.hstack((pos, neg))
    y_true = np.hstack((pos_label, neg_label))
    auc_all = roc_auc_score(y_true, y)
    aupr_all = average_precision_score(y_true, y)

    Te = {}
    Te_all = {}
    Te_pairs = np.where(total_label)
    Te_pairs = np.array(Te_pairs).transpose()
    for pair in Te_pairs:
        drug_id = pair[0]
        SE_id = pair[1]
        if drug_id not in Te:
            Te[drug_id] = [SE_id]
        else:
            Te[drug_id].append(SE_id)
    shape = total_label.shape
    for i in range(shape[0]):
        Te_all[i] = [i for i in range(shape[1])]

    positions = [1, 5, 10, 15]
    map_value, auc_value, ndcg, prec, rec = evaluate_others(total_preds, Te_all, Te, positions)
    p1, p5, p10, p15 = prec[0], prec[1], prec[2], prec[3]
    r1, r5, r10, r15 = rec[0], rec[1], rec[2], rec[3]
    return auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15


def load_side_node_features(side_feature_file=None, side_feature_concat=False):
    node_label = scipy.io.loadmat(side_effect_label)['node_label']
    if side_feature_file:
        semantic_feat = np.load(side_feature_file)
        if semantic_feat.shape[0] != node_label.shape[0]:
            raise ValueError(
                'side feature row count mismatch: {} vs {}'.format(
                    semantic_feat.shape[0], node_label.shape[0]
                )
            )
        if side_feature_concat:
            node_label = np.hstack([node_label, semantic_feat])
        else:
            node_label = semantic_feat
    return node_label


def main(modeling, metric, train_batch, lr, num_epoch, knn, weight_decay, lamb, log_interval, cuda_name, frequencyMat,
         id, mask, result_folder, save_model, DF, not_FC, output_dim, eps, pca,
         use_cross_attn=True, fusion_mode="gate", gate_mode="new", fusion_alpha=0.5, gat_dropout=0.0,
         rank_score_mix=0.7, dual_task=False, assoc_weight=1.0, freq_weight=0.2, rank_weight=0.1,
         bpr_samples=32, seed=42, side_feature_file=None, side_feature_concat=False,
         drug_feature_file=None, evidence_dropout=0.1, dataset_suffix='',
         pop_weight=0.0, bias_weight=1.0, list_weight=0.0,
         assoc_base_weight=1.0, assoc_residual_weight=1.0,
         prevalence_debias=False, debias_gamma=1.0, rare_pos_boost=1.0,
         matched_bpr=False, prevalence_bins=10, external_pairs_dir=None,
         external_weight=0.0, external_samples_per_drug=64, external_target="rank"):
    print('\n=======================================================================================')
    print('\n第 {} 次训练：\n'.format(id))
    print('model: ', modeling.__name__)
    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
    print('Batch size: ', train_batch)
    print('Lambda: ', lamb)
    print('weight_decay: ', weight_decay)
    print('KNN: ', knn)
    print('metric: ', metric)
    print('tenfold: ', tenfold)
    print('DF: ', DF)
    print('not_FC: ', not_FC)
    print('output_dim: ', output_dim)
    print('Eps: ', eps)
    print('PCA: ', pca)
    print('Seed: ', seed + id - 1)
    set_global_seed(seed + id - 1)
    if modeling.__name__ in ["CAFNet", "CAFNetV2", "CAFNetDecoupled"]:
        print('use_cross_attn: ', use_cross_attn)
        print('fusion_mode: ', fusion_mode)
        print('gate_mode: ', gate_mode)
        print('fusion_alpha: ', fusion_alpha)
        print('gat_dropout: ', gat_dropout)
    if modeling.__name__ in ["CAFNetV2", "CAFNetDecoupled"]:
        print('rank_score_mix: ', rank_score_mix)
        print('dual_task: ', dual_task)
        print('assoc_weight: ', assoc_weight)
        print('freq_weight: ', freq_weight)
        print('rank_weight: ', rank_weight)
        print('bpr_samples: ', bpr_samples)
    if modeling.__name__ == "CAFNetDecoupled":
        print('pop_weight: ', pop_weight)
        print('bias_weight: ', bias_weight)
        print('list_weight: ', list_weight)
        print('assoc_base_weight: ', assoc_base_weight)
        print('assoc_residual_weight: ', assoc_residual_weight)
        print('prevalence_debias: ', prevalence_debias)
        print('debias_gamma: ', debias_gamma)
        print('rare_pos_boost: ', rare_pos_boost)
        print('matched_bpr: ', matched_bpr)
        print('prevalence_bins: ', prevalence_bins)
        print('external_pairs_dir: ', external_pairs_dir)
        print('external_weight: ', external_weight)
        print('external_samples_per_drug: ', external_samples_per_drug)
        print('external_target: ', external_target)
    if side_feature_file:
        print('side_feature_file: ', side_feature_file)
        print('side_feature_concat: ', side_feature_concat)
    drug_evidence_dim = 0
    if drug_feature_file:
        drug_features_for_dim = np.load(drug_feature_file)
        drug_evidence_dim = int(drug_features_for_dim.shape[1])
        print('drug_feature_file: ', drug_feature_file)
        print('drug_evidence_dim: ', drug_evidence_dim)
        print('evidence_dropout: ', evidence_dropout)

    model_st = modeling.__name__
    train_losses = []

    print('\nrunning on ', model_st + '_' + dataset)
    processed_raw = raw_file

    if not os.path.isfile(processed_raw):
        print('Missing raw FrequencyMat, exit!!!')
        exit(1)

    # 生成副作用的graph信息
    train_frequency_for_weights = frequencyMat.copy()
    frequencyMat = frequencyMat.T
    if pca:
        pca_ = PCA(n_components=256)
        similarity_pca = pca_.fit_transform(frequencyMat)
        print('PCA 信息保留比例： ')
        print(sum(pca_.explained_variance_ratio_))
        A = kneighbors_graph(similarity_pca, knn, mode='connectivity', metric=metric, include_self=False)
    else:
        A = kneighbors_graph(frequencyMat, knn, mode='connectivity', metric=metric, include_self=False)
    G = nx.from_numpy_matrix(A.todense())
    edges = []
    for (u, v) in G.edges():
        edges.append([u, v])
        edges.append([v, u])

    edges = np.array(edges).T
    edges = torch.tensor(edges, dtype=torch.long)

    node_label = load_side_node_features(side_feature_file, side_feature_concat)
    feat = torch.tensor(node_label, dtype=torch.float)
    sideEffectsGraph = Data(x=feat, edge_index=edges)
    input_dim_e = int(feat.shape[1])

    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']

    # make data_WS Pytorch mini-batch processing ready
    train_data = myDataset(root='data_ICS', dataset=dataset + dataset_suffix + '_blind_train' + str(id - 1))
    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
    test_data = myDataset(root='data_ICS', dataset=dataset + dataset_suffix + '_blind_test' + str(id - 1))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    side_weights = None
    side_prevalence = None
    ext_drug_ids = ext_pos_idx = ext_neg_idx = None
    train_prevalence = (train_frequency_for_weights != 0).mean(axis=0).astype(np.float32)
    if prevalence_debias:
        prevalence = train_prevalence
        base_weight = np.power(1.0 - prevalence, float(debias_gamma)).astype(np.float32)
        if rare_pos_boost != 1.0:
            q50 = np.quantile(prevalence, 0.5)
            base_weight[prevalence <= q50] *= float(rare_pos_boost)
        base_weight = base_weight / max(float(base_weight.mean()), 1e-6)
        side_weights = torch.tensor(base_weight, dtype=torch.float32)
    if matched_bpr:
        side_prevalence = torch.tensor(train_prevalence, dtype=torch.float32)
    if external_pairs_dir and external_weight > 0:
        ext_path = os.path.join(external_pairs_dir, 'offsides_contrastive_pairs_fold{}.npz'.format(id - 1))
        if not os.path.exists(ext_path):
            raise FileNotFoundError('External contrastive pair file not found: {}'.format(ext_path))
        ext = np.load(ext_path)
        ext_drug_ids = torch.tensor(ext['train_local_index'], dtype=torch.long)
        ext_pos_idx = torch.tensor(ext['pos_side_index'], dtype=torch.long)
        ext_neg_idx = torch.tensor(ext['neg_side_index'], dtype=torch.long)
        print('external contrastive pairs loaded: ', len(ext_drug_ids))

    print('CPU/GPU: ', torch.cuda.is_available())

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    if modeling.__name__ in ["CAFNet", "CAFNetV2", "CAFNetDecoupled"]:
        model = modeling(
            input_dim=input_dim,
            input_dim_e=input_dim_e,
            output_dim=output_dim,
            use_cross_attn=use_cross_attn,
            fusion_mode=fusion_mode,
            gate_mode=gate_mode,
            fusion_alpha=fusion_alpha,
            gat_dropout=gat_dropout,
            **({"rank_score_mix": rank_score_mix} if modeling.__name__ in ["CAFNetV2", "CAFNetDecoupled"] else {}),
            **({"pop_weight": pop_weight, "bias_weight": bias_weight,
                "assoc_base_weight": assoc_base_weight, "assoc_residual_weight": assoc_residual_weight,
                "drug_evidence_dim": drug_evidence_dim, "evidence_dropout": evidence_dropout}
               if modeling.__name__ == "CAFNetDecoupled" else {}),
        ).to(device)
        if modeling.__name__ == "CAFNetDecoupled":
            train_observed = frequencyMat.T != 0
            side_popularity = train_observed.sum(axis=0) / max(1.0, float(train_observed.shape[0]))
            global_mean = frequencyMat.T[train_observed].mean() if np.any(train_observed) else 0.0
            model.set_frequency_priors(side_popularity=side_popularity, global_mean=global_mean)
    else:
        model = modeling(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model_file_name = str(id) + 'Blind_MF_' + model_st + '_epoch=' + str(num_epoch) + '.model'
    result_log = result_folder + '/' + model_st + '_result.csv'
    metrics_log = result_folder + '/' + model_st + '_metrics.csv'
    if not os.path.exists(metrics_log):
        with open(metrics_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'auc_all', 'aupr_all', 'drugAUC', 'drugAUPR'])
    loss_fig_name = str(id) + model_st + '_loss'
    pearson_fig_name = str(id) + model_st + '_pearson'
    MSE_fig_name = str(id) + model_st + '_MSE'

    for epoch in range(num_epoch):
        train_loss = train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, lamb=lamb,
                           epoch=epoch + 1, log_interval=log_interval, sideEffectsGraph=sideEffectsGraph, raw=raw,
                           id=id, DF=DF, not_FC=not_FC, eps=eps, dual_task=dual_task,
                           assoc_weight=assoc_weight, freq_weight=freq_weight, rank_weight=rank_weight,
                           bpr_samples=bpr_samples, list_weight=list_weight, side_weights=side_weights,
                           matched_bpr=matched_bpr, side_prevalence=side_prevalence,
                           prevalence_bins=prevalence_bins,
                           ext_drug_ids=ext_drug_ids, ext_pos_idx=ext_pos_idx, ext_neg_idx=ext_neg_idx,
                           external_weight=external_weight,
                           external_samples_per_drug=external_samples_per_drug,
                           external_target=external_target)
        train_losses.append(train_loss)

        if (epoch + 1) % 10 == 0:
            auc_all, aupr_all, drugAUC, drugAUPR, _, _, _, _, _, _, _, _, _, _ = evaluate_metrics(
                model=model, device=device, loader=test_loader, sideEffectsGraph=sideEffectsGraph,
                DF=DF, not_FC=not_FC)
            with open(metrics_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, train_loss, auc_all, aupr_all, drugAUC, drugAUPR])

    torch.save(model, "model_cold.pt")
    test_labels, test_preds = predict(model=model, device=device, loader=test_loader,
                                      sideEffectsGraph=sideEffectsGraph, DF=DF, not_FC=not_FC)
    ret_test = [mse(test_labels, test_preds), pearson(test_labels, test_preds), rmse(test_labels, test_preds),
                spearman(test_labels, test_preds), MAE(test_labels, test_preds)]
    test_pearsons, test_rMSE, test_spearman, test_MAE = ret_test[1], ret_test[2], ret_test[3], ret_test[4]
    auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15 = evaluate(model=model,
                                                                                                         device=device,
                                                                                                         loader=test_loader,
                                                                                                         sideEffectsGraph=sideEffectsGraph,
                                                                                                         DF=DF,
                                                                                                         not_FC=not_FC,
                                                                                                         result_folder=result_folder,
                                                                                              id=id)
    if save_model:
        checkpointsFolder = result_folder + '/checkpoints/'
        isCheckpointExist = os.path.exists(checkpointsFolder)
        if not isCheckpointExist:
            os.makedirs(checkpointsFolder)
        torch.save(model.state_dict(), checkpointsFolder + model_file_name)

    result = [test_pearsons, test_rMSE, test_spearman, test_MAE, auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg,
              p1, p5, p10, p15, r1, r5, r10, r15]
    with open(result_log, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)

    print('Test:\nPearson: {:.5f}\trMSE: {:.5f}\tSpearman: {:.5f}\tMAE: {:.5f}'.format(result[0], result[1], result[2],
                                                                                       result[3]))
    print('\tall AUC: {:.5f}\tall AUPR: {:.5f}\tdrug AUC: {:.5f}\tdrug AUPR: {:.5f}'.format(result[4], result[5],
                                                                                            result[6], result[7]))
    print('\tMAP: {:.5f}\tnDCG@10: {:.5f}'.format(map_value, ndcg))
    print('\tP@1: {:.5f}\tP@5: {:.5f}\tP@10: {:.5f}\tP@15: {:.5f}'.format(p1, p5, p10, p15))
    print('\tR@1: {:.5f}\tR@5: {:.5f}\tR@10: {:.5f}\tR@15: {:.5f}'.format(r1, r5, r10, r15))
    # train loss
    my_draw_loss(train_losses, loss_fig_name, result_folder)
    # # test pearson
    # draw_pearson(test_pearsons, pearson_fig_name, result_folder)
    # # test mse
    # my_draw_mse(test_MSE, test_rMSE, MSE_fig_name, result_folder)


if __name__ == '__main__':

    total_start = datetime.datetime.now()
    # 参数定义
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0,
                        help='0:CAFNet, 1:A3Net, 2:CAFNetV2, 3:CAFNetDecoupled')
    parser.add_argument('--metric', type=int, required=False, default=0, help='0: cosine, 1: jaccard, 2: euclidean')
    parser.add_argument('--train_batch', type=int, required=False, default=10, help='Batch size training set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, required=False, default=0.001, help='weight_decay')
    parser.add_argument('--lamb', type=float, required=False, default=0.03, help='LAMBDA')
    parser.add_argument('--epoch', type=int, required=False, default=3000, help='Number of epoch')
    parser.add_argument('--knn', type=int, required=False, default=5, help='Number of KNN')
    parser.add_argument('--log_interval', type=int, required=False, default=20, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default='cuda:0', help='Cuda')
    parser.add_argument('--dim', type=int, required=False, default=200, help='output dim, <= 109')
    parser.add_argument('--eps', type=float, required=False, default=0.5, help='regard 0 as eps when training')

    parser.add_argument('--tenfold', action='store_true', default=False, help='use 10 folds Cross-validation ')
    parser.add_argument('--save_model', action='store_true', default=False, help='save model and features')
    parser.add_argument('--DF', action='store_true', default=False, help='use DF decoder')
    parser.add_argument('--not_FC', action='store_true', default=False, help='not use Linear layers')
    parser.add_argument('--PCA', action='store_true', default=False, help='use PCA')
    parser.add_argument('--mask_file', type=str, required=False, default=blind_mask_mat_file,
                        help='Cold-start mask .mat file with mask0..mask9')
    parser.add_argument('--result_prefix', type=str, required=False, default='ICS',
                        help='Result prefix used in output folder names, e.g. ICS or scaffold')
    parser.add_argument('--short_result_name', action='store_true', default=False,
                        help='Use a compact result folder name for long ablation parameter sets')
    parser.add_argument('--max_folds', type=int, required=False, default=10,
                        help='Maximum number of folds to run, useful for smoke tests')
    parser.add_argument('--no_cross_attn', action='store_true', default=False,
                        help='Disable CAFNet cross-attention branch')
    parser.add_argument('--fusion_mode', type=str, required=False, default='gate',
                        choices=['gate', 'fixed', 'none'], help='CAFNet graph/raw feature fusion mode')
    parser.add_argument('--gate_mode', type=str, required=False, default='new',
                        choices=['new', 'old'], help='CAFNet gate fusion implementation')
    parser.add_argument('--fusion_alpha', type=float, required=False, default=0.5,
                        help='CAFNet fixed-fusion alpha')
    parser.add_argument('--gat_dropout', type=float, required=False, default=0.0,
                        help='CAFNet GATConv attention dropout for experimental runs')
    parser.add_argument('--rank_score_mix', type=float, required=False, default=0.7,
                        help='CAFNetV2 final score mix: association weight in [0, 1]')
    parser.add_argument('--dual_task', action='store_true', default=False,
                        help='Use CAFNetV2 asymmetric association/frequency/ranking loss')
    parser.add_argument('--assoc_weight', type=float, required=False, default=1.0,
                        help='CAFNetV2 association BCE loss weight')
    parser.add_argument('--freq_weight', type=float, required=False, default=0.2,
                        help='CAFNetV2 frequency Huber loss weight')
    parser.add_argument('--rank_weight', type=float, required=False, default=0.1,
                        help='CAFNetV2 BPR ranking loss weight')
    parser.add_argument('--bpr_samples', type=int, required=False, default=32,
                        help='CAFNetV2 positive/negative samples per row for BPR')
    parser.add_argument('--seed', type=int, required=False, default=42,
                        help='Base random seed; fold index is added inside each run')
    parser.add_argument('--side_feature_file', type=str, required=False, default=None,
                        help='Optional .npy side-effect semantic feature matrix')
    parser.add_argument('--side_feature_concat', action='store_true', default=False,
                        help='Concatenate semantic side features to original node_label instead of replacing it')
    parser.add_argument('--drug_feature_file', type=str, required=False, default=None,
                        help='Optional .npy drug-level external evidence feature matrix, aligned to raw_frequency_750 drugs')
    parser.add_argument('--drug_feature_tag', type=str, required=False, default='drugfeat',
                        help='Short tag for dataset/result names when drug_feature_file is used')
    parser.add_argument('--evidence_dropout', type=float, required=False, default=0.1,
                        help='Dropout in the drug evidence encoder')
    parser.add_argument('--pop_weight', type=float, required=False, default=0.0)
    parser.add_argument('--bias_weight', type=float, required=False, default=1.0)
    parser.add_argument('--list_weight', type=float, required=False, default=0.0)
    parser.add_argument('--assoc_base_weight', type=float, required=False, default=1.0)
    parser.add_argument('--assoc_residual_weight', type=float, required=False, default=1.0)
    parser.add_argument('--prevalence_debias', action='store_true', default=False,
                        help='Up-weight rare/mid-prevalence positive labels in association, frequency, BPR, and listwise losses')
    parser.add_argument('--debias_gamma', type=float, required=False, default=1.0,
                        help='Exponent for inverse prevalence weighting: (1 - prevalence)^gamma')
    parser.add_argument('--rare_pos_boost', type=float, required=False, default=1.0,
                        help='Extra multiplier for side effects below the training-fold median prevalence')
    parser.add_argument('--matched_bpr', action='store_true', default=False,
                        help='Use prevalence-bin matched negatives in BPR loss')
    parser.add_argument('--prevalence_bins', type=int, required=False, default=10,
                        help='Number of prevalence bins for matched BPR negative sampling')
    parser.add_argument('--external_pairs_dir', type=str, required=False, default=None,
                        help='Directory containing fold-specific external contrastive .npz pair files')
    parser.add_argument('--external_weight', type=float, required=False, default=0.0,
                        help='Weight for external positive vs prevalence-matched negative contrastive loss')
    parser.add_argument('--external_samples_per_drug', type=int, required=False, default=64,
                        help='Maximum external contrastive pairs sampled per batch drug')
    parser.add_argument('--external_target', type=str, required=False, default='rank',
                        choices=['rank', 'assoc'],
                        help='Score tensor used by external contrastive loss')
    args = parser.parse_args()

    modeling = [CAFNet, A3_Net, CAFNetV2, CAFNetDecoupled][args.model]
    metric = ['cosine', 'jaccard', 'euclidean'][args.metric]
    train_batch = args.train_batch
    lr = args.lr
    knn = args.knn
    num_epoch = args.epoch
    weight_decay = args.wd
    lamb = args.lamb
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    tenfold = args.tenfold
    save_model = args.save_model
    DF = args.DF
    not_FC = args.not_FC
    output_dim = args.dim
    eps = args.eps
    pca = args.PCA
    mask_file = args.mask_file
    result_prefix = args.result_prefix
    short_result_name = args.short_result_name
    max_folds = args.max_folds
    use_cross_attn = not args.no_cross_attn
    fusion_mode = args.fusion_mode
    gate_mode = args.gate_mode
    fusion_alpha = args.fusion_alpha
    gat_dropout = args.gat_dropout
    rank_score_mix = args.rank_score_mix
    dual_task = args.dual_task or modeling.__name__ in ["CAFNetV2", "CAFNetDecoupled"]
    assoc_weight = args.assoc_weight
    freq_weight = args.freq_weight
    rank_weight = args.rank_weight
    bpr_samples = args.bpr_samples
    seed = args.seed
    side_feature_file = args.side_feature_file
    side_feature_concat = args.side_feature_concat
    drug_feature_file = args.drug_feature_file
    drug_feature_tag = args.drug_feature_tag
    evidence_dropout = args.evidence_dropout
    pop_weight = args.pop_weight
    bias_weight = args.bias_weight
    list_weight = args.list_weight
    assoc_base_weight = args.assoc_base_weight
    assoc_residual_weight = args.assoc_residual_weight
    prevalence_debias = args.prevalence_debias
    debias_gamma = args.debias_gamma
    rare_pos_boost = args.rare_pos_boost
    matched_bpr = args.matched_bpr
    prevalence_bins = args.prevalence_bins
    external_pairs_dir = args.external_pairs_dir
    external_weight = args.external_weight
    external_samples_per_drug = args.external_samples_per_drug
    external_target = args.external_target

    processed_mask_mat = mask_file
    if not os.path.isfile(processed_mask_mat):
        print('Missing data_WS files, generating......')
        generateMat()

    result_folder = './result_ICS/'
    cafnet_tag = ''
    if modeling.__name__ in ["CAFNet", "CAFNetV2", "CAFNetDecoupled"] and (
            (not use_cross_attn) or fusion_mode != 'gate' or gate_mode != 'new' or
            abs(fusion_alpha - 0.5) > 1e-12 or abs(gat_dropout) > 1e-12 or
            modeling.__name__ in ["CAFNetV2", "CAFNetDecoupled"]):
        cafnet_tag = '_cross=' + str(use_cross_attn) + '_fusion=' + fusion_mode + '_gate=' + gate_mode + \
                     '_fa=' + str(fusion_alpha) + '_gatdrop=' + str(gat_dropout)
        if modeling.__name__ in ["CAFNetV2", "CAFNetDecoupled"]:
            cafnet_tag += '_mix=' + str(rank_score_mix) + '_aw=' + str(assoc_weight) + \
                          '_fw=' + str(freq_weight) + '_rw=' + str(rank_weight)
        if modeling.__name__ == "CAFNetDecoupled":
            cafnet_tag += '_popw=' + str(pop_weight) + '_biasw=' + str(bias_weight) + \
                          '_listw=' + str(list_weight) + '_abw=' + str(assoc_base_weight) + \
                          '_arw=' + str(assoc_residual_weight)
            if prevalence_debias:
                cafnet_tag += '_pdeb=True_dg=' + str(debias_gamma) + '_rb=' + str(rare_pos_boost)
            if matched_bpr:
                cafnet_tag += '_mbpr=True_bins=' + str(prevalence_bins)
            if external_pairs_dir and external_weight > 0:
                cafnet_tag += '_extw=' + str(external_weight) + '_extsamp=' + str(external_samples_per_drug) + \
                              '_exttarget=' + str(external_target)
    if side_feature_file:
        cafnet_tag += '_sidefeat=' + ('concat' if side_feature_concat else 'replace')
    dataset_suffix = ''
    if drug_feature_file:
        dataset_suffix = '_' + drug_feature_tag
        cafnet_tag += '_drugfeat=' + drug_feature_tag + '_edrop=' + str(evidence_dropout)

    if short_result_name:
        result_folder += ('10' if tenfold else '1') + result_prefix + '_' + modeling.__name__
    elif tenfold:
        result_folder += '10' + result_prefix + '_' + modeling.__name__ + '_knn=' + str(knn) + '_wd=' + str(
            weight_decay) + '_epoch=' + str(num_epoch) + '_lamb=' + str(lamb) + '_lr' + str(lr) + '_dim=' + str(
            output_dim) + '_eps=' + str(eps) + '_DF=' + str(DF) + '_PCA=' + str(pca) + '_not-FC=' + str(not_FC) + cafnet_tag + '_' + str(metric)
    else:
        result_folder += '1' + result_prefix + '_' + modeling.__name__ + '_knn=' + str(knn) + '_wd=' + str(
            weight_decay) + '_epoch=' + str(num_epoch) + '_lamb=' + str(lamb) + '_lr' + str(lr) + '_dim=' + str(
            output_dim) + '_eps=' + str(eps) + '_DF=' + str(DF) + '_PCA=' + str(pca) + '_not-FC=' + str(not_FC) + cafnet_tag + '_' + str(metric)

    isExist = os.path.exists(result_folder)
    if not isExist:
        os.makedirs(result_folder)
    else:
        # 清空原文件 添加表头
        shutil.rmtree(result_folder)
        os.makedirs(result_folder)

    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']
    pred_result = result_folder + '/blind_pred.csv'
    pred_ = pd.DataFrame(columns=[i for i in range(raw.shape[1])])
    pred_.to_csv(pred_result, header=True, index=False)
    raw_result = result_folder + '/blind_raw.csv'
    raw_ = pd.DataFrame(columns=[i for i in range(raw.shape[1])])
    raw_.to_csv(raw_result, header=True, index=False)

    result_log = result_folder + '/' + modeling.__name__ + '_result.csv'
    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']

    with open(result_log, 'w', newline='') as f:
        fieldnames = ['pearson', 'rMSE', 'spearman', 'MAE', 'auc_all', 'aupr_all', 'drugAUC', 'drugAUPR', 'MAP', 'nDCG',
                      'P1', 'P5', 'P10', 'P15', 'R1', 'R5', 'R10', 'R15']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for (id, frequencyMat, mask) in split_data(tenfold, mask_file=mask_file, max_folds=max_folds,
                                               drug_feature_file=drug_feature_file,
                                               dataset_suffix=dataset_suffix):
        start = datetime.datetime.now()
        main(modeling, metric, train_batch, lr, num_epoch, knn, weight_decay, lamb, log_interval, cuda_name,
             frequencyMat, id + 1, mask, result_folder, save_model, DF, not_FC, output_dim, eps, pca,
             use_cross_attn, fusion_mode, gate_mode, fusion_alpha, gat_dropout,
             rank_score_mix, dual_task, assoc_weight, freq_weight, rank_weight, bpr_samples, seed,
             side_feature_file, side_feature_concat, drug_feature_file, evidence_dropout, dataset_suffix,
             pop_weight, bias_weight, list_weight,
             assoc_base_weight, assoc_residual_weight,
             prevalence_debias, debias_gamma, rare_pos_boost,
             matched_bpr, prevalence_bins,
             external_pairs_dir, external_weight, external_samples_per_drug, external_target)
        end = datetime.datetime.now()
        print('本次运行时间：{}\t'.format(end - start))

    data = pd.read_csv(result_log)
    L = len(data.rMSE)
    avg = [sum(data.pearson) / L, sum(data.rMSE) / L, sum(data.spearman) / L, sum(data.MAE) / L, sum(data.auc_all) / L,
           sum(data.aupr_all) / L, sum(data.drugAUC) / L, sum(data.drugAUPR) / L, sum(data.MAP) / L, sum(data.nDCG) / L,
           sum(data.P1) / L, sum(data.P5) / L, sum(data.P10) / L, sum(data.P15) / L, sum(data.R1) / L, sum(data.R5) / L,
           sum(data.R10) / L, sum(data.R15) / L]
    print('\n\tavg pearson: {:.4f}\tavg rMSE: {:.4f}\tavg spearman: {:.4f}\tavg MAE: {:.4f}'.format(avg[0], avg[1],
                                                                                                    avg[2], avg[3]))
    print('\tavg all AUC: {:.4f}\tavg all AUPR: {:.4f}\tavg drug AUC: {:.4f}\tavg drug AUPR: {:.4f}'.format(avg[4],
                                                                                                            avg[5],
                                                                                                            avg[6],
                                                                                                            avg[7]))
    print('\tavg MAP: {:.4f}\tavg nDCG@10: {:.4f}'.format(avg[8], avg[9]))
    print('\tavg P@1: {:.4f}\tavg P@5: {:.4f}\tavg P@10: {:.4f}\tavg P@15: {:.4f}'.format(avg[10], avg[11], avg[12],
                                                                                          avg[13]))
    print('\tavg R@1: {:.4f}\tavg R@5: {:.4f}\tavg R@10: {:.4f}\tavg R@15: {:.4f}'.format(avg[14], avg[15], avg[16],
                                                                                          avg[17]))
    with open(result_log, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['avg'])
        writer.writerow(avg)
    total_end = datetime.datetime.now()
    print('总体运行时间：{}\t'.format(total_end - total_start))
