import os

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
mask_mat_file = 'data/mask_mat_750.mat'
side_effect_label = 'data/side_effect_label_750.mat'
input_dim = 109


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def loss_fun(output, label, lam, eps, eligible_mask=None):
    if eligible_mask is None:
        eligible_mask = torch.ones_like(label, dtype=torch.bool)
    eligible_mask = eligible_mask.to(device=label.device, dtype=torch.bool)
    x0 = torch.where((label == 0) & eligible_mask)
    x1 = torch.where((label != 0) & eligible_mask)
    loss = torch.sum((output[x1] - label[x1]) ** 2) + lam * torch.sum((output[x0] - eps) ** 2)
    return loss


def bpr_loss(logits, labels, samples_per_row=32, eligible_mask=None):
    labels = labels.to(device=logits.device)
    if eligible_mask is None:
        eligible_mask = torch.ones_like(labels, dtype=torch.bool)
    eligible_mask = eligible_mask.to(device=logits.device, dtype=torch.bool)
    losses = []
    for row_idx in range(logits.size(0)):
        pos_idx = torch.where((labels[row_idx] != 0) & eligible_mask[row_idx])[0]
        neg_idx = torch.where((labels[row_idx] == 0) & eligible_mask[row_idx])[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        n = min(int(samples_per_row), pos_idx.numel(), neg_idx.numel())
        pos_perm = torch.randperm(pos_idx.numel(), device=logits.device)[:n]
        neg_perm = torch.randperm(neg_idx.numel(), device=logits.device)[:n]
        pos_scores = logits[row_idx, pos_idx[pos_perm]]
        neg_scores = logits[row_idx, neg_idx[neg_perm]]
        losses.append(-F.logsigmoid(pos_scores - neg_scores).mean())
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def listnet_loss(scores, labels, temperature=1.0, eligible_mask=None, target_mode="binary"):
    labels = labels.to(device=scores.device, dtype=scores.dtype)
    if eligible_mask is None:
        eligible_mask = torch.ones_like(labels, dtype=torch.bool)
    eligible_mask = eligible_mask.to(device=scores.device, dtype=torch.bool)
    losses = []
    for row_idx in range(scores.size(0)):
        allowed = eligible_mask[row_idx]
        known = (labels[row_idx] != 0) & allowed
        if known.sum() == 0:
            continue
        allowed_labels = labels[row_idx, allowed]
        if target_mode == "binary":
            target = (allowed_labels != 0).to(dtype=scores.dtype)
            target_prob = target / target.sum().clamp_min(1.0)
        elif target_mode == "graded":
            target = torch.zeros_like(allowed_labels)
            target[allowed_labels != 0] = (
                allowed_labels[allowed_labels != 0] / max(float(temperature), 1e-6)
            )
            target_prob = F.softmax(target, dim=0)
        else:
            raise ValueError("target_mode must be 'binary' or 'graded'")
        pred_log_prob = F.log_softmax(scores[row_idx, allowed], dim=0)
        losses.append(-(target_prob * pred_log_prob).sum())
    if not losses:
        return scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def cafnet_v2_loss(model, output, label, lam, eps, assoc_weight, freq_weight, rank_weight, bpr_samples,
                   ordinal_weight=0.0, list_weight=0.0, eligible_mask=None,
                   listnet_target="binary"):
    assoc_logits = getattr(model, "last_assoc_logits", None)
    freq_pred = getattr(model, "last_freq_pred", None)
    if assoc_logits is None or freq_pred is None:
        flat_mask = None if eligible_mask is None else eligible_mask.flatten()
        return loss_fun(output.flatten(), label.flatten(), lam, eps, flat_mask)

    label = label.to(device=assoc_logits.device)
    if eligible_mask is None:
        eligible_mask = torch.ones_like(label, dtype=torch.bool)
    eligible_mask = eligible_mask.to(device=assoc_logits.device, dtype=torch.bool)
    target_assoc = (label != 0).to(dtype=assoc_logits.dtype)
    eligible_target = target_assoc[eligible_mask]
    pos_count = eligible_target.sum().clamp_min(1.0)
    neg_count = (eligible_target.numel() - eligible_target.sum()).clamp_min(1.0)
    pos_weight = torch.clamp(neg_count / pos_count, max=20.0)
    assoc_loss = F.binary_cross_entropy_with_logits(
        assoc_logits[eligible_mask], eligible_target, pos_weight=pos_weight
    )

    pos = (label != 0) & eligible_mask
    neg = (label == 0) & eligible_mask
    freq_loss = freq_pred.new_tensor(0.0)
    if pos.any():
        freq_loss = F.smooth_l1_loss(freq_pred[pos], label[pos])
    if neg.any():
        freq_loss = freq_loss + lam * F.smooth_l1_loss(freq_pred[neg], torch.full_like(freq_pred[neg], eps))

    rank_source = output if model.__class__.__name__ == "CAFNetDecoupled" else assoc_logits
    rank_loss = bpr_loss(
        rank_source, label, samples_per_row=bpr_samples, eligible_mask=eligible_mask
    )
    if list_weight > 0:
        rank_loss = rank_loss + list_weight * listnet_loss(
            rank_source, label, eligible_mask=eligible_mask, target_mode=listnet_target
        )
    ordinal_loss = output.new_tensor(0.0)
    ordinal_logits = getattr(model, "last_ordinal_logits", None)
    if ordinal_weight > 0 and ordinal_logits is not None:
        ordinal_target = torch.clamp(label.round().long(), min=0, max=ordinal_logits.size(-1) - 1)
        flat_eligible = eligible_mask.reshape(-1)
        ordinal_loss = F.cross_entropy(
            ordinal_logits.reshape(-1, ordinal_logits.size(-1))[flat_eligible],
            ordinal_target.reshape(-1)[flat_eligible],
        )
    return assoc_weight * assoc_loss + freq_weight * freq_loss + rank_weight * rank_loss + ordinal_weight * ordinal_loss


def _zscore_np(x):
    x = np.asarray(x, dtype=np.float64)
    std = np.nanstd(x)
    if std < 1e-8:
        return x * 0.0
    return (x - np.nanmean(x)) / std


def calibrate_prediction_matrix(pred, raw, mask, mode="none", seed=42):
    """Fit lightweight warm-start calibration on training entries only."""
    if mode is None or mode == "none":
        return pred

    pred = np.asarray(pred, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    train = raw * mask
    observed = train != 0
    if not np.any(observed):
        return pred

    global_mean = float(train[observed].mean())
    side_count = observed.sum(axis=0).astype(np.float64)
    drug_count = observed.sum(axis=1).astype(np.float64)
    side_mean = np.divide(train.sum(axis=0), np.maximum(side_count, 1.0))
    drug_mean = np.divide(train.sum(axis=1), np.maximum(drug_count, 1.0))
    side_mean[side_count == 0] = global_mean
    drug_mean[drug_count == 0] = global_mean
    prior = 0.5 * drug_mean[:, None] + 0.5 * side_mean[None, :]

    calibrated = pred.copy()
    if mode in ["regression_prior", "hybrid"]:
        best_alpha, best_rmse = 0.0, np.inf
        y = raw[observed]
        for alpha in np.linspace(0.0, 1.0, 21):
            candidate = (1.0 - alpha) * pred[observed] + alpha * prior[observed]
            rmse_val = float(np.sqrt(np.mean((candidate - y) ** 2)))
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_alpha = alpha
        calibrated = (1.0 - best_alpha) * pred + best_alpha * prior

    if mode in ["rank_pop", "hybrid"]:
        side_pop = side_count / max(1.0, float(raw.shape[0]))
        pop_score = np.tile(_zscore_np(side_pop), (raw.shape[0], 1))
        base_score = _zscore_np(calibrated)

        rng = np.random.default_rng(seed)
        pos_idx = np.argwhere(observed)
        neg_idx = np.argwhere((train == 0) & (mask != 0))
        if pos_idx.shape[0] > 0 and neg_idx.shape[0] > 0:
            n = min(pos_idx.shape[0], neg_idx.shape[0], 200000)
            pos_sel = pos_idx[rng.choice(pos_idx.shape[0], size=n, replace=False)]
            neg_sel = neg_idx[rng.choice(neg_idx.shape[0], size=n, replace=False)]
            best_gamma, best_margin = 0.0, -np.inf
            for gamma in np.linspace(0.0, 2.0, 21):
                candidate = base_score + gamma * pop_score
                margin = float(candidate[pos_sel[:, 0], pos_sel[:, 1]].mean() -
                               candidate[neg_sel[:, 0], neg_sel[:, 1]].mean())
                if margin > best_margin:
                    best_margin = margin
                    best_gamma = gamma
            calibrated = base_score + best_gamma * pop_score

    return calibrated.astype(np.float32)


def load_side_node_features(side_feature_file=None, side_feature_concat=False):
    node_label = scipy.io.loadmat(side_effect_label)['node_label']
    if side_feature_file:
        semantic_feat = np.load(side_feature_file)
        if semantic_feat.shape[0] != node_label.shape[0]:
            raise ValueError('side feature row count mismatch: {} vs {}'.format(semantic_feat.shape[0], node_label.shape[0]))
        node_label = np.hstack([node_label, semantic_feat]) if side_feature_concat else semantic_feat
    return node_label


def build_ablation_tag(use_cross_attn, fusion_mode, gate_mode):
    if not use_cross_attn:
        return f"noCA_{fusion_mode}_{gate_mode}"
    return f"CA_{fusion_mode}_{gate_mode}"

def generateMat(k=10):
    """
    灏嗙煩闃垫寜姣斾緥mask, 灏嗚mask鐨勯儴鍒嗗垎涓?0浠斤紝鐢熸垚10浠絤ask浣嶇疆鐭╅樀锛屼繚瀛樺湪./data_WS/processed/mask_mat.mat
    :return:
    """
    # 姣忔鍔犺浇閮芥妸涔嬪墠鐨勬暟鎹垹闄?
    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']

    # mask, get mask Mat
    index_pair = np.where(raw != 0)
    index_arr = np.arange(0, index_pair[0].shape[0], 1)
    np.random.shuffle(index_arr)
    x = []
    n = math.ceil(index_pair[0].shape[0] / k)
    for i in range(k):
        if i == k - 1:
            x.append(index_arr[0:].tolist())
        else:
            x.append(index_arr[0:n].tolist())
            index_arr = index_arr[n:]

    dic = {}
    for i in range(k):
        mask = np.ones(raw.shape)
        mask[index_pair[0][x[i]], index_pair[1][x[i]]] = 0
        dic['mask' + str(i)] = mask
    scipy.io.savemat(mask_mat_file, dic)


def split_data(tenfold=False, max_folds=10, start_fold=0):
    """
    璇诲彇 data/mask_mat.mat锛屾牴鎹師濮嬮鐜囩煩闃电敓鎴?0浠借mask鐨勯鐜囩煩闃靛苟yield
    :return:
    """
    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']
    mask_mat = scipy.io.loadmat(mask_mat_file)
    drug_dict, drug_smile = load_drug_smile(SMILES_file)
    print(len(drug_dict))

    simle_graph = convert2graph(drug_smile)
    dataset = 'drug_sideEffect'

    for i in range(int(start_fold), min(int(max_folds), 10)):
        mask = mask_mat['mask' + str(i)]
        frequencyMat = raw * mask
        # np.set_printoptions(precision=4)    # 淇濈暀鍥涗綅灏忔暟
        # np.savetxt('./data_WS/processed/frequencyMat.csv', frequencyMat, fmt='%.2f')
        data = myDataset(root='data_WS', dataset=dataset + '_data' + str(i), drug_simles=drug_smile,
                         frequencyMat=frequencyMat,
                         simle_graph=simle_graph)
        yield i, frequencyMat, mask

        if not tenfold and i == 0:
            break


# training function at each epoch
def train(model, device, train_loader, optimizer, lamb, epoch, log_interval, sideEffectsGraph, raw, id, DF, not_FC,
          eps, dual_task=False, assoc_weight=1.0, freq_weight=0.2, rank_weight=0.1, bpr_samples=32,
          ordinal_weight=0.0, list_weight=0.0, train_entry_mask=None,
          listnet_target="binary"):
    """

    :param model:
    :param device:
    :param train_loader: 鏁版嵁鍔犺浇鍣?
    :param optimizer: 浼樺寲鍣?
    :param epoch: 璁粌鏁?
    :param log_interval: 璁板綍闂撮殧
    :param sideEffectsGraph: 鍓綔鐢ㄥ浘淇℃伅锛?
    :param raw: 鍘熷鏁版嵁
    :param id: 绗琲d娆¤缁?绗琲d鎶橈級
    :return: 鏈璁粌鐨勫钩鍧囨崯澶?
    """
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    singleDrug_auc = []
    avg_loss = []
    for batch_idx, data in enumerate(train_loader):
        # 鏌ユ壘琚玬ask鐨勬暟鎹?
        index = [int(x[0]) for x in data.index]
        label = data.y
        sideEffectsGraph = sideEffectsGraph.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        out, x, x_e = model(data, sideEffectsGraph, DF, not_FC)

        pred = out.to(device)
        train_label = torch.FloatTensor(label).to(device)
        eligible_mask = None
        if train_entry_mask is not None:
            eligible_mask = torch.as_tensor(
                train_entry_mask[index], dtype=torch.bool, device=device
            )
        if dual_task:
            loss = cafnet_v2_loss(
                model, pred, train_label, lamb, eps,
                assoc_weight=assoc_weight,
                freq_weight=freq_weight,
                rank_weight=rank_weight,
                bpr_samples=bpr_samples,
                ordinal_weight=ordinal_weight,
                list_weight=list_weight,
                eligible_mask=eligible_mask,
                listnet_target=listnet_target,
            )
        else:
            flat_mask = None if eligible_mask is None else eligible_mask.flatten()
            loss = loss_fun(pred.flatten(), train_label.flatten(), lamb, eps, flat_mask)
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


def predict(model, device, loader, sideEffectsGraph, raw, DF, not_FC):
    """
    :param model:
    :param device:
    :param loader: 鏁版嵁鍔犺浇鍣?
    :param sideEffectsGraph: 鍓綔鐢ㄥ浘淇℃伅锛?
    :param raw: 鍘熷鏁版嵁
    :return: 鎵€鏈夌殑琚玬ask鐨勫師濮嬪€硷紝鎵€鏈夌殑琚玬ask鐨勯娴嬪€硷紝閮芥槸1缁?
    """
    # 澹版槑涓哄紶閲?
    total_preds = torch.Tensor()
    total_reals = torch.Tensor()
    model.eval()
    torch.cuda.manual_seed(42)
    # print('Make prediction for {} samples...'.format(len(loader.dataset)))
    # 瀵逛簬tensor鐨勮绠楁搷浣滐紝榛樿鏄杩涜璁＄畻鍥剧殑鏋勫缓鐨勶紝鍦ㄨ繖绉嶆儏鍐典笅锛屽彲浠ヤ娇鐢╳ith torch.no_grad():鏉ュ己鍒朵箣鍚庣殑鍐呭涓嶈繘琛岃绠楀浘鏋勫缓
    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for data in loader:
            # 鏌ユ壘琚玬ask鐨勬暟鎹?
            index = [x[0] for x in data.index]

            label = data.y
            raw_label = torch.FloatTensor(raw[index])
            index_pair = torch.where(raw_label != label)
            data = data.to(device)
            output, _, _ = model(data, sideEffectsGraph, DF, not_FC)
            if model.__class__.__name__ == "CAFNetDecoupled":
                output = model.last_freq_pred

            pred = output.cpu()[index_pair]
            real = raw_label[index_pair]

            # torch.cat()锛氬皢涓や釜tensor鎷兼帴锛屾寜缁存暟0鎷兼帴锛堝線涓嬫嫾锛夋垨鎸夌淮鏁?鎷兼帴锛堝線鍙虫嫾锛?
            total_preds = torch.cat((total_preds, pred), 0)
            total_reals = torch.cat((total_reals, real), 0)

    return total_reals.numpy().flatten(), total_preds.numpy().flatten()


def getAllResultMatrix(model, device, loader, sideEffectsGraph, raw, mask, result_folder, DF, not_FC,
                       calibration_mode="none", seed=42, fold_id=None, save_full_pred=False):
    """
    淇濆瓨棰勬祴缁撴灉
    """
    # 澹版槑涓哄紶閲?
    pred_result = pd.read_csv(result_folder + '/pred_result.csv', header=None, index_col=None).values
    freq_pred_result_path = result_folder + '/freq_pred_result.csv'
    if os.path.exists(freq_pred_result_path):
        freq_pred_result = pd.read_csv(freq_pred_result_path, header=None, index_col=None).values
    else:
        freq_pred_result = np.zeros(raw.shape)
    # pred_result = np.loadtxt(result_folder + '/pred_result.csv')
    # print(pred_result.shape)
    pred = torch.Tensor()
    freq_pred = torch.Tensor()
    model.eval()
    torch.cuda.manual_seed(42)
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    # 瀵逛簬tensor鐨勮绠楁搷浣滐紝榛樿鏄杩涜璁＄畻鍥剧殑鏋勫缓鐨勶紝鍦ㄨ繖绉嶆儏鍐典笅锛屽彲浠ヤ娇鐢╳ith torch.no_grad():鏉ュ己鍒朵箣鍚庣殑鍐呭涓嶈繘琛岃绠楀浘鏋勫缓
    # 椤哄簭鍔犺浇鏁版嵁
    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for data in loader:
            data = data.to(device)
            output, _, _ = model(data, sideEffectsGraph, DF, not_FC)
            # print(output.shape, type(output))
            # exit(1)
            pred = torch.cat((pred, output.cpu()), 0)
            freq_output = getattr(model, "last_freq_pred", None)
            if freq_output is None:
                freq_output = output
            freq_pred = torch.cat((freq_pred, freq_output.detach().cpu()), 0)
        # 淇濆瓨姝ゆ棰勬祴鐨勬墍鏈変綅缃?
        # np.set_printoptions(precision=4)
        pred = pred.numpy()
        freq_pred = freq_pred.numpy()
        if calibration_mode != "none":
            pred = calibrate_prediction_matrix(pred, raw, mask, calibration_mode, seed)
        if save_full_pred:
            full_dir = os.path.join(result_folder, "full_predictions")
            os.makedirs(full_dir, exist_ok=True)
            fold_tag = "unknown" if fold_id is None else str(fold_id - 1)
            pd.DataFrame(pred).to_csv(
                os.path.join(full_dir, f"full_pred_fold{fold_tag}.csv"),
                header=False,
                index=False,
                float_format="%.6f",
            )
            freq_full_dir = os.path.join(result_folder, "full_freq_predictions")
            os.makedirs(freq_full_dir, exist_ok=True)
            pd.DataFrame(freq_pred).to_csv(
                os.path.join(freq_full_dir, f"full_freq_pred_fold{fold_tag}.csv"),
                header=False,
                index=False,
                float_format="%.6f",
            )
        # print(pred.shape)
        mask = (mask == 0).astype(int)
        pred_result = pred_result + pred * mask
        freq_pred_result = freq_pred_result + freq_pred * mask
        # print(pred_result.shape)

        pred_result = pd.DataFrame(pred_result)
        pred_result.to_csv(result_folder + '/pred_result.csv', header=False, index=False, float_format='%.4f')
        freq_pred_result = pd.DataFrame(freq_pred_result)
        freq_pred_result.to_csv(freq_pred_result_path, header=False, index=False, float_format='%.4f')


def evaluate(model, device, loader, sideEffectsGraph, mask, raw, DF, not_FC, calibration_mode="none", seed=42):
    total_preds = torch.Tensor()
    total_assoc_preds = torch.Tensor()
    singleDrug_auc = []
    singleDrug_aupr = []
    model.eval()
    torch.cuda.manual_seed(42)

    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for data in loader:
            # 鏌ユ壘琚玬ask鐨勬暟鎹?
            index = [x[0] for x in data.index]
            train_label = data.y.numpy().flatten()
            data = data.to(device)
            output, _, _ = model(data, sideEffectsGraph, DF, not_FC)
            pred = output.cpu()
            assoc_pred = model.last_assoc_logits.detach().cpu() if model.__class__.__name__ == "CAFNetDecoupled" else pred
            # torch.cat()锛氬皢涓や釜tensor鎷兼帴锛屾寜缁存暟0鎷兼帴锛堝線涓嬫嫾锛夋垨鎸夌淮鏁?鎷兼帴锛堝線鍙虫嫾锛?
            total_preds = torch.cat((total_preds, pred), 0)
            total_assoc_preds = torch.cat((total_assoc_preds, assoc_pred), 0)

            pred = assoc_pred.numpy().flatten()
            train_label = (train_label != 0).astype(int)
            if sum(mask[index].flatten()) == len(mask[index].flatten()):
                continue
            posi = pred[np.where(mask[index].flatten() == 0)[0]]
            nege = pred[np.where((mask[index].flatten() - train_label))[0]]
            y = np.hstack((posi, nege))
            y_true = np.hstack((np.ones(len(posi)), np.zeros(len(nege))))
            singleDrug_auc.append(roc_auc_score(y_true, y))
            singleDrug_aupr.append(average_precision_score(y_true, y))

    drugAUC = sum(singleDrug_auc) / len(singleDrug_auc)
    drugAUPR = sum(singleDrug_aupr) / len(singleDrug_aupr)
    print('num of singleDrug_auc: ', len(singleDrug_auc))
    # print('drugAUPR: ', drugAUPR)
    total_preds = total_preds.numpy()
    total_assoc_preds = total_assoc_preds.numpy()
    if calibration_mode != "none":
        total_preds = calibrate_prediction_matrix(total_preds, raw, mask, calibration_mode, seed)
        total_assoc_preds = calibrate_prediction_matrix(total_assoc_preds, raw, mask, calibration_mode, seed)

    pos = total_assoc_preds[np.where(mask == 0)]
    pos_label = np.ones(len(pos))

    neg = total_assoc_preds[np.where(raw == 0)]
    neg_label = np.zeros(len(neg))

    y = np.hstack((pos, neg))
    y_true = np.hstack((pos_label, neg_label))
    auc_all = roc_auc_score(y_true, y)
    aupr_all = average_precision_score(y_true, y)
    # others
    Tr_neg = {}
    Te = {}
    train_data = raw * mask
    Te_pairs = np.where(mask == 0)
    Tr_neg_pairs = np.where(train_data == 0)
    Te_pairs = np.array(Te_pairs).transpose()
    Tr_neg_pairs = np.array(Tr_neg_pairs).transpose()
    for te_pair in Te_pairs:
        drug_id = te_pair[0]
        SE_id = te_pair[1]
        if drug_id not in Te:
            Te[drug_id] = [SE_id]
        else:
            Te[drug_id].append(SE_id)

    for te_pair in Tr_neg_pairs:
        drug_id = te_pair[0]
        SE_id = te_pair[1]
        if drug_id not in Tr_neg:
            Tr_neg[drug_id] = [SE_id]
        else:
            Tr_neg[drug_id].append(SE_id)

    positions = [1, 5, 10, 15]
    map_value, auc_value, ndcg, prec, rec = evaluate_others(total_preds, Tr_neg, Te, positions)

    p1, p5, p10, p15 = prec[0], prec[1], prec[2], prec[3]
    r1, r5, r10, r15 = rec[0], rec[1], rec[2], rec[3]
    return auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15
def evaluatee(model, device, loader, sideEffectsGraph, mask, raw, DF, not_FC, calibration_mode="none", seed=42):
    total_preds = torch.Tensor()
    model.eval()
    torch.cuda.manual_seed(42)

    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for data in loader:
            data = data.to(device)
            output, _, _ = model(data, sideEffectsGraph, DF, not_FC)
            pred = output.cpu()
            # torch.cat()锛氬皢涓や釜tensor鎷兼帴锛屾寜缁存暟0鎷兼帴锛堝線涓嬫嫾锛夋垨鎸夌淮鏁?鎷兼帴锛堝線鍙虫嫾锛?
            total_preds = torch.cat((total_preds, pred), 0)
    total_preds = total_preds.numpy()
    if calibration_mode != "none":
        total_preds = calibrate_prediction_matrix(total_preds, raw, mask, calibration_mode, seed)
    pos = total_preds[np.where(mask == 0)]
    pos_label = np.ones(len(pos))
    neg = total_preds[np.where(raw == 0)]
    neg_label = np.zeros(len(neg))

    y = np.hstack((pos, neg))
    y_true = np.hstack((pos_label, neg_label))
    auc_all = roc_auc_score(y_true, y)
    print('灏忛妫嶆潵鍜? ', auc_all)
    return auc_all

class EarlyStopping:
    """鏃╂湡鍋滄浠ラ槻姝㈣繃鎷熷悎"""

    def __init__(self, patience=20, min_delta=0.001, metric='max'):
        """
        Args:
            patience (int): 鍦ㄦ病鏈夋敼杩涚殑鎯呭喌涓嬬户缁缁冪殑epoch鏁?
            min_delta (float): 琚涓烘槸鏀硅繘鐨勬渶灏忓彉鍖栭噺
            metric (str): 'max'琛ㄧず鎸囨爣瓒婂ぇ瓒婂ソ锛?min'琛ㄧず鎸囨爣瓒婂皬瓒婂ソ
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric  # 'max' or 'min'
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif self.metric == 'max':
            if current_score <= self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.counter = 0
        else:  # metric == 'min'
            if current_score >= self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.counter = 0

        return self.early_stop

def main(modeling, metric, train_batch, lr, num_epoch, knn, weight_decay, lamb, log_interval, cuda_name, frequencyMat,
         id, mask, result_folder, save_model, DF, not_FC, output_dim, eps, pca,
         use_cross_attn, fusion_mode, gate_mode, fusion_alpha, loss_type, loss_gamma, loss_alpha, ablation_tag,
         rank_score_mix=0.7, dual_task=False, assoc_weight=1.0, freq_weight=0.2, rank_weight=0.1,
         bpr_samples=32, seed=42, side_feature_file=None, side_feature_concat=False, calibration_mode="none",
         ordinal_weight=0.0, ordinal_score_mix=0.2, pop_weight=0.0, bias_weight=1.0, list_weight=0.0,
         assoc_base_weight=1.0, assoc_residual_weight=1.0, save_full_pred=False,
         strict_train_mask=True, monitor_outer_test=False, listnet_target="binary"):
    print('\n=======================================================================================')
    print('\n绗?{} 娆¤缁冿細\n'.format(id))
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
    if modeling.__name__ in ["CAFNetV2", "CAFNetOrdinal", "CAFNetDecoupled"]:
        print('rank_score_mix: ', rank_score_mix)
        print('dual_task: ', dual_task)
        print('assoc_weight: ', assoc_weight)
        print('freq_weight: ', freq_weight)
        print('rank_weight: ', rank_weight)
    if modeling.__name__ == "CAFNetOrdinal":
        print('ordinal_weight: ', ordinal_weight)
        print('ordinal_score_mix: ', ordinal_score_mix)
    if modeling.__name__ == "CAFNetDecoupled":
        print('pop_weight: ', pop_weight)
        print('bias_weight: ', bias_weight)
        print('list_weight: ', list_weight)
        print('assoc_base_weight: ', assoc_base_weight)
        print('assoc_residual_weight: ', assoc_residual_weight)
    if side_feature_file:
        print('side_feature_file: ', side_feature_file)
        print('side_feature_concat: ', side_feature_concat)
    print('calibration_mode: ', calibration_mode)
    print('strict_train_mask: ', strict_train_mask)
    print('monitor_outer_test: ', monitor_outer_test)
    print('listnet_target: ', listnet_target)

    model_st = modeling.__name__
    dataset = 'drug_sideEffect'
    train_losses = []
    # test_MSE = []
    # test_pearsons = []
    # test_rMSE = []
    # test_spearman = []
    # test_MAE = []
    print('\nrunning on ', model_st + '_' + dataset)
    processed_raw = raw_file

    if not os.path.isfile(processed_raw):
        print('Missing FrequencyMat, exit!!!')
        exit(1)

    # 鐢熸垚鍓綔鐢ㄧ殑graph淇℃伅
    frequencyMat = frequencyMat.T
    if pca:
        pca_ = PCA(n_components=256)
        similarity_pca = pca_.fit_transform(frequencyMat)
        print('PCA 淇℃伅淇濈暀姣斾緥锛?')
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

    # load  side_effect_label mat 锛岀敤node_label鍋氱偣淇℃伅 994*243
    node_label = load_side_node_features(side_feature_file, side_feature_concat)
    feat = torch.tensor(node_label, dtype=torch.float)
    sideEffectsGraph = Data(x=feat, edge_index=edges)
    input_dim_e = int(feat.shape[1])

    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']
    train_data_mat = raw * mask
    pos = np.count_nonzero(train_data_mat)
    neg = train_data_mat.size - pos
    # 浣跨敤鍘熷鍥炲綊寮忔崯澶憋紙loss_fun锛夛紝涓嶅惎鐢?CAFNetLoss

    # make data_WS Pytorch mini-batch processing ready
    train_data = myDataset(root='data_WS', dataset='drug_sideEffect_data' + str(id - 1))
    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
    test_loader = DataLoader(train_data, batch_size=1, shuffle=False)

    print('CPU/GPU: ', torch.cuda.is_available())

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    if modeling.__name__ in ["CAFNet", "CAFNetV2", "CAFNetOrdinal", "CAFNetDecoupled"]:
        model = modeling(
            input_dim=input_dim,
            input_dim_e=input_dim_e,
            output_dim=output_dim,
            use_cross_attn=use_cross_attn,
            fusion_mode=fusion_mode,
            gate_mode=gate_mode,
            fusion_alpha=fusion_alpha,
            **({"rank_score_mix": rank_score_mix} if modeling.__name__ in ["CAFNetV2", "CAFNetOrdinal"] else {}),
            **({"ordinal_score_mix": ordinal_score_mix} if modeling.__name__ == "CAFNetOrdinal" else {}),
            **({"rank_score_mix": rank_score_mix, "pop_weight": pop_weight, "bias_weight": bias_weight,
                "assoc_base_weight": assoc_base_weight, "assoc_residual_weight": assoc_residual_weight}
               if modeling.__name__ == "CAFNetDecoupled" else {}),
        ).to(device)
        if modeling.__name__ == "CAFNetDecoupled":
            train_observed = train_data_mat != 0
            side_popularity = train_observed.sum(axis=0) / max(1.0, float(train_data_mat.shape[0]))
            global_mean = train_data_mat[train_observed].mean() if np.any(train_observed) else 0.0
            model.set_frequency_priors(side_popularity=side_popularity, global_mean=global_mean)
    else:
        model = modeling(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model_file_name = str(id) + 'MF_' + model_st + '_epoch=' + str(num_epoch) + '.model'
    result_log = result_folder + '/' + model_st + '_result.csv'
    loss_fig_name = str(id) + model_st + '_loss'
    pearson_fig_name = str(id) + model_st + '_pearson'
    MSE_fig_name = str(id) + model_st + '_MSE'
    rMSE_fig_name = str(id) + model_st + '_rMSE'
    auc_all=[]
    metrics_log = os.path.join(result_folder, f"{model_st}_metrics_{ablation_tag}_{loss_type}.csv")
    if not os.path.exists(metrics_log):
        with open(metrics_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'auc_all', 'aupr_all', 'drugAUC', 'drugAUPR'])

    # 鍦ㄥ嚱鏁板紑濮嬪垵濮嬪寲鏃╁仠瀵硅薄
    early_stopping = None

    for epoch in range(num_epoch):
        train_loss = train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, lamb=lamb,
                           epoch=epoch + 1, log_interval=log_interval, sideEffectsGraph=sideEffectsGraph, raw=raw,
                           id=id, DF=DF, not_FC=not_FC, eps=eps, dual_task=dual_task,
                           assoc_weight=assoc_weight, freq_weight=freq_weight, rank_weight=rank_weight,
                           bpr_samples=bpr_samples, ordinal_weight=ordinal_weight, list_weight=list_weight,
                           train_entry_mask=(mask != 0) if strict_train_mask else None,
                           listnet_target=listnet_target)

        train_losses.append(train_loss)

        # 姣忛殧涓€瀹歟poch璇勪及涓€娆★紝閬垮厤棰戠箒璇勪及褰卞搷璁粌鏁堢巼
        if monitor_outer_test and (epoch + 1) % 10 == 0:
            # 鑾峰彇楠岃瘉闆嗕笂鐨勬€ц兘鎸囨爣
            auc_all, aupr_all, drugAUC, drugAUPR, _, _, _, _, _, _, _, _, _, _ = evaluate(
                model=model, device=device, loader=test_loader,
                sideEffectsGraph=sideEffectsGraph, mask=mask,
                raw=raw, DF=DF, not_FC=not_FC, calibration_mode=calibration_mode, seed=seed + id - 1)

            # 閫夋嫨鍚堥€傜殑鎸囨爣浣滀负鏃╁仠渚濇嵁锛岃繖閲屼娇鐢╝uc_all
            current_metric = auc_all
            print(f'Epoch {epoch + 1}: Validation AUC = {current_metric:.5f}')

            with open(metrics_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, train_loss, auc_all, aupr_all, drugAUC, drugAUPR])

            # 妫€鏌ユ槸鍚﹀簲璇ユ棭鍋?
            # early stopping disabled

    if save_model:
        checkpointsFolder = result_folder + '/checkpoints/'
        isCheckpointExist = os.path.exists(checkpointsFolder)
        if not isCheckpointExist:
            os.makedirs(checkpointsFolder)
        torch.save(model.state_dict(), checkpointsFolder + model_file_name)

    test_labels, test_preds = predict(model=model, device=device, loader=test_loader,
                                      sideEffectsGraph=sideEffectsGraph, raw=raw, DF=DF, not_FC=not_FC)

    ret_test = [mse(test_labels, test_preds), pearson(test_labels, test_preds), rmse(test_labels, test_preds),
                spearman(test_labels, test_preds), MAE(test_labels, test_preds)]
    test_pearsons, test_rMSE, test_spearman, test_MAE = ret_test[1], ret_test[2], ret_test[3], ret_test[4]
    auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15 = evaluate(model=model,
                                                                                                          device=device,
                                                                                                          loader=test_loader,
                                                                                                          sideEffectsGraph=sideEffectsGraph,
                                                                                                          mask=mask,
                                                                                                          raw=raw, DF=DF,
                                                                                                          not_FC=not_FC,
                                                                                                          calibration_mode=calibration_mode,
                                                                                                          seed=seed + id - 1)

    # 鍐欏叆棰勬祴鏁堟灉
    result = [test_pearsons, test_rMSE, test_spearman, test_MAE, auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg,
              p1, p5, p10, p15, r1, r5, r10, r15]
    with open(result_log, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)
    # 鍐欏叆棰勬祴鍊?
    getAllResultMatrix(model=model, device=device, loader=test_loader, sideEffectsGraph=sideEffectsGraph, raw=raw,
                       mask=mask, result_folder=result_folder, DF=DF, not_FC=not_FC,
                       calibration_mode=calibration_mode, seed=seed + id - 1,
                       fold_id=id, save_full_pred=save_full_pred)

    print('Test:\nPearson: {:.5f}\trMSE: {:.5f}\tSpearman: {:.5f}\tMAE: {:.5f}'.format(result[0], result[1], result[2],
                                                                                       result[3]))
    print('\tall AUC: {:.5f}\tall AUPR: {:.5f}\tdrug AUC: {:.5f}\tdrug AUPR: {:.5f}'.format(result[4], result[5],
                                                                                            result[6], result[7]))
    print('\tMAP: {:.5f}\tnDCG@10: {:.5f}'.format(map_value, ndcg))
    print('\tP@1: {:.5f}\tP@5: {:.5f}\tP@10: {:.5f}\tP@15: {:.5f}'.format(p1, p5, p10, p15))
    print('\tR@1: {:.5f}\tR@5: {:.5f}\tR@10: {:.5f}\tR@15: {:.5f}'.format(r1, r5, r10, r15))
    # train loss
    # Disabled for batch experiments: on the Windows A3 environment matplotlib
    # can terminate the interpreter after savefig, even though the PNG is written.
    # Fold metrics and prediction matrices are the required experiment artifacts.
    print('Training-loss figure skipped for fold {}.'.format(id - 1))
    # test pearson
    # draw_pearson(test_pearsons, pearson_fig_name, result_folder)
    # # test mse
    # my_draw_mse(test_MSE, test_rMSE, MSE_fig_name, result_folder)


if __name__ == '__main__':

    # ??????
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0,
                        help='0:CAFNet, 1:A3Net, 2:CAFNetV2, 3:CAFNetOrdinal, 4:CAFNetDecoupled')
    parser.add_argument('--metric', type=int, required=False, default=0, help='0: cosine, 1: jaccard, 2: euclidean')
    parser.add_argument('--train_batch', type=int, required=False, default=10, help='Batch size training set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, required=False, default=0.001, help='weight_decay')
    parser.add_argument('--lamb', type=float, required=False, default=0.03, help='LAMBDA')
    parser.add_argument('--epoch', type=int, required=False, default=100, help='Number of epochs')
    parser.add_argument('--knn', type=int, required=False, default=10, help='Number of KNN')
    parser.add_argument('--log_interval', type=int, required=False, default=40, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default='cuda:0', help='Cuda')
    parser.add_argument('--dim', type=int, required=False, default=200, help='features dimensions of drugs and side effects')
    parser.add_argument('--eps', type=float, required=False, default=0.5, help='regard 0 as eps when training')

    parser.add_argument('--tenfold', action='store_true', default=False, help='use 10 folds Cross-validation ')
    parser.add_argument('--save_model', action='store_true', default=False, help='save model and features')
    parser.add_argument('--DF', action='store_true', default=False, help='use DF decoder')
    parser.add_argument('--not_FC', action='store_true', default=False, help='not use Linear layers')
    parser.add_argument('--PCA', action='store_true', default=False, help='use PCA')
    parser.add_argument('--ablation_all', action='store_true', default=False, help='run all ablations')
    parser.add_argument('--no_cross_attn', action='store_true', default=False, help='disable cross attention')
    parser.add_argument('--fusion_mode', type=str, required=False, default='gate',
                        choices=['gate', 'fixed', 'none'], help='fusion mode for CAFNet')
    parser.add_argument('--gate_mode', type=str, required=False, default='new',
                        choices=['new', 'old'], help='gate implementation for CAFNet')
    parser.add_argument('--fusion_alpha', type=float, required=False, default=0.5,
                        help='alpha for fixed fusion')
    parser.add_argument('--loss_type', type=str, required=False, default='focal',
                        choices=['bce', 'focal'], help='loss type')
    parser.add_argument('--loss_gamma', type=float, required=False, default=2.0,
                        help='focal loss gamma')
    parser.add_argument('--loss_alpha', type=float, required=False, default=0.25,
                        help='focal loss alpha')
    parser.add_argument('--result_prefix', type=str, required=False, default='WS')
    parser.add_argument('--short_result_name', action='store_true', default=False,
                        help='use result_prefix as the complete result directory name')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='allow deletion and replacement of an existing result directory')
    parser.add_argument('--max_folds', type=int, required=False, default=10)
    parser.add_argument('--start_fold', type=int, required=False, default=0,
                        help='zero-based fold index for resuming an existing result directory')
    parser.add_argument('--rank_score_mix', type=float, required=False, default=0.7)
    parser.add_argument('--dual_task', action='store_true', default=False)
    parser.add_argument('--assoc_weight', type=float, required=False, default=1.0)
    parser.add_argument('--freq_weight', type=float, required=False, default=0.2)
    parser.add_argument('--rank_weight', type=float, required=False, default=0.1)
    parser.add_argument('--bpr_samples', type=int, required=False, default=32)
    parser.add_argument('--seed', type=int, required=False, default=42)
    parser.add_argument('--side_feature_file', type=str, required=False, default=None)
    parser.add_argument('--side_feature_concat', action='store_true', default=False)
    parser.add_argument('--calibration_mode', type=str, required=False, default='none',
                        choices=['none', 'regression_prior', 'rank_pop', 'hybrid'])
    parser.add_argument('--ordinal_weight', type=float, required=False, default=0.0)
    parser.add_argument('--ordinal_score_mix', type=float, required=False, default=0.2)
    parser.add_argument('--pop_weight', type=float, required=False, default=0.0)
    parser.add_argument('--bias_weight', type=float, required=False, default=1.0)
    parser.add_argument('--list_weight', type=float, required=False, default=0.0)
    parser.add_argument('--assoc_base_weight', type=float, required=False, default=1.0)
    parser.add_argument('--assoc_residual_weight', type=float, required=False, default=1.0)
    parser.add_argument('--save_full_pred', action='store_true', default=False,
                        help='save fold-specific full 750x994 prediction matrices under result/full_predictions')
    parser.add_argument('--legacy_missing_as_zero', action='store_true', default=False,
                        help='include outer-test coordinates as zero targets (legacy protocol; not for submission)')
    parser.add_argument('--monitor_outer_test', action='store_true', default=False,
                        help='log outer-test metrics during training (diagnostic only; not for submission)')
    parser.add_argument('--listnet_target', choices=['binary', 'graded'], default='binary',
                        help='ListNet relevance: binary is the submission protocol; graded reproduces legacy warm runs')

    # ???????rgs???
    args = parser.parse_args()

    modeling = [CAFNet, A3_Net, CAFNetV2, CAFNetOrdinal, CAFNetDecoupled][args.model]
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
    use_cross_attn = not args.no_cross_attn
    fusion_mode = args.fusion_mode
    gate_mode = args.gate_mode
    fusion_alpha = args.fusion_alpha
    loss_type = args.loss_type
    loss_gamma = args.loss_gamma
    loss_alpha = args.loss_alpha
    result_prefix = args.result_prefix
    short_result_name = args.short_result_name
    max_folds = args.max_folds
    start_fold = args.start_fold
    if start_fold < 0 or start_fold >= min(int(max_folds), 10):
        raise ValueError('start_fold must satisfy 0 <= start_fold < min(max_folds, 10)')
    rank_score_mix = args.rank_score_mix
    dual_task = args.dual_task or modeling.__name__ in ["CAFNetV2", "CAFNetOrdinal", "CAFNetDecoupled"]
    assoc_weight = args.assoc_weight
    freq_weight = args.freq_weight
    rank_weight = args.rank_weight
    bpr_samples = args.bpr_samples
    seed = args.seed
    side_feature_file = args.side_feature_file
    side_feature_concat = args.side_feature_concat
    calibration_mode = args.calibration_mode
    ordinal_weight = args.ordinal_weight
    ordinal_score_mix = args.ordinal_score_mix
    pop_weight = args.pop_weight
    bias_weight = args.bias_weight
    list_weight = args.list_weight
    assoc_base_weight = args.assoc_base_weight
    assoc_residual_weight = args.assoc_residual_weight
    save_full_pred = args.save_full_pred
    strict_train_mask = not args.legacy_missing_as_zero
    monitor_outer_test = args.monitor_outer_test
    listnet_target = args.listnet_target

    # ???????????
    dataset = 'drug_sideEffect'

    processed_mask_mat = mask_mat_file
    if not os.path.isfile(processed_mask_mat):
        raise FileNotFoundError('Required fixed warm-start mask file is missing: {}'.format(processed_mask_mat))

    if args.ablation_all:
        ablation_runs = [
            {'name': 'full', 'use_cross_attn': True, 'fusion_mode': 'gate', 'gate_mode': 'new', 'fusion_alpha': fusion_alpha},
            {'name': 'noCA', 'use_cross_attn': False, 'fusion_mode': 'gate', 'gate_mode': 'new', 'fusion_alpha': fusion_alpha},
            {'name': 'noGate', 'use_cross_attn': True, 'fusion_mode': 'none', 'gate_mode': 'new', 'fusion_alpha': fusion_alpha},
            {'name': 'fixedFusion', 'use_cross_attn': True, 'fusion_mode': 'fixed', 'gate_mode': 'new', 'fusion_alpha': fusion_alpha},
            {'name': 'oldGate', 'use_cross_attn': True, 'fusion_mode': 'gate', 'gate_mode': 'old', 'fusion_alpha': fusion_alpha},
        ]
    else:
        ablation_runs = [
            {'name': 'custom', 'use_cross_attn': use_cross_attn, 'fusion_mode': fusion_mode,
             'gate_mode': gate_mode, 'fusion_alpha': fusion_alpha},
        ]

    for run in ablation_runs:
        ablation_tag = build_ablation_tag(run['use_cross_attn'], run['fusion_mode'], run['gate_mode'])

        ######################################################################################
        if short_result_name:
            result_folder = os.path.join('./result_WS', result_prefix)
        else:
            result_folder = './result_WS/'
            if tenfold:
                result_folder += '10' + result_prefix + '_' + modeling.__name__ + '_knn=' + str(knn) + '_wd=' + str(
                    weight_decay) + '_epoch=' + str(num_epoch) + '_lamb=' + str(lamb) + '_lr' + str(lr) + '_dim=' + str(
                    output_dim) + '_eps=' + str(eps) + '_DF=' + str(DF) + '_PCA=' + str(pca) + '_not-FC=' + str(not_FC) + '_' + str(metric)
            else:
                result_folder += '1' + result_prefix + '_' + modeling.__name__ + '_knn=' + str(knn) + '_wd=' + str(
                    weight_decay) + '_epoch=' + str(num_epoch) + '_lamb=' + str(lamb) + '_lr' + str(lr) + '_dim=' + str(
                    output_dim) + '_eps=' + str(eps) + '_DF=' + str(DF) + '_PCA=' + str(pca) + '_not-FC=' + str(not_FC) + '_' + str(metric)
            result_folder += '_abl=' + ablation_tag + '_loss=' + loss_type
            if modeling.__name__ in ["CAFNetV2", "CAFNetOrdinal", "CAFNetDecoupled"]:
                result_folder += '_mix=' + str(rank_score_mix) + '_aw=' + str(assoc_weight) + '_fw=' + str(freq_weight) + '_rw=' + str(rank_weight)
            if modeling.__name__ == "CAFNetOrdinal":
                result_folder += '_ow=' + str(ordinal_weight) + '_omix=' + str(ordinal_score_mix)
            if modeling.__name__ == "CAFNetDecoupled":
                result_folder += '_popw=' + str(pop_weight) + '_biasw=' + str(bias_weight) + '_listw=' + str(list_weight)
                result_folder += '_abw=' + str(assoc_base_weight) + '_arw=' + str(assoc_residual_weight)
            if side_feature_file:
                result_folder += '_sidefeat=' + ('concat' if side_feature_concat else 'replace')
            if calibration_mode != 'none':
                result_folder += '_cal=' + calibration_mode
            result_folder += '_strictmask=' + str(strict_train_mask)
            if modeling.__name__ in ["CAFNetV2", "CAFNetOrdinal", "CAFNetDecoupled"]:
                result_folder += '_listtarget=' + listnet_target

        isExist = os.path.exists(result_folder)
        if not isExist:
            os.makedirs(result_folder)
        elif start_fold > 0:
            print('Resuming existing result directory at fold {}: {}'.format(start_fold, result_folder))
        elif args.overwrite:
            shutil.rmtree(result_folder)
            os.makedirs(result_folder)
        else:
            raise FileExistsError(
                'Refusing to overwrite existing result directory: {}. Use --overwrite explicitly.'.format(result_folder)
            )
        ######################################################################################

        result_log = result_folder + '/' + modeling.__name__ + '_result.csv'
        raw_frequency = scipy.io.loadmat(raw_file)
        raw = raw_frequency['R']

        if start_fold == 0:
            with open(result_log, 'w', newline='') as f:
                fieldnames = ['pearson', 'rMSE', 'spearman', 'MAE', 'auc_all', 'aupr_all', 'drugAUC', 'drugAUPR', 'MAP', 'nDCG',
                              'P1', 'P5', 'P10', 'P15', 'R1', 'R5', 'R10', 'R15']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

            pred_result = np.zeros(raw.shape)
            pred_result = pd.DataFrame(pred_result)
            pred_result.to_csv(result_folder + '/pred_result.csv', header=False, index=False)
            freq_pred_result = np.zeros(raw.shape)
            freq_pred_result = pd.DataFrame(freq_pred_result)
            freq_pred_result.to_csv(result_folder + '/freq_pred_result.csv', header=False, index=False)
        elif not os.path.exists(result_log):
            raise FileNotFoundError('Cannot resume because the fold metrics file is missing: {}'.format(result_log))

        start = datetime.datetime.now()
        for (id, frequencyMat, mask) in split_data(tenfold, max_folds=max_folds, start_fold=start_fold):
            start_ = datetime.datetime.now()
            main(modeling, metric, train_batch, lr, num_epoch, knn, weight_decay, lamb, log_interval, cuda_name,
                 frequencyMat, id + 1, mask, result_folder, save_model, DF, not_FC, output_dim, eps, pca,
                 run['use_cross_attn'], run['fusion_mode'], run['gate_mode'], run['fusion_alpha'],
                 loss_type, loss_gamma, loss_alpha, ablation_tag,
                 rank_score_mix, dual_task, assoc_weight, freq_weight, rank_weight, bpr_samples, seed,
                 side_feature_file, side_feature_concat, calibration_mode, ordinal_weight, ordinal_score_mix,
                  pop_weight, bias_weight, list_weight, assoc_base_weight, assoc_residual_weight,
                  save_full_pred, strict_train_mask, monitor_outer_test, listnet_target)

            end_ = datetime.datetime.now()
            print('鏈杩愯鏃堕棿锛歿}\t'.format(end_ - start_))
        end = datetime.datetime.now()

    data = pd.read_csv(result_log)
    metric_cols = ['pearson', 'rMSE', 'spearman', 'MAE', 'auc_all', 'aupr_all', 'drugAUC', 'drugAUPR', 'MAP', 'nDCG',
                   'P1', 'P5', 'P10', 'P15', 'R1', 'R5', 'R10', 'R15']

    # Exclude non-numeric marker rows (e.g., manually appended ["avg", ...]) from fold averaging.
    fold_rows = data.copy()
    for col in metric_cols:
        fold_rows[col] = pd.to_numeric(fold_rows[col], errors='coerce')
    fold_rows = fold_rows.dropna(subset=['pearson'])
    L = len(fold_rows)
    if L == 0:
        raise RuntimeError('No numeric metric rows found in {} for averaging.'.format(result_log))
    avg = fold_rows[metric_cols].mean(axis=0).tolist()
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
        print('杩愯鏃堕棿锛歿}\t'.format(end - start))
