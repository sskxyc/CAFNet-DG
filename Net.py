import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GCNConv, GINConv, RGCNConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch.nn import Parameter as Param
import numpy as np
from torch.nn.utils.weight_norm import weight_norm
import math
# GCN  model
class GCN(torch.nn.Module):
    def __init__(self, input_dim=78, input_dim_e=243, output_dim=64, output_dim_e=64):
        super(GCN, self).__init__()

        # graph layers : drug
        self.gcn1 = GCNConv(input_dim, 64)
        self.gcn2 = GCNConv(64, output_dim)
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        # # graph layers : sideEffect
        self.gcn3 = GCNConv(input_dim_e, 128)
        self.gcn4 = GCNConv(128, output_dim)
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)

        # activation and regularization
        self.relu = nn.ReLU()
        self.diag = DiagLayer(in_dim=output_dim)

    def forward(self, data, data_e, DF=False, not_FC=True):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_e, edge_index_e = data_e.x, data_e.edge_index

        # 药物
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = global_max_pool(x, batch)  # global max pooling

        # 副作用
        x_e = F.dropout(x_e, p=0.2, training=self.training)
        x_e = self.gcn3(x_e, edge_index_e)
        x_e = self.relu(x_e)
        x_e = F.dropout(x_e, p=0.2, training=self.training)
        x_e = self.gcn4(x_e, edge_index_e)
        x_e = self.relu(x_e)

        if not not_FC:
            x = self.relu(self.fc_g1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc_g2(x)
            x_e = self.relu(self.fc_g3(x_e))
            x_e = F.dropout(x_e, p=0.5, training=self.training)
            x_e = self.fc_g4(x_e)

        # 结合
        x_ = self.diag(x) if DF else x

        xc = torch.matmul(x_, x_e.T)

        return xc, x, x_e


# GAT  model
class GAT(torch.nn.Module):
    def __init__(self, input_dim=109, input_dim_e=243, output_dim=200, output_dim_e=64, dropout=0.2, heads=10):
        super(GAT, self).__init__()

        # graph layers : drug
        self.gcn1 = GATConv(input_dim, 128, heads=heads, dropout=dropout)
        self.gcn2 = GATConv(128 * heads, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        # # graph layers : sideEffect
        self.gcn3 = GATConv(input_dim_e, 128, heads=heads, dropout=dropout)
        self.gcn4 = GATConv(128 * heads, output_dim, dropout=dropout)
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)

        # activation and regularization
        self.relu = nn.ReLU()
        self.diag = DiagLayer(in_dim=output_dim)

    def forward(self, data, data_e, DF=False, not_FC=True):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_e, edge_index_e = data_e.x, data_e.edge_index
        # 药物
        x = self.relu(self.gcn1(x, edge_index))
        x = self.relu(self.gcn2(x, edge_index))
        x = global_max_pool(x, batch)  # global max pooling

        # 副作用
        x_e = self.gcn3(x_e, edge_index_e)
        x_e = self.relu(x_e)
        x_e = self.gcn4(x_e, edge_index_e)
        x_e = self.relu(x_e)

        if not not_FC:
            x = self.relu(self.fc_g1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc_g2(x)
            x_e = self.relu(self.fc_g3(x_e))
            x_e = F.dropout(x_e, p=0.5, training=self.training)
            x_e = self.fc_g4(x_e)

        # 结合
        x_ = self.diag(x) if DF else x

        xc = torch.matmul(x_, x_e.T)

        return xc, x, x_e


# GINConv model
class GIN(torch.nn.Module):

    def __init__(self, input_dim=78, input_dim_e=243, output_dim=64, dropout=0.2):
        super(GIN, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # convolution layers
        nn1 = Sequential(Linear(input_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(output_dim)

        nn2 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(output_dim)

        nn3 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(output_dim)

        nn4 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(output_dim)

        nn5 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(output_dim)

        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        nn6 = Sequential(Linear(input_dim_e, output_dim), ReLU(), Linear(output_dim, output_dim))  # 时序容器。

        self.conv6 = GINConv(nn6)
        self.bn6 = torch.nn.BatchNorm1d(output_dim)

        nn7 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv7 = GINConv(nn7)
        self.bn7 = torch.nn.BatchNorm1d(output_dim)

        nn8 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv8 = GINConv(nn8)
        self.bn8 = torch.nn.BatchNorm1d(output_dim)

        nn9 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv9 = GINConv(nn9)
        self.bn9 = torch.nn.BatchNorm1d(output_dim)

        nn10 = Sequential(Linear(output_dim, output_dim), ReLU(), Linear(output_dim, output_dim))
        self.conv10 = GINConv(nn10)
        self.bn10 = torch.nn.BatchNorm1d(output_dim)

        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)

        self.diag = DiagLayer(in_dim=output_dim)
        # activation and regularization
        self.relu = nn.ReLU()

    def forward(self, data, data_e, DF=False, not_FC=True):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_e, edge_index_e = data_e.x, data_e.edge_index

        # drug
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)

        # side effect
        x_e = F.relu(self.conv6(x_e, edge_index_e))
        x_e = self.bn6(x_e)
        x_e = F.relu(self.conv7(x_e, edge_index_e))
        x_e = self.bn7(x_e)
        x_e = F.relu(self.conv8(x_e, edge_index_e))
        x_e = self.bn8(x_e)
        x_e = F.relu(self.conv9(x_e, edge_index_e))
        x_e = self.bn9(x_e)
        x_e = F.relu(self.conv10(x_e, edge_index_e))
        x_e = self.bn10(x_e)

        if not not_FC:
            x = self.relu(self.fc_g1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc_g2(x)
            x_e = self.relu(self.fc_g3(x_e))
            x_e = F.dropout(x_e, p=0.5, training=self.training)
            x_e = self.fc_g4(x_e)

        # 结合
        x_ = self.diag(x) if DF else x
        xc = torch.matmul(x_, x_e.T)

        return xc, x, x_e



# RGCN  model
class RGCN(torch.nn.Module):
    def __init__(self, input_dim=109, input_dim_e=243, output_dim=64, output_dim_e=64, dropout=0.2, heads=10):
        super(RGCN, self).__init__()

        # graph layers : drug
        self.gcn1 = RGCNConv(input_dim, 64, num_relations=5, num_bases=4, aggr='mean')
        self.gcn2 = RGCNConv(64, output_dim, num_relations=5, num_bases=4, aggr='mean')
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        # # graph layers : sideEffect
        self.gcn3 = GATConv(input_dim_e, 128, heads=heads, dropout=dropout)
        self.gcn4 = GATConv(128 * heads, output_dim, dropout=dropout)
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)

        # activation and regularization
        self.relu = nn.ReLU()
        self.diag = DiagLayer(in_dim=output_dim)

    def forward(self, data, data_e, DF=False, not_FC=True):
        # graph input feed-forward
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        x_e, edge_index_e = data_e.x, data_e.edge_index
        # print(x.shape)
        # 药物
        x = F.dropout(x, p=0.2, training=self.training)  # 将模型整体的training状态参数传入dropout函数
        x = torch.tanh(self.gcn1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index, edge_type)
        x = torch.tanh(x)
        x = global_max_pool(x, batch)  # global max pooling

        # 副作用
        x_e = self.gcn3(x_e, edge_index_e)
        x_e = self.relu(x_e)
        x_e = self.gcn4(x_e, edge_index_e)
        x_e = self.relu(x_e)

        if not not_FC:
            x = self.relu(self.fc_g1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc_g2(x)
            x_e = self.relu(self.fc_g3(x_e))
            x_e = F.dropout(x_e, p=0.5, training=self.training)
            x_e = self.fc_g4(x_e)

        # 结合
        x_ = self.diag(x) if DF else x

        xc = torch.matmul(x_, x_e.T)

        return xc, x, x_e


class GAT1(torch.nn.Module):
    def __init__(self, input_dim=109, input_dim_e=243, output_dim=200, output_dim_e=64, dropout=0.2, heads=10):
        super(GAT1, self).__init__()

        # graph layers : drug
        self.gcn1 = GATConv(input_dim, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        # # graph layers : sideEffect
        self.gcn3 = GATConv(input_dim_e, output_dim, dropout=dropout)
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)

        # activation and regularization
        self.relu = nn.ReLU()
        self.diag = DiagLayer(in_dim=output_dim)

    def forward(self, data, data_e, DF=False, not_FC=True):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_e, edge_index_e = data_e.x, data_e.edge_index
        # 药物
        x = self.relu(self.gcn1(x, edge_index))
        x = global_max_pool(x, batch)  # global max pooling

        # 副作用
        x_e = self.gcn3(x_e, edge_index_e)
        x_e = self.relu(x_e)

        if not not_FC:
            x = self.relu(self.fc_g1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc_g2(x)
            x_e = self.relu(self.fc_g3(x_e))
            x_e = F.dropout(x_e, p=0.5, training=self.training)
            x_e = self.fc_g4(x_e)

        # 结合
        x_ = self.diag(x) if DF else x
        xc = torch.matmul(x_, x_e.T)

        return xc, x, x_e

class DiagLayer(torch.nn.Module):
    def __init__(self, in_dim, num_et=1):
        super(DiagLayer, self).__init__()
        self.num_et = num_et
        self.in_dim = in_dim
        self.weight = Param(torch.Tensor(num_et, in_dim))

        self.reset_parameters()

    def forward(self, x):
        # print(self.weight)
        value = x * self.weight
        return value

    def reset_parameters(self):
        self.weight.data.normal_(std=1/np.sqrt(self.in_dim))
        # self.weight.data.fill_(1)

"""
difbanben所记录的
"""
class CrossAttention(nn.Module):
    """双向交叉注意力模块 - 重构版"""

    def __init__(self, embed_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 药物到副作用的交叉注意力
        self.drug_query = nn.Linear(embed_dim, embed_dim)
        self.side_effect_key = nn.Linear(embed_dim, embed_dim)
        self.side_effect_value = nn.Linear(embed_dim, embed_dim)
        self.out_drug = nn.Linear(embed_dim, embed_dim)

        # 副作用到药物的交叉注意力
        self.side_effect_query = nn.Linear(embed_dim, embed_dim)
        self.drug_key = nn.Linear(embed_dim, embed_dim)
        self.drug_value = nn.Linear(embed_dim, embed_dim)
        self.out_side_effect = nn.Linear(embed_dim, embed_dim)

        self.scale = (embed_dim // num_heads) ** -0.5

    def forward(self, drug_features, side_effect_features):
        batch_drug, embed_dim_drug = drug_features.size()
        batch_side_effect, embed_dim_side_effect = side_effect_features.size()

        # 药物到副作用的交叉注意力
        Q_drug = self.drug_query(drug_features).unsqueeze(1)  # (batch_drug, 1, embed_dim)
        K_se = self.side_effect_key(side_effect_features).unsqueeze(0)  # (1, batch_side_effect, embed_dim)
        V_se = self.side_effect_value(side_effect_features).unsqueeze(0)  # (1, batch_side_effect, embed_dim)

        # 计算注意力分数
        attn_scores = torch.matmul(Q_drug, K_se.transpose(-2, -1)) * self.scale  # (batch_drug, 1, batch_side_effect)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 应用注意力到V
        cross_attn_drug = torch.matmul(attn_weights, V_se).squeeze(1)  # (batch_drug, embed_dim)
        cross_attn_drug = self.out_drug(cross_attn_drug)

        # 副作用到药物的交叉注意力
        Q_se = self.side_effect_query(side_effect_features).unsqueeze(1)  # (batch_side_effect, 1, embed_dim)
        K_drug = self.drug_key(drug_features).unsqueeze(0)  # (1, batch_drug, embed_dim)
        V_drug = self.drug_value(drug_features).unsqueeze(0)  # (1, batch_drug, embed_dim)

        # 计算注意力分数
        attn_scores_se = torch.matmul(Q_se, K_drug.transpose(-2, -1)) * self.scale  # (batch_side_effect, 1, batch_drug)
        attn_weights_se = torch.softmax(attn_scores_se, dim=-1)

        # 应用注意力到V
        cross_attn_se = torch.matmul(attn_weights_se, V_drug).squeeze(1)  # (batch_side_effect, embed_dim)
        cross_attn_se = self.out_side_effect(cross_attn_se)

        return cross_attn_drug, cross_attn_se

class GateFusionUnit(nn.Module):
    """Gate-based adaptive fusion without truncation."""
    def __init__(self, feature_dim):
        super(GateFusionUnit, self).__init__()
        self.feature_dim = feature_dim
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, feature1, feature2):
        # Assumes feature1 and feature2 are aligned on batch dimension.
        combined = torch.cat([feature1, feature2], dim=-1)
        gate = self.gate(combined)
        gated = gate * feature1 + (1.0 - gate) * feature2
        fused = self.proj(combined) + 0.5 * gated
        return fused

class GateFusionUnitOld(nn.Module):
    """Legacy gate fusion with softmax weights."""
    def __init__(self, feature_dim):
        super(GateFusionUnitOld, self).__init__()
        self.feature_dim = feature_dim
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, feature1, feature2):
        combined = torch.cat([feature1, feature2], dim=-1)
        gate_weights = self.gate(combined)
        fused_features = gate_weights[:, 0:1] * feature1 + gate_weights[:, 1:2] * feature2
        return fused_features

# class CrossAttention(nn.Module):
#     """增强的双向交叉注意力模块"""
#
#     def __init__(self, embed_dim, num_heads=8, dropout=0.1):
#         super(CrossAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
#
#         self.scale = self.head_dim ** -0.5
#         self.dropout = dropout
#
#         # 药物到副作用的交叉注意力
#         self.drug_qkv = nn.Linear(embed_dim, embed_dim * 3)
#         self.side_effect_qkv = nn.Linear(embed_dim, embed_dim * 3)
#         self.out_proj_drug = nn.Linear(embed_dim, embed_dim)
#         self.out_proj_side_effect = nn.Linear(embed_dim, embed_dim)
#
#         self.dropout_layer = nn.Dropout(dropout)
#
#     def forward(self, drug_features, side_effect_features):
#         batch_drug, embed_dim_drug = drug_features.size()
#         batch_side_effect, embed_dim_side_effect = side_effect_features.size()
#
#         # 药物特征的QKV
#         qkv_drug = self.drug_qkv(drug_features).reshape(-1, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
#         q_drug, k_drug, v_drug = qkv_drug[0], qkv_drug[1], qkv_drug[2]
#
#         # 副作用特征的QKV
#         qkv_se = self.side_effect_qkv(side_effect_features).reshape(-1, 3, self.num_heads, self.head_dim).permute(1, 0,
#                                                                                                                   2, 3)
#         q_se, k_se, v_se = qkv_se[0], qkv_se[1], qkv_se[2]
#
#         # 药物查询 + 副作用键值
#         attn_scores_ds = torch.matmul(q_drug.unsqueeze(2), k_se.unsqueeze(1).transpose(-2, -1)) * self.scale
#         attn_weights_ds = torch.softmax(attn_scores_ds, dim=-1)
#         attn_weights_ds = self.dropout_layer(attn_weights_ds)
#         cross_attn_drug = torch.matmul(attn_weights_ds, v_se.unsqueeze(1)).transpose(1, 2).contiguous().view(batch_drug,
#                                                                                                              -1)
#         cross_attn_drug = self.out_proj_drug(cross_attn_drug)
#
#         # 副作用查询 + 药物键值
#         attn_scores_sd = torch.matmul(q_se.unsqueeze(2), k_drug.unsqueeze(1).transpose(-2, -1)) * self.scale
#         attn_weights_sd = torch.softmax(attn_scores_sd, dim=-1)
#         attn_weights_sd = self.dropout_layer(attn_weights_sd)
#         cross_attn_se = torch.matmul(attn_weights_sd, v_drug.unsqueeze(1)).transpose(1, 2).contiguous().view(
#             batch_side_effect, -1)
#         cross_attn_se = self.out_proj_side_effect(cross_attn_se)
#
#         return cross_attn_drug, cross_attn_se
#
#
# class GateFusionUnit(nn.Module):
#     """改进的门控自适应融合单元"""
#
#     def __init__(self, feature_dim, fusion_type='concat'):
#         super(GateFusionUnit, self).__init__()
#         self.feature_dim = feature_dim
#         self.fusion_type = fusion_type
#
#         if fusion_type == 'concat':
#             # 使用连接的方式，需要调整输出维度
#             gate_input_dim = feature_dim * 2
#             self.gate = nn.Sequential(
#                 nn.Linear(gate_input_dim, feature_dim),
#                 nn.ReLU(),
#                 nn.Linear(feature_dim, feature_dim),
#                 nn.Sigmoid()
#             )
#             self.output_proj = nn.Linear(feature_dim * 2, feature_dim)
#         else:  # weighted
#             gate_input_dim = feature_dim * 3  # 特征1, 特征2, 组合特征
#             self.gate = nn.Sequential(
#                 nn.Linear(gate_input_dim, feature_dim),
#                 nn.ReLU(),
#                 nn.Linear(feature_dim, 2),
#                 nn.Softmax(dim=-1)
#             )
#
#     def forward(self, feature1, feature2):
#         # 确保两个特征张量具有相同的形状
#         min_size = min(feature1.size(0), feature2.size(0))
#         feature1 = feature1[:min_size]
#         feature2 = feature2[:min_size]
#
#         if self.fusion_type == 'concat':
#             # 连接特征并应用门控
#             combined_features = torch.cat([feature1, feature2], dim=-1)
#             gate_values = self.gate(combined_features)
#             # 门控原始特征
#             gated_features = feature1 * gate_values + feature2 * (1 - gate_values)
#             # 最终投影
#             fused_features = self.output_proj(combined_features)
#             return fused_features
#         else:  # weighted
#             # 计算组合特征
#             combined = feature1 + feature2
#             gate_input = torch.cat([feature1, feature2, combined], dim=-1)
#             gate_weights = self.gate(gate_input)
#
#             # 使用门控权重融合特征
#             fused_features = gate_weights[:, 0:1] * feature1 + gate_weights[:, 1:2] * feature2
#             return fused_features
"""
A3Net
"""

class A3_Net(torch.nn.Module):
    def __init__(self, input_dim=109, input_dim_e=243, output_dim=200, output_dim_e=64, dropout=0.2, heads=10):
        super(A3_Net, self).__init__()

        self.fc_1= nn.Linear(input_dim,output_dim)
        self.fc_2 = nn.Linear(input_dim_e, output_dim)
        self.att = nn.TransformerEncoderLayer(output_dim, 8)
        self.Att = nn.TransformerEncoder(self.att,num_layers=6)
        # graph layers : drug
        self.gcn1 = GATConv(input_dim, 128, heads=heads)
        self.gcn2 = GATConv(128 * heads, output_dim, heads=heads)
        self.gcn5 = GATConv(output_dim * heads, output_dim)
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        # # graph layers : sideEffect
        self.gcn3 = GATConv(input_dim_e, 128, heads=heads)
        self.gcn4 = GATConv(128 * heads, output_dim, heads=heads)
        self.gcn6 = GATConv(output_dim * heads, output_dim)
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)
        # activation and regularization
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm([input_dim])
        self.norm2 = nn.LayerNorm([input_dim_e])
        self.norm3 = nn.LayerNorm([200])
        self.norm4 = nn.LayerNorm([200])

        self.norm_1 = nn.LayerNorm([input_dim])
        self.norm_2 = nn.LayerNorm([1280])
        self.norm_3 = nn.LayerNorm([2000])

        self.norm_e_1 = nn.LayerNorm([input_dim_e])
        self.norm_e_2 = nn.LayerNorm([1280])
        self.norm_e_3 = nn.LayerNorm([2000])

        self.diag = DiagLayer(in_dim=output_dim)


    def forward(self, data, data_e, DF=False, not_FC=True,alpha=0.15):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_e, edge_index_e = data_e.x, data_e.edge_index

        x = self.norm1(x)
        x_fc=self.relu(self.fc_1(x))
        x_e = self.norm2(x_e)
        x_e_fc=self.relu(self.fc_2(x_e))

        x_x_e = torch.cat((x_fc, x_e_fc), dim=0)
        x_x_e = self.relu(self.Att(x_x_e))
        drug_emb0, si_eff_emb0 = torch.split(x_x_e, [x_x_e.shape[0]-994, 994], dim=0)

        drug_emb0 = global_max_pool(drug_emb0, batch)

        # 药物
        x=self.norm_1(x)
        x = self.relu(self.gcn1(x, edge_index))
        x = self.norm_2(x)
        x = self.relu(self.gcn2(x, edge_index))
        x = self.norm_3(x)
        x = self.relu(self.gcn5(x, edge_index))
        x = global_max_pool(x, batch)  # global max pooling

        # 副作用
        x_e=self.norm_e_1(x_e)
        x_e = self.relu(self.gcn3(x_e, edge_index_e))
        x_e = self.norm_e_2(x_e)
        x_e = self.relu(self.gcn4(x_e, edge_index_e))
        x_e = self.norm_e_3(x_e)
        x_e = self.relu(self.gcn6(x_e, edge_index_e))

        x=(1-alpha)*x+alpha*drug_emb0
        x_e=(1-alpha*x_e)+alpha*si_eff_emb0

        if not not_FC:
            x=self.norm3(x)
            x = self.relu(self.fc_g1(x))
            x = self.fc_g2(x)

            x_e=self.norm4(x_e)
            x_e = self.relu(self.fc_g3(x_e))
            x_e = self.fc_g4(x_e)

        # 结合
        x_ = self.diag(x) if DF else x

        xc = torch.matmul(x_, x_e.T)

        return xc, x, x_e
"""
修改后的A3Net（即CAFNet）
"""


# class A3_Net(torch.nn.Module):
#     def __init__(self, input_dim=109, input_dim_e=243, output_dim=200, output_dim_e=64, dropout=0.2, heads=10):
#         super(A3_Net, self).__init__()
#
#         self.fc_1 = nn.Linear(input_dim, output_dim)
#         self.fc_2 = nn.Linear(input_dim_e, output_dim)
#
#         # 使用双向交叉注意力替换自注意力
#         self.cross_attention = CrossAttention(embed_dim=output_dim, num_heads=8)
#
#         # 门控融合单元
#         self.gate_fusion_drug = GateFusionUnit(output_dim)
#         self.gate_fusion_side_effect = GateFusionUnit(output_dim)
#
#         # graph layers : drug
#         self.gcn1 = GATConv(input_dim, 128, heads=heads)
#         self.gcn2 = GATConv(128 * heads, output_dim, heads=heads)
#         self.gcn5 = GATConv(output_dim * heads, output_dim)
#         self.fc_g1 = nn.Linear(output_dim, output_dim)
#         self.fc_g2 = nn.Linear(output_dim, output_dim)
#
#         # graph layers : sideEffect
#         self.gcn3 = GATConv(input_dim_e, 128, heads=heads)
#         self.gcn4 = GATConv(128 * heads, output_dim, heads=heads)
#         self.gcn6 = GATConv(output_dim * heads, output_dim)
#         self.fc_g3 = nn.Linear(output_dim, output_dim)
#         self.fc_g4 = nn.Linear(output_dim, output_dim)
#
#         # activation and regularization
#         self.relu = nn.ReLU()
#         self.norm1 = nn.LayerNorm([input_dim])
#         self.norm2 = nn.LayerNorm([input_dim_e])
#         self.norm3 = nn.LayerNorm([200])
#         self.norm4 = nn.LayerNorm([200])
#
#         self.norm_1 = nn.LayerNorm([input_dim])
#         self.norm_2 = nn.LayerNorm([1280])
#         self.norm_3 = nn.LayerNorm([2000])
#
#         self.norm_e_1 = nn.LayerNorm([input_dim_e])
#         self.norm_e_2 = nn.LayerNorm([1280])
#         self.norm_e_3 = nn.LayerNorm([2000])
#
#         self.diag = DiagLayer(in_dim=output_dim)
#
#     def forward(self, data, data_e, DF=False, not_FC=True, alpha=0.15):
#         # graph input feed-forward
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x_e, edge_index_e = data_e.x, data_e.edge_index
#
#         x = self.norm1(x)
#         x_fc = self.relu(self.fc_1(x))
#         x_e = self.norm2(x_e)
#         x_e_fc = self.relu(self.fc_2(x_e))
#
#         # 使用双向交叉注意力进行药物和副作用交互
#         cross_drug, cross_side_effect = self.cross_attention(x_fc, x_e_fc)
#
#         # 全局池化药物特征
#         drug_emb0 = global_max_pool(x_fc + cross_drug, batch)  # 融合原始特征和交叉注意力特征
#         si_eff_emb0 = x_e_fc + cross_side_effect  # 融合原始特征和交叉注意力特征
#
#         # 药物
#         x = self.norm_1(x)
#         x = self.relu(self.gcn1(x, edge_index))
#         x = self.norm_2(x)
#         x = self.relu(self.gcn2(x, edge_index))
#         x = self.norm_3(x)
#         x = self.relu(self.gcn5(x, edge_index))
#         x = global_max_pool(x, batch)  # global max pooling
#
#         # 副作用
#         x_e = self.norm_e_1(x_e)
#         x_e = self.relu(self.gcn3(x_e, edge_index_e))
#         x_e = self.norm_e_2(x_e)
#         x_e = self.relu(self.gcn4(x_e, edge_index_e))
#         x_e = self.norm_e_3(x_e)
#         x_e = self.relu(self.gcn6(x_e, edge_index_e))
#
#         # 使用门控自适应融合单元替换固定权重融合
#         if x.size(0) != drug_emb0.size(0):
#             raise ValueError(f"drug fusion mismatch: {x.size(0)} vs {drug_emb0.size(0)}")
#         if x_e.size(0) != si_eff_emb0.size(0):
#             raise ValueError(f"side-effect fusion mismatch: {x_e.size(0)} vs {si_eff_emb0.size(0)}")
#         x = self.gate_fusion_drug(x, drug_emb0)
#         x_e = self.gate_fusion_side_effect(x_e, si_eff_emb0)
#
#         if not not_FC:
#             x = self.norm3(x)
#             x = self.relu(self.fc_g1(x))
#             x = self.fc_g2(x)
#
#             x_e = self.norm4(x_e)
#             x_e = self.relu(self.fc_g3(x_e))
#             x_e = self.fc_g4(x_e)
#
#         # 结合
#         x_ = self.diag(x) if DF else x
#
#         xc = torch.matmul(x_, x_e.T)
#
#         return xc, x, x_e


class CAFNet(torch.nn.Module):
    def __init__(self, input_dim=109, input_dim_e=243, output_dim=200, output_dim_e=64, dropout=0.2, heads=10,
                 use_cross_attn=True, fusion_mode="gate", gate_mode="new", fusion_alpha=0.5,
                 gat_dropout=0.0):
        super(CAFNet, self).__init__()

        self.use_cross_attn = use_cross_attn
        self.fusion_mode = fusion_mode
        self.gate_mode = gate_mode
        self.fusion_alpha = fusion_alpha

        self.fc_1 = nn.Linear(input_dim, output_dim)
        self.fc_2 = nn.Linear(input_dim_e, output_dim)

        self.cross_attention = CrossAttention(embed_dim=output_dim, num_heads=8)

        if gate_mode == "old":
            self.gate_fusion_drug = GateFusionUnitOld(output_dim)
            self.gate_fusion_side_effect = GateFusionUnitOld(output_dim)
        else:
            self.gate_fusion_drug = GateFusionUnit(output_dim)
            self.gate_fusion_side_effect = GateFusionUnit(output_dim)

        # graph layers : drug
        self.gcn1 = GATConv(input_dim, 128, heads=heads, dropout=gat_dropout)
        self.gcn2 = GATConv(128 * heads, output_dim, heads=heads, dropout=gat_dropout)
        self.gcn5 = GATConv(output_dim * heads, output_dim, dropout=gat_dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        # graph layers : sideEffect
        self.gcn3 = GATConv(input_dim_e, 128, heads=heads, dropout=gat_dropout)
        self.gcn4 = GATConv(128 * heads, output_dim, heads=heads, dropout=gat_dropout)
        self.gcn6 = GATConv(output_dim * heads, output_dim, dropout=gat_dropout)
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)

        # activation and regularization
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm([input_dim])
        self.norm2 = nn.LayerNorm([input_dim_e])
        self.norm3 = nn.LayerNorm([200])
        self.norm4 = nn.LayerNorm([200])

        self.norm_1 = nn.LayerNorm([input_dim])
        self.norm_2 = nn.LayerNorm([1280])
        self.norm_3 = nn.LayerNorm([2000])

        self.norm_e_1 = nn.LayerNorm([input_dim_e])
        self.norm_e_2 = nn.LayerNorm([1280])
        self.norm_e_3 = nn.LayerNorm([2000])

        self.diag = DiagLayer(in_dim=output_dim)

    def forward(self, data, data_e, DF=False, not_FC=True, alpha=0.15):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_e, edge_index_e = data_e.x, data_e.edge_index

        x = self.norm1(x)
        x_fc = self.relu(self.fc_1(x))
        x_e = self.norm2(x_e)
        x_e_fc = self.relu(self.fc_2(x_e))

        if self.use_cross_attn:
            cross_drug, cross_side_effect = self.cross_attention(x_fc, x_e_fc)
        else:
            cross_drug = torch.zeros_like(x_fc)
            cross_side_effect = torch.zeros_like(x_e_fc)

        drug_emb0 = global_max_pool(x_fc + cross_drug, batch)
        si_eff_emb0 = x_e_fc + cross_side_effect

        # drug
        x = self.norm_1(x)
        x = self.relu(self.gcn1(x, edge_index))
        x = self.norm_2(x)
        x = self.relu(self.gcn2(x, edge_index))
        x = self.norm_3(x)
        x = self.relu(self.gcn5(x, edge_index))
        x = global_max_pool(x, batch)

        # side effect
        x_e = self.norm_e_1(x_e)
        x_e = self.relu(self.gcn3(x_e, edge_index_e))
        x_e = self.norm_e_2(x_e)
        x_e = self.relu(self.gcn4(x_e, edge_index_e))
        x_e = self.norm_e_3(x_e)
        x_e = self.relu(self.gcn6(x_e, edge_index_e))

        if self.fusion_mode == "gate":
            if x.size(0) != drug_emb0.size(0):
                raise ValueError(f"drug fusion mismatch: {x.size(0)} vs {drug_emb0.size(0)}")
            if x_e.size(0) != si_eff_emb0.size(0):
                raise ValueError(f"side-effect fusion mismatch: {x_e.size(0)} vs {si_eff_emb0.size(0)}")
            x = self.gate_fusion_drug(x, drug_emb0)
            x_e = self.gate_fusion_side_effect(x_e, si_eff_emb0)
        elif self.fusion_mode == "fixed":
            if x.size(0) != drug_emb0.size(0):
                raise ValueError(f"drug fusion mismatch: {x.size(0)} vs {drug_emb0.size(0)}")
            if x_e.size(0) != si_eff_emb0.size(0):
                raise ValueError(f"side-effect fusion mismatch: {x_e.size(0)} vs {si_eff_emb0.size(0)}")
            x = (1.0 - self.fusion_alpha) * x + self.fusion_alpha * drug_emb0
            x_e = (1.0 - self.fusion_alpha) * x_e + self.fusion_alpha * si_eff_emb0
        elif self.fusion_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        if not not_FC:
            x = self.norm3(x)
            x = self.relu(self.fc_g1(x))
            x = self.fc_g2(x)

            x_e = self.norm4(x_e)
            x_e = self.relu(self.fc_g3(x_e))
            x_e = self.fc_g4(x_e)

        x_ = self.diag(x) if DF else x
        xc = torch.matmul(x_, x_e.T)

        return xc, x, x_e


class CAFNetV2(CAFNet):
    """CAFNet with asymmetric association and frequency prediction heads."""

    def __init__(self, *args, rank_score_mix=0.7, **kwargs):
        super(CAFNetV2, self).__init__(*args, **kwargs)
        output_dim = kwargs.get("output_dim", 200)
        pair_dim = output_dim * 4 + 1
        self.rank_score_mix = rank_score_mix
        self.association_head = nn.Sequential(
            nn.Linear(pair_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, 1),
        )
        self.frequency_head = nn.Sequential(
            nn.Linear(pair_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, 1),
        )
        self.last_assoc_logits = None
        self.last_freq_pred = None

    def _pair_features(self, drug_embed, side_embed, base_score):
        n_drug, dim = drug_embed.size()
        n_side = side_embed.size(0)
        drug_expand = drug_embed.unsqueeze(1).expand(n_drug, n_side, dim)
        side_expand = side_embed.unsqueeze(0).expand(n_drug, n_side, dim)
        pair = torch.cat(
            [
                drug_expand,
                side_expand,
                drug_expand * side_expand,
                torch.abs(drug_expand - side_expand),
                base_score.unsqueeze(-1),
            ],
            dim=-1,
        )
        return pair

    def forward(self, data, data_e, DF=False, not_FC=True, alpha=0.15):
        base_score, drug_embed, side_embed = super(CAFNetV2, self).forward(data, data_e, DF, not_FC, alpha)
        pair = self._pair_features(drug_embed, side_embed, base_score)
        assoc_logits = self.association_head(pair).squeeze(-1)
        freq_pred = self.frequency_head(pair).squeeze(-1)
        self.last_assoc_logits = assoc_logits
        self.last_freq_pred = freq_pred
        score = self.rank_score_mix * assoc_logits + (1.0 - self.rank_score_mix) * freq_pred
        return score, drug_embed, side_embed


class CAFNetOrdinal(CAFNetV2):
    """CAFNetV2 with an ordinal frequency classification head."""

    def __init__(self, *args, ordinal_score_mix=0.2, num_ordinal_classes=6, **kwargs):
        super(CAFNetOrdinal, self).__init__(*args, **kwargs)
        output_dim = kwargs.get("output_dim", 200)
        pair_dim = output_dim * 4 + 1
        self.ordinal_score_mix = ordinal_score_mix
        self.num_ordinal_classes = num_ordinal_classes
        self.ordinal_head = nn.Sequential(
            nn.Linear(pair_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, num_ordinal_classes),
        )
        self.register_buffer(
            "ordinal_values",
            torch.arange(num_ordinal_classes, dtype=torch.float32),
        )
        self.last_ordinal_logits = None
        self.last_ordinal_expected = None

    def forward(self, data, data_e, DF=False, not_FC=True, alpha=0.15):
        base_score, drug_embed, side_embed = CAFNet.forward(self, data, data_e, DF, not_FC, alpha)
        pair = self._pair_features(drug_embed, side_embed, base_score)
        assoc_logits = self.association_head(pair).squeeze(-1)
        freq_pred = self.frequency_head(pair).squeeze(-1)
        ordinal_logits = self.ordinal_head(pair)
        ordinal_expected = torch.sum(
            F.softmax(ordinal_logits, dim=-1) * self.ordinal_values.view(1, 1, -1),
            dim=-1,
        )
        self.last_assoc_logits = assoc_logits
        self.last_freq_pred = freq_pred
        self.last_ordinal_logits = ordinal_logits
        self.last_ordinal_expected = ordinal_expected
        base_rank = self.rank_score_mix * assoc_logits + (1.0 - self.rank_score_mix) * freq_pred
        score = (1.0 - self.ordinal_score_mix) * base_rank + self.ordinal_score_mix * ordinal_expected
        return score, drug_embed, side_embed


class CAFNetDecoupled(CAFNetV2):
    """CAFNetV2 with decoupled ranking and bias-residual frequency prediction."""

    def __init__(
        self,
        *args,
        num_drugs=750,
        num_sides=994,
        pop_weight=0.0,
        bias_weight=1.0,
        assoc_base_weight=1.0,
        assoc_residual_weight=1.0,
        drug_evidence_dim=0,
        evidence_dropout=0.1,
        **kwargs,
    ):
        super(CAFNetDecoupled, self).__init__(*args, **kwargs)
        output_dim = kwargs.get("output_dim", 200)
        self.num_drugs = num_drugs
        self.num_sides = num_sides
        self.pop_weight = pop_weight
        self.bias_weight = bias_weight
        self.assoc_base_weight = assoc_base_weight
        self.assoc_residual_weight = assoc_residual_weight
        self.drug_evidence_dim = int(drug_evidence_dim or 0)
        self.drug_bias = nn.Embedding(num_drugs, 1)
        self.side_bias = nn.Embedding(num_sides, 1)
        if self.drug_evidence_dim > 0:
            self.evidence_encoder = nn.Sequential(
                nn.Linear(self.drug_evidence_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(evidence_dropout),
                nn.Linear(output_dim, output_dim),
            )
            self.evidence_gate = nn.Linear(output_dim * 2, output_dim)
        else:
            self.evidence_encoder = None
            self.evidence_gate = None
        nn.init.zeros_(self.drug_bias.weight)
        nn.init.zeros_(self.side_bias.weight)
        self.register_buffer("side_popularity", torch.zeros(num_sides, dtype=torch.float32))
        self.register_buffer("global_mean", torch.zeros(1, dtype=torch.float32))

    @staticmethod
    def _drug_indices(data, device):
        idx = []
        for item in data.index:
            if torch.is_tensor(item):
                idx.append(int(item.flatten()[0].item()))
            elif isinstance(item, (list, tuple)):
                idx.append(int(item[0]))
            else:
                idx.append(int(item))
        return torch.tensor(idx, dtype=torch.long, device=device)

    def set_frequency_priors(self, side_popularity=None, global_mean=0.0):
        if side_popularity is not None:
            side_pop = torch.as_tensor(side_popularity, dtype=torch.float32, device=self.side_popularity.device)
            if side_pop.numel() != self.num_sides:
                raise ValueError(f"side_popularity length mismatch: {side_pop.numel()} vs {self.num_sides}")
            side_pop = (side_pop - side_pop.mean()) / side_pop.std().clamp_min(1e-6)
            self.side_popularity.copy_(side_pop)
        self.global_mean.fill_(float(global_mean))

    def forward(self, data, data_e, DF=False, not_FC=True, alpha=0.15):
        base_score, drug_embed, side_embed = CAFNet.forward(self, data, data_e, DF, not_FC, alpha)
        if self.drug_evidence_dim > 0 and hasattr(data, "drug_evidence"):
            evidence = data.drug_evidence.to(device=drug_embed.device, dtype=drug_embed.dtype)
            if evidence.dim() == 1:
                evidence = evidence.view(drug_embed.size(0), -1)
            evidence_embed = self.evidence_encoder(evidence)
            gate = torch.sigmoid(self.evidence_gate(torch.cat([drug_embed, evidence_embed], dim=-1)))
            drug_embed = gate * drug_embed + (1.0 - gate) * evidence_embed
            drug_for_score = self.diag(drug_embed) if DF else drug_embed
            base_score = torch.matmul(drug_for_score, side_embed.T)
        pair = self._pair_features(drug_embed, side_embed, base_score)
        assoc_residual = self.association_head(pair).squeeze(-1)
        assoc_logits = self.assoc_base_weight * base_score + self.assoc_residual_weight * assoc_residual
        freq_residual = self.frequency_head(pair).squeeze(-1)

        drug_idx = self._drug_indices(data, base_score.device)
        side_idx = torch.arange(base_score.size(1), device=base_score.device)
        bias = self.drug_bias(drug_idx).view(-1, 1) + self.side_bias(side_idx).view(1, -1)
        freq_pred = freq_residual + self.bias_weight * bias + self.global_mean.view(1, 1)

        rank_score = self.rank_score_mix * assoc_logits + (1.0 - self.rank_score_mix) * freq_residual
        if self.pop_weight != 0:
            rank_score = rank_score + self.pop_weight * self.side_popularity.view(1, -1)

        self.last_assoc_logits = assoc_logits
        self.last_freq_pred = freq_pred
        return rank_score, drug_embed, side_embed


class CAFNetLoss(nn.Module):
    """
    Loss helper for CAFNet.
    Supports BCEWithLogits and Focal loss with optional class imbalance handling.
    """

    def __init__(
        self,
        loss_type="bce",
        pos_weight=None,
        gamma=2.0,
        alpha=None,
        label_smoothing=0.0,
        reduction="mean",
        eps=1e-6,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.eps = eps

    @staticmethod
    def _apply_mask(loss, mask):
        if mask is None:
            return loss
        loss = loss * mask
        if loss.ndim == 0:
            return loss
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom

    def _smooth_targets(self, targets):
        if self.label_smoothing <= 0.0:
            return targets
        return targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

    def forward(self, logits, targets, mask=None):
        targets = targets.to(dtype=logits.dtype)
        targets = self._smooth_targets(targets)

        pos_weight = None
        if self.pos_weight is not None:
            if isinstance(self.pos_weight, torch.Tensor):
                pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
            else:
                pos_weight = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)

        if self.loss_type == "bce":
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight, reduction="none"
            )
            loss = self._apply_mask(loss, mask)
            if self.reduction == "sum" and mask is None:
                return loss.sum()
            if self.reduction == "none":
                return loss
            return loss.mean() if mask is None else loss

        if self.loss_type == "focal":
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight, reduction="none"
            )
            probs = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
            p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
            focal_term = (1.0 - p_t) ** self.gamma

            if self.alpha is not None:
                alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
                focal_term = focal_term * alpha_t

            loss = focal_term * bce
            loss = self._apply_mask(loss, mask)
            if self.reduction == "sum" and mask is None:
                return loss.sum()
            if self.reduction == "none":
                return loss
            return loss.mean() if mask is None else loss

        raise ValueError(f"Unsupported loss_type: {self.loss_type}")
