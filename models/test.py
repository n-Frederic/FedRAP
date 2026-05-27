#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, average_precision_score
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd


def test_model(net_g, datatest, args, idxs=None, threshold=None):
    rt = {}
    net_g.eval()
    test_loss = 0
    correct = 0

    if idxs is not None:
        data_loader = DataLoader(Subset(datatest, idxs), batch_size=args.bs)
    else:
        data_loader = DataLoader(datatest, batch_size=args.bs)

    y_labels = torch.tensor([])
    y_preds = torch.tensor([])
    y_preds_values = torch.tensor([])

    with torch.no_grad():
        for idx, (data, target, _) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            _, log_probs = net_g(data)

            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_preds_value = F.softmax(log_probs, dim=1)[:, 1]

            thresh = threshold if threshold else 0.5
            y_pred = (y_preds_value > thresh).long().unsqueeze(1)

            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            y_preds = torch.cat((y_preds, y_pred.cpu()), 0)
            y_preds_values = torch.cat((y_preds_values, y_preds_value.cpu()), 0)
            y_labels = torch.cat((y_labels, target.data.view_as(y_pred).cpu()), 0)

    # 基础数值计算
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)

    y_labels_np = y_labels.detach().numpy().flatten()
    y_preds_np = y_preds.detach().numpy().flatten()
    y_preds_values_np = y_preds_values.detach().numpy().flatten()

    # # ================= 概率分布分析 =================
    #
    # pos_probs = y_preds_values_np[y_labels_np == 1]
    # neg_probs = y_preds_values_np[y_labels_np == 0]
    #
    # print("\n========== Probability Distribution ==========")
    #
    # print(f"Total Samples : {len(y_preds_values_np)}")
    # print(f"Fraud Samples : {len(pos_probs)}")
    # print(f"Normal Samples: {len(neg_probs)}")
    #
    # if len(pos_probs) > 0:
    #     print("\n[Fraud Probabilities]")
    #     print(
    #         "mean={:.6f} | median={:.6f} | max={:.6f} | min={:.6f}".format(
    #             pos_probs.mean(),
    #             np.median(pos_probs),
    #             pos_probs.max(),
    #             pos_probs.min()
    #         )
    #     )
    #
    # if len(neg_probs) > 0:
    #     print("\n[Normal Probabilities]")
    #     print(
    #         "mean={:.6f} | median={:.6f} | max={:.6f} | min={:.6f}".format(
    #             neg_probs.mean(),
    #             np.median(neg_probs),
    #             neg_probs.max(),
    #             neg_probs.min()
    #         )
    #     )
    #
    # print("\n[Top 20 Highest Fraud Probabilities]")
    # top_probs = np.sort(y_preds_values_np)[-20:][::-1]
    #
    # for i, p in enumerate(top_probs):
    #     print(f"{i + 1:02d}: {p:.6f}")
    #
    # print("\n[Threshold Analysis]")
    #
    # pred_pos = (y_preds_values_np > thresh).sum()
    # ratio = pred_pos / len(y_preds_values_np)
    #
    # print(
    #     "[BEST THRESHOLD CHECK] "
    #     "threshold={:.6f} | pred_pos={} | ratio={:.6f} | "
    #     "prob_mean={:.6f} | prob_max={:.6f} | prob_min={:.6f}".format(
    #         thresh,
    #         pred_pos,
    #         ratio,
    #         y_preds_values_np.mean(),
    #         y_preds_values_np.max(),
    #         y_preds_values_np.min()
    #     )
    # )
    # print("==============================================\n")

    # --- 关键修改：只在有两类数据时计算复杂指标，否则只存基础数据 ---
    if len(np.unique(y_labels_np)) > 1:
        rt['auc'] = roc_auc_score(y_labels_np, y_preds_values_np)
        rt['f1'] = f1_score(y_labels_np, y_preds_np)
    else:
        rt['auc'] = 0.5
        rt['f1'] = 0.0

    # 金额计算逻辑（注意：如果使用了 Subset，iloc 索引需对应原始数据位置）
    # 这里的 index1 是相对于当前 data_loader 的局部索引，映射回全量数据
    amount1 = 0.0
    amount1_pred = 0.0
    if 'Amount' in datatest.data.columns:
        if idxs is not None:
            actual_indices = np.array(idxs)
            fraud_mask = (y_labels_np == 1)
            pred_fraud_mask = (y_labels_np == 1) & (y_preds_np == 1)

            # 仅在有对应样本时计算，防止空索引报错
            if np.any(fraud_mask):
                amount1 = datatest.data.iloc[actual_indices[fraud_mask]].Amount.sum()
            if np.any(pred_fraud_mask):
                amount1_pred = datatest.data.iloc[actual_indices[pred_fraud_mask]].Amount.sum()
        else:
            index1 = [i for i in range(len(y_labels_np)) if y_labels_np[i] == 1]
            index1_pred = [i for i in range(len(y_labels_np)) if (y_labels_np[i] == 1 and y_preds_np[i] == 1)]

            if index1:
                amount1 = datatest.data.iloc[index1].Amount.sum()
            if index1_pred:
                amount1_pred = datatest.data.iloc[index1_pred].Amount.sum()
    else:
        amount1 = 0.0
        amount1_pred = 0.0

    # 返回供 main_fed 汇总的原始结果
    rt['loss'] = test_loss
    rt['accuracy'] = accuracy
    rt['amount1'] = amount1
    rt['amount1_pred'] = amount1_pred
    rt['y_labels'] = y_labels_np
    rt['y_preds'] = y_preds_np
    rt['y_probs'] = y_preds_values_np

    return rt

