#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAVG(w,dw,sample_nums):
    # 计算第所有客户端样本数量比例
    num_ratio = [sample_nums[i]/sum(sample_nums) for i in range(len(sample_nums))]
    # 计算第0个客户端待聚合的参数更新
    dw_avg = copy.deepcopy(dw[0])
    for k in dw_avg.keys():
        dw_avg[k] = num_ratio[0] * dw_avg[k]
    # 聚合所有客户端的参数更新，加上样本数量比例为权值
    for k in dw_avg.keys(): #k代表模型中的每种参数，例如'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'等
        for i in range(1, len(dw)):
            dw_avg[k] += num_ratio[i] * dw[i][k]
    # 更新联合模型的参数
    for k in w.keys():
        w[k] = w[k] + dw_avg[k]
    return w

def FedMEAN(w,dw):
    dw_mean = copy.deepcopy(dw[0])
    for k in dw_mean.keys(): #k代表模型中的每种参数，例如'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'等
        for i in range(1, len(dw)):
            dw_mean[k] += dw[i][k]
        dw_mean[k] = torch.div(dw_mean[k], len(dw))
    # 更新联合模型的参数
    for k in w.keys():
        w[k] = w[k] + dw_mean[k]
    return w


def FedRWA(w, dw, risk_scores, masks=None):
    """
    支持 FedSelect 掩码机制的风险权值联邦聚合算法 (FedRWA)
    :param w: 联合模型的上一轮参数 (OrderedDict)
    :param dw: 各本地模型的参数更新值 (List[OrderedDict])
    :param risk_scores: 各本地模型的风险分数
    :param masks: 各客户端的掩码列表 (List[OrderedDict])，1代表本地，0代表全局
    :return: 聚合后的全局模型参数
    """

    total_risk = sum(risk_scores)+ 1e-9
    dw_numerator = copy.deepcopy(dw[0])
    weight_denominator = copy.deepcopy(dw[0])

    for k in dw_numerator.keys():
        dw_numerator[k].zero_()
        weight_denominator[k].zero_()

    for i in range(len(dw)):
        alpha = risk_scores[i] / total_risk

        for k in dw[i].keys():
            if masks is not None and masks[i] is not None and k in masks[i]:
                valid_map = (masks[i][k] == 0).float()
            else:
                valid_map = 1.0

            if "num_batches_tracked" in k:
                continue

            dw_numerator[k] += dw[i][k] * alpha * valid_map
            weight_denominator[k] += alpha * valid_map

    w_updated = copy.deepcopy(w)
    for k in w_updated.keys():
        if k in dw_numerator:
            update_gate = (weight_denominator[k] > 0).float()
            safe_denom = weight_denominator[k] + 1e-12

            avg_dw = (dw_numerator[k] / safe_denom) * update_gate
            w_updated[k] = w_updated[k] + avg_dw

    return w_updated


def FedPM_Aggregate(global_scores, client_masks, lr_g=0.1):
    with torch.no_grad():
        m = len(client_masks)
        for name in global_scores.keys():
            # 1. 提取所有客户端上传的采样掩码并计算经验均值
            avg_prob = torch.stack([client_masks[i][name] for i in range(m)], dim=0).mean(dim=0)

            # 2. 【机制对齐：基于伯努利似然的梯度下降】
            p_global = torch.sigmoid(global_scores[name])
            p_new = (1 - lr_g) * p_global + lr_g * avg_prob

            p_new = torch.clamp(p_new, 1e-6, 1 - 1e-6)

            global_scores[name] = torch.log(
                p_new / (1 - p_new)
            )

    return global_scores