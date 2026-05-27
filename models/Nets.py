#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 120)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(120, dim_out)

    def forward(self, x):
        # x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        # x = self.dropout(x)
        code = self.relu(x)
        y = self.layer_hidden(code)
        return code,y


class mlp(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(mlp, self).__init__()

        self.layer_input = nn.Sequential(
            nn.Linear(dim_in, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.layer_hidden = nn.Linear(64, dim_out)

    def forward(self, x):
        code = self.layer_input(x)
        y = self.layer_hidden(code)
        return code, y

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        # 1. 冻结原始权重，FedPM 不更新 w
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

        self.weight.requires_grad = False

        # 2. 定义分数矩阵 s，形状与 weight 一致
        self.score = nn.Parameter(torch.Tensor(self.weight.size()))
        # 初始化分数，通常设为能产生 0.5 概率的值（即 sigma(0) = 0.5）
        nn.init.normal_(self.score, mean=0.0, std=0.01)

    def reset_mask(self):
        with torch.no_grad():
            probs = torch.sigmoid(self.score)
            self.sampled_mask = torch.bernoulli(probs)

    def forward(self, input):
        # 3. 使用 Sigmoid 将分数转为概率
        probs = torch.sigmoid(self.score)

        if self.training:
            if self.sampled_mask is None:
                self.reset_mask()

            mask = self.sampled_mask + probs - probs.detach()
        else:
            # 5. 测试时：根据阈值（通常0.5）生成确定性掩码
            mask = (probs>=0.5).float()

        masked_weight = self.weight * mask

        return F.linear(input, masked_weight, self.bias)


def convert_to_fedpm(model):
    # 使用 named_modules 递归查找所有子模块
    for name, module in model.named_modules():
        # 遍历该模块的所有直接子模块
        for sub_name, sub_module in module.named_children():
            if isinstance(sub_module, nn.Linear) and not isinstance(sub_module, MaskedLinear):
                # 创建新层
                new_layer = MaskedLinear(sub_module.in_features, sub_module.out_features, sub_module.bias is not None)
                # 拷贝原始初始化权重
                new_layer.weight.data.copy_(sub_module.weight.data)
                if sub_module.bias is not None:
                    new_layer.bias.data.copy_(sub_module.bias.data)

                # 将新层移动到原层所在的设备
                new_layer.to(sub_module.weight.device)

                # 动态替换
                setattr(module, sub_name, new_layer)
    return model
