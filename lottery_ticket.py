# 导入库
import copy
import numpy as np
# from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import types
from collections import OrderedDict
from typing import List, Tuple, Dict, OrderedDict, Optional, Union


def eval_per_layer_sparsity(mask: OrderedDict) -> List[Tuple[str, str, str, float]]:
    """计算掩码中每个权重层的稀疏度信息。

    参数:
        mask: 包含模型参数二进制掩码的 OrderedDict

    返回:
        每个权重层的统计信息列表，每个元素为 (1的数量, 0的数量, 层名, 稀疏度)
    """
    return [
        (
            f"1: {torch.count_nonzero(mask[name])}",
            f"0: {torch.count_nonzero(1-mask[name])}",
            name,
            (
                torch.count_nonzero(1 - mask[name])
                / (
                    torch.count_nonzero(mask[name])
                    + torch.count_nonzero(1 - mask[name])
                )
            ).item(),
        )
        for name in mask.keys()
        if "weight" in name
    ]


def eval_layer_sparsity(
    mask: OrderedDict, layer_name: str
) -> Tuple[str, str, str, float]:
    """计算掩码中某个指定层的稀疏度信息。

    参数:
        mask: 包含模型参数二进制掩码的 OrderedDict
        layer_name: 要分析的层名

    返回:
        (1的数量, 0的数量, 层名, 稀疏度) 的元组
    """
    return (
        f"1: {torch.count_nonzero(mask[layer_name])}",
        f"0: {torch.count_nonzero(1-mask[layer_name])}",
        layer_name,
        (
            torch.count_nonzero(1 - mask[layer_name])
            / (
                torch.count_nonzero(mask[layer_name])
                + torch.count_nonzero(1 - mask[layer_name])
            )
        ).item(),
    )


def print_nonzeros(
    model: OrderedDict, verbose: bool = False, invert: bool = False
) -> float:
    """打印模型中非零参数的统计信息。

    参数:
        model: 包含模型参数的 OrderedDict
        verbose: 是否打印详细的每层统计信息
        invert: 是否统计 0 的数量而不是非零数量

    返回:
        被剪枝参数的百分比
    """
    nonzero = total = 0
    for name, p in model.items():
        tensor = p.data.cpu().numpy()
        nz_count = (
            np.count_nonzero(tensor) if not invert else np.count_nonzero(1 - tensor)
        )
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        if verbose:
            print(
                f"{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}"
            )
    if verbose:
        print(
            f"alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)"
        )
    return 100 * (total - nonzero) / total


def print_lth_stats(mask: OrderedDict, invert: bool = False) -> None:
    """打印 Lottery Ticket Hypothesis (LTH) 掩码的稀疏度信息。

    参数:
        mask: 包含二进制掩码的 OrderedDict
        invert: 是否反转稀疏度计算
    """
    current_prune = print_nonzeros(mask, invert=invert)
    print(f"Mask Sparsity: {current_prune:.2f}%")


def _violates_bound(
    mask: torch.Tensor, bound: Optional[float] = None, invert: bool = False
) -> bool:
    """判断掩码稀疏度是否超过指定的上限。

    参数:
        mask: 二进制掩码张量
        bound: 最大允许稀疏度
        invert: 是否反转稀疏度计算

    返回:
        如果超过上限返回 True，否则返回 False
    """
    if invert:
        return (
            torch.count_nonzero(mask)
            / (torch.count_nonzero(mask) + torch.count_nonzero(1 - mask))
        ).item() > bound
    else:
        return (
            torch.count_nonzero(1 - mask)
            / (torch.count_nonzero(mask) + torch.count_nonzero(1 - mask))
        ).item() > bound


def init_mask(model: nn.Module) -> OrderedDict:
    """为模型参数初始化全 1 的二进制掩码。

    参数:
        model: 神经网络模型

    返回:
        包含全 1 掩码的 OrderedDict
    """
    mask = OrderedDict()
    for name, param in model.named_parameters():
        mask[name] = torch.ones_like(param)
    return mask


def init_mask_zeros(model: nn.Module) -> OrderedDict:
    """为模型参数初始化全 0 的二进制掩码。

    参数:
        model: 神经网络模型

    返回:
        包含全 0 掩码的 OrderedDict
    """
    mask = OrderedDict()
    for name, param in model.named_parameters():
        mask[name] = torch.zeros_like(param)
    return mask


def init_mask_ones(model: nn.Module) -> OrderedDict:
    """为模型参数初始化全 0 的二进制掩码。

    参数:
        model: 神经网络模型

    返回:
        包含全 0 掩码的 OrderedDict
    """
    mask = OrderedDict()
    for name, param in model.named_parameters():
        mask[name] = torch.ones_like(param)
    return mask


def get_mask_from_delta(
    prune_percent: float,
    current_state_dict: OrderedDict,
    prev_state_dict: OrderedDict,
    current_mask: OrderedDict,
    bound: float = 0.80,
    invert: bool = True,
) -> OrderedDict:
    """基于模型两次状态之间的权重变化生成新的掩码。

    参数:
        prune_percent: 要剪枝的比例
        current_state_dict: 当前模型状态
        prev_state_dict: 上一次的模型状态
        current_mask: 当前二进制掩码
        bound: 最大允许稀疏度
        invert: 是否反转剪枝逻辑

    返回:
        根据权重变化更新后的二进制掩码
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return_mask = copy.deepcopy(current_mask)
    delta_tensor_dict = {}
    for name, param in current_state_dict.items():
        if "weight" in name:
            if _violates_bound(current_mask[name], bound=bound, invert=invert):
                continue
            tensor = param.data.cpu().numpy()
            compare_tensor = prev_state_dict[name].cpu().numpy()
            delta_tensor = np.abs(tensor - compare_tensor)
            delta_tensor_dict[name] = torch.from_numpy(delta_tensor).to(device)

            delta_percentile_tensor = (
                delta_tensor[current_mask[name].cpu().numpy() == 1]
                if not invert
                else delta_tensor[current_mask[name].cpu().numpy() == 0]
            )
            sorted_weights = np.sort(np.abs(delta_percentile_tensor))
            if sorted_weights.size == 0:
                continue
            if not invert:
                cutoff_index = np.round(prune_percent * sorted_weights.size).astype(int)
                cutoff = sorted_weights[cutoff_index]

                # 将张量转换为 numpy 进行计算
                new_mask = np.where(
                    abs(delta_tensor) <= cutoff, 0, return_mask[name].cpu().numpy()
                )
                return_mask[name] = torch.from_numpy(new_mask).to(device)
            else:
                cutoff_index = np.round((1 - prune_percent) * sorted_weights.size).astype(int)
                cutoff = sorted_weights[cutoff_index]

                new_mask_np = np.where(abs(delta_tensor) >= cutoff, 1, 0)
                new_mask_tensor = torch.from_numpy(new_mask_np).to(device)

                retain_prob = 0.9
                random_gate = (torch.rand_like(current_mask[name].to(device).float()) < retain_prob).int()

                return_mask[name] = torch.clamp(current_mask[name].to(device) * random_gate + new_mask_tensor, 0, 1)
            # else:
            #     cutoff_index = np.round(
            #         (1 - prune_percent) * sorted_weights.size
            #     ).astype(int)
            #     cutoff = sorted_weights[cutoff_index]
            #
            #     # 设置衰减系数，每轮随机遗忘一些特色参数的掩码
            #     # retain_prob = 0.9
            #     # old_mask = return_mask[name].cpu()
            #     # R = (torch.rand_like(old_mask.float()) < retain_prob).int()
            #     # decayed_prev_mask = old_mask * R
            #     #
            #     # new_mask_tensor = torch.from_numpy(new_mask).to(device)
            #     #
            #     # final_mask = torch.max(decayed_prev_mask, new_mask_tensor)
            #     # return_mask[name] = final_mask
            #
            #     # 将张量转换为 numpy 进行计算
            #     new_mask = np.where(
            #         abs(delta_tensor) >= cutoff, 1, return_mask[name].cpu().numpy()
            #     )
            #     return_mask[name] = torch.from_numpy(new_mask).to(device)

    # print(eval_per_layer_sparsity(return_mask))
    # print("Sparsity per layer after update:")
    # print(eval_per_layer_sparsity(return_mask))
    return return_mask, delta_tensor_dict


def delta_update(
    prune_percent: float,
    current_state_dict: OrderedDict,
    prev_state_dict: OrderedDict,
    current_mask: OrderedDict,
    bound: float = 0.80,
    invert: bool = False,
) -> OrderedDict:
    """基于权重变化更新掩码。

    参数:
        prune_percent: 要剪枝的比例
        current_state_dict: 当前模型状态
        prev_state_dict: 上一次的模型状态
        current_mask: 当前二进制掩码
        bound: 最大允许稀疏度
        invert: 是否反转剪枝逻辑

    返回:
        更新后的二进制掩码
    """
    mask, delta_tensor_dict = get_mask_from_delta(
        prune_percent,
        current_state_dict,
        prev_state_dict,
        current_mask,
        bound=bound,
        invert=invert,
    )
    # print_lth_stats(mask, invert=invert)
    return mask, delta_tensor_dict
