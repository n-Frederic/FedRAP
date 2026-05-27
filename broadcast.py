import torch
from typing import OrderedDict


def broadcast_server_to_client_initialization(
        server_weights: OrderedDict[str, torch.Tensor],
        mask: OrderedDict[str, torch.Tensor],
        client_initialization: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """将服务器权重广播给客户端初始化（仅对 mask 为非本地参数的位置进行覆盖）

    参数:
        fusion_module:
        delta_tensor_dict:
        epsilon:
        server_weights: 服务器模型的 state dict
        mask: 二值 mask，1 表示本地参数，0 表示全局参数
        client_initialization: 客户端待更新的模型 state dict

    返回:
        覆盖非本地参数后的客户端模型 state dict
    """
    # if fusion_module is not None:
    #     print(">>> USING MODIFIED broadcast <<<")
    #     return fusion_module(client_initialization, server_weights)
    for key in client_initialization.keys():
        # 只在 mask 非本地（0）的位置覆盖客户端参数
        if "weight" in key or "bias" in key:
            # local_param = client_initialization[key]
            # global_param = server_weights[key]
            #
            # if delta_tensor_dict is None or key not in delta_tensor_dict:
            #     client_initialization[key][mask[key] == 0] = global_param[mask[key] == 0]
            # else:
            #     print(">>> USING MODIFIED fedselect.py <<<")
            #     delta_tensor = delta_tensor_dict[key].to(local_param.device)
            #     mean = delta_tensor.mean()
            #     std = delta_tensor.std() + epsilon
            #     normalized = (delta_tensor - mean) / std
            #     weight_factor = torch.sigmoid(normalized)
            #     weight_factor = weight_factor * (1 - mask[key].float())
            #     client_initialization[key] = weight_factor * local_param + (1 - weight_factor) * global_param
            #     print(
            #         key,
            #         weight_factor.min().item(),
            #         weight_factor.max().item(),
            #         weight_factor.mean().item()
            #     )
            client_initialization[key][mask[key] == 0] = server_weights[key][
                mask[key] == 0
            ]

    return client_initialization


def div_server_weights(
        server_weights: OrderedDict[str, torch.Tensor],
        server_mask: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """对服务器权重进行除法归一化，仅在 mask 非零的位置进行除法。

    参数:
        server_weights: 服务器模型的 state dict
        server_mask: 记录每个参数的贡献客户端数量的 mask

    返回:
        按贡献数归一化后的服务器权重
    """
    for key in server_weights.keys():
        # 只在 server_mask 非零的位置进行除法
        if "weight" in key or "bias" in key:
            server_weights[key][server_mask[key] != 0] /= server_mask[key][
                server_mask[key] != 0
            ]
    return server_weights


def add_masks(
        server_dict: OrderedDict[str, torch.Tensor],
        client_dict: OrderedDict[str, torch.Tensor],
        invert: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """累加客户端 mask 到服务器 mask 中。

    参数:
        server_dict: 服务器端的 mask 累加器
        client_dict: 客户端 mask
        invert: 是否在累加前对 mask 取反

    返回:
        更新后的服务器 mask 累加器
    """
    for key in client_dict.keys():
        if "weight" in key or "bias" in key:
            if key not in server_dict.keys():
                server_dict[key] = 1 - client_dict[key] if invert else client_dict[key]
            else:
                server_dict[key] += (
                    (1 - client_dict[key]) if invert else client_dict[key]
                )
    return server_dict


def add_server_weights(
        server_weights: OrderedDict[str, torch.Tensor],
        client_weights: OrderedDict[str, torch.Tensor],
        client_mask: OrderedDict[str, torch.Tensor],
        invert: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """将客户端权重按 mask 累加到服务器权重中。

    参数:
        server_weights: 服务器权重累加器
        client_weights: 客户端模型权重
        client_mask: 二值 mask，指示参数是否被加入
        invert: 是否在应用 mask 前取反

    返回:
        更新后的服务器权重累加器
    """
    for key in client_weights.keys():
        if "weight" in key or "bias" in key:
            mask = 1 - client_mask[key] if invert else client_mask[key]
            if key not in server_weights.keys():
                server_weights[key] = client_weights[key] * mask
            else:
                server_weights[key] += client_weights[key] * mask
    return server_weights
