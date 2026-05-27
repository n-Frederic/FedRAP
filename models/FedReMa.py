import torch
import numpy as np
import copy
import torch.nn.functional as F


def aggregate_protos(local_protos_list):
    agg_protos = {}
    counts = {}
    for local_p in local_protos_list:
        if local_p is None: continue
        for label, proto in local_p.items():
            if label not in agg_protos:
                agg_protos[label] = torch.zeros_like(proto)
                counts[label] = 0
            agg_protos[label] += proto
            counts[label] += 1
    for label in agg_protos.keys():
        if counts[label] > 0:
            agg_protos[label] = agg_protos[label] / counts[label]
    return agg_protos


# 修改后的 FedReMa 聚合逻辑
def aggregate_rema_heads(client_state_dicts, idxs_users, global_protos, args, net_glob, is_clp, stable_neighbors=None):
    new_state_dicts = copy.deepcopy(client_state_dicts)
    user_list = list(idxs_users)
    num_selected = len(user_list)

    # 1. 构造关系向量 (保持你的实现，但建议加入温度)
    labels = sorted(global_protos.keys())
    temp_net = copy.deepcopy(net_glob)
    relation_vectors = []
    tem = 0.5  # 引入温度系数

    for idx in user_list:
        temp_net.load_state_dict(client_state_dicts[idx])
        temp_net.eval()
        client_probs = []
        with torch.no_grad():
            for label in labels:
                proto = global_protos[label].to(args.device).unsqueeze(0)
                logits = temp_net.layer_hidden(proto)
                # 使用温度系数使概率分布更尖锐
                probs = F.softmax(logits / tem, dim=1).view(-1)
                client_probs.append(probs)
        relation_vectors.append(torch.cat(client_probs))

    relation_vectors = torch.stack(relation_vectors)
    norm_vecs = F.normalize(relation_vectors, p=2, dim=1)
    sim_matrix = torch.mm(norm_vecs, norm_vecs.t()).cpu().numpy()

    current_max_diffs = []
    round_neighbors = {}

    # 3. 聚合逻辑
    for i in range(num_selected):
        curr_user = user_list[i]

        if is_clp:
            sim_vec = sim_matrix[i]
            sort_idx = np.argsort(sim_vec)
            sorted_sim = sim_vec[sort_idx]
            sim_diff = np.diff(sorted_sim)

            if len(sim_diff) > 0:
                # 仅在后半段找最大跳跃点
                max_diff_idx = np.argmax(sim_diff)
                current_max_diffs.append(sim_diff[max_diff_idx])
                neighbor_indices = sort_idx[max_diff_idx + 1:]
            else:
                neighbor_indices = [i]

            if i not in neighbor_indices:
                neighbor_indices = np.append(neighbor_indices, i)
            round_neighbors[curr_user] = neighbor_indices
        else:
            # 如果 CLP 结束，使用传入的稳定邻居关系
            neighbor_indices = stable_neighbors.get(curr_user, [i])

        # 4. 执行聚合
        # 注意：不要直接改 new_state_dicts 的所有客户端，仅计算当前选中的
        target_weight = torch.zeros_like(client_state_dicts[curr_user]['layer_hidden.weight'])
        target_bias = torch.zeros_like(client_state_dicts[curr_user]['layer_hidden.bias'])

        for nb_local_idx in neighbor_indices:
            nb_user = user_list[nb_local_idx]
            target_weight += client_state_dicts[nb_user]['layer_hidden.weight']
            target_bias += client_state_dicts[nb_user]['layer_hidden.bias']

        new_state_dicts[curr_user]['layer_hidden.weight'] = target_weight / len(neighbor_indices)
        new_state_dicts[curr_user]['layer_hidden.bias'] = target_bias / len(neighbor_indices)

    avg_max_diff = np.mean(current_max_diffs) if current_max_diffs else 0
    return new_state_dicts, avg_max_diff, round_neighbors
