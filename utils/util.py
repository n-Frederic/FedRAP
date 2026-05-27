import copy

import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score, recall_score, precision_score, confusion_matrix
from torch.utils.data import Subset, DataLoader

from models.Fed import FedAVG


def analyze_client_importance(net_glob, dw_locals, idxs_users, dataset_test, args):
    """
    探究客户端对全局模型 F1-Score 的重要性贡献
    """
    importance_report = {}

    # 准备完整测试集（计算 F1 必须包含负样本以衡量 Precision）
    # 如果全量测试太慢，可以随机采样一个平衡后的验证集
    test_loader = DataLoader(dataset_test, batch_size=args.bs, shuffle=False)

    def get_f1_score(net):
        net.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels, _ in test_loader:
                imgs = imgs.to(args.device)
                labels = labels.to(args.device)
                _, outputs = net(imgs)
                # 获取预测类别 (0 或 1)
                preds = outputs.data.max(1, keepdim=True)[1].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # 计算 F1，由于是反欺诈，重点关注类 1
        return f1_score(all_labels, all_preds, zero_division=0)

    # A. 基准：全量聚合后的 F1-Score
    # 假设 FedAVG 返回的是权重增量后的 state_dict
    w_glob_all = FedAVG(copy.deepcopy(net_glob.state_dict()), dw_locals, [1] * len(dw_locals))
    net_glob_copy = copy.deepcopy(net_glob)
    net_glob_copy.load_state_dict(w_glob_all)
    base_f1 = get_f1_score(net_glob_copy)

    # 获取最后几层的 Key 名（通常是 classifier.weight 或 fc.weight）
    last_layer_key = [k for k in w_glob_all.keys() if 'weight' in k][-1]

    print(f"\n--- 客户端重要性深度诊断 (Base F1-Score: {base_f1:.4f}) ---")

    for i, idx in enumerate(idxs_users):
        # 1. 特征空间分析 (Last Layer dw)
        # 考察最后全连接层对第 1 类（欺诈类）权重的更新趋势
        dw_last = dw_locals[i][last_layer_key][1].cpu().numpy()
        feature_impact = np.sum(dw_last)

        # 2. 留一法验证 (LOO) - 核心：计算剔除该客户端后 F1 的变化
        dw_minus_i = [dw for j, dw in enumerate(dw_locals) if i != j]
        # 如果当前轮次只有一个客户端（极端情况），LOO 无意义，给个 0
        if len(dw_minus_i) > 0:
            w_glob_minus_i = FedAVG(copy.deepcopy(net_glob.state_dict()), dw_minus_i, [1] * len(dw_minus_i))
            net_temp = copy.deepcopy(net_glob)
            net_temp.load_state_dict(w_glob_minus_i)
            f1_minus_i = get_f1_score(net_temp)
            loo_importance = base_f1 - f1_minus_i  # 正值表示该客户端提升了 F1
        else:
            loo_importance = 0.0

        importance_report[idx] = {
            'feature_impact': feature_impact,
            'loo_importance': loo_importance,
            'dw_norm': torch.norm(torch.cat([v.flatten() for v in dw_locals[i].values()])).item()
        }

        # 打印诊断：重点关注 loo_importance 为负的客户端（这些就是拖后腿的噪声源）
        print(
            f"Client {idx:2d} | 决策层贡献: {feature_impact:+.6f} | LOO重要性(F1): {loo_importance:+.4f} | 梯度范数: {importance_report[idx]['dw_norm']:.4f}")

    return importance_report


def visualize_gradients(dw_locals, w_glob, w_glob_prev, idxs_users, round_idx):
    """
    dw_locals: 参与本轮训练的客户端梯度列表 (字典格式)
    w_glob: 本轮聚合后的全局参数
    w_glob_prev: 上一轮的全局参数
    """
    # 1. 展平客户端梯度
    local_vecs = []
    for dw in dw_locals:
        # 只取 weight 参与降维，避免偏置项干扰，且速度更快
        vec = torch.cat([v.flatten() for k, v in dw.items() if 'weight' in k]).cpu().detach().numpy()
        local_vecs.append(vec)

    # 2. 计算实际的全局更新方向 (w_new - w_old)
    global_dw = {}
    for k in w_glob.keys():
        if 'weight' in k:
            global_dw[k] = w_glob[k].cpu() - w_glob_prev[k].cpu()

    global_vec = torch.cat([v.flatten() for v in global_dw.values()]).numpy()

    # 3. PCA 降维
    all_vecs = np.array(local_vecs + [global_vec])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_vecs)

    # 4. 绘图
    plt.figure(figsize=(10, 7))
    plt.scatter(coords[:-1, 0], coords[:-1, 1], c='blue', label='Client Updates', alpha=0.5)
    for i, txt in enumerate(idxs_users):
        plt.annotate(f"C{txt}", (coords[i, 0], coords[i, 1]), fontsize=8)

    plt.scatter(coords[-1, 0], coords[-1, 1], c='red', marker='X', s=200, label='Global Aggregated')

    plt.title(f"Gradient Space Visualization - Round {round_idx}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'./log/test1/grad_vis_round_{round_idx}.png')
    plt.close()


def monitor_perf(net, dataset_test, user_groups_test, idx, args, phase, actual_ratio):
    """
    phase: "BEFORE" 或 "AFTER"
    actual_ratio: 当前真实的个性化比例
    """
    client_test_subset = Subset(dataset_test, list(user_groups_test[idx]))
    test_loader = DataLoader(client_test_subset, batch_size=args.bs, shuffle=False)

    net.eval()
    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for imgs, labels,_ in test_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device).long()
            _,outputs = net(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.data.max(1, keepdim=True)[1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.flatten().cpu().numpy())

    # 计算5个核心指标
    prauc = average_precision_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    pre = precision_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    tn = cm[0, 0] if cm.shape == (2, 2) else 0
    fp = cm[0, 1] if cm.shape == (2, 2) else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    gmean = (rec * tnr) ** 0.5

    print(f"| Client: {idx:2d} | Ratio: {actual_ratio:>5.1f}% | {phase:6s} | PRAUC: {prauc:.4f} | Rec: {rec:.4f} | G-M: {gmean:.4f} | F1: {f1:.4f} | Pre: {pre:.4f} |")