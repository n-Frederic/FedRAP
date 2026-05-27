#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random

import torch
import matplotlib
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import recall_score, f1_score, average_precision_score, precision_score, confusion_matrix, \
    roc_auc_score, accuracy_score

from broadcast import div_server_weights, broadcast_server_to_client_initialization, add_masks, add_server_weights
from lottery_ticket import init_mask_zeros, delta_update
from models.FedAS import FedAS_Aggregate
from models.FedReMa import aggregate_protos, aggregate_rema_heads
from utils.util import analyze_client_importance, visualize_gradients, monitor_perf

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import pandas as pd
from utils.sampling import creditcard_iid, creditcard_noniid, load_partition, save_partition
from utils.options import args_parser, Args
from utils.dataprocess import DatasetFromCSV
from models.Update import LocalUpdate
from models.Nets import mlp, MLP
from models.Fed import FedAVG, FedMEAN, FedRWA
from models.test import test_model

if __name__ == '__main__':
    # parse args
    # args = args_parser() #使用命令行参数方式
    args = Args()  # 使用配置方式
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 固定 random 种子
    random.seed(args.seed)
    # 固定 Numpy 随机库种子
    np.random.seed(args.seed)
    # 固定 PyTorch CPU 随机种子
    torch.manual_seed(args.seed)
    # 固定 PyTorch GPU 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load dataset and split users
    dataset = getattr(args, 'dataset', 'creditcard')
    data_path = None
    if dataset == 'creditcard':
        data_path = './data/creditcard/creditcard.csv'
    elif dataset == 'IEEE-CIS':
        data_path = './data/IEEE-CIS/ieee_cis_processed_subset300000.csv'
    elif dataset == 'paysim':
        data_path = './data/paysim/paysim_processed_subset.csv'

    dataset_train = DatasetFromCSV(data_path, train=True, dataset_name=dataset)
    dataset_test = DatasetFromCSV(data_path, train=False, dataset_name=dataset)

    # sample users
    if args.iid:
        dict_users = creditcard_iid(dataset_train, args.num_users)
        dict_users_test = creditcard_iid(dataset_test, args.num_users)
    else:
        # 分别进行聚类训练+划分和应用聚类模型进行划分
        dict_users, fitted_scaler, fitted_kmeans, pca = creditcard_noniid(
            dataset_train, args.num_users, type=args.split_dataset_type, list_ratio=args.split_dataset_ratio, seed=args.seed
        )

        dict_users_test = creditcard_noniid(
            dataset_test, args.num_users, type=args.split_dataset_type, list_ratio=args.split_dataset_ratio,
            pre_scaler=fitted_scaler, pre_kmeans=fitted_kmeans, pca=pca
        )

    # print("--- 客户端样本分布检查 ---")
    # empty_clients = []
    # for i in range(args.num_users):
    #     train_size = len(dict_users[i])
    #     test_size = len(dict_users_test[i])
    #     print(f"Client {i:2d}: Train={train_size:6d}, Test={test_size:6d}")
    #     if train_size == 0 or test_size == 0:
    #         empty_clients.append(i)
    #
    # if empty_clients:
    #     print(f"警告：以下客户端没有测试数据: {empty_clients}")
    # else:
    #     print("所有客户端均分配到了数据。")
    # print("--------------------------")

    for ui in range(len(dict_users)):
        user_data = dataset_train.data.loc[list(dict_users[ui])]
        print("第{}个客户端: 总样本数量为{}，欺诈样本数量为{}".format(ui, len(user_data), len(
            user_data[user_data.Class == 1])))

    # build model
    len_in = dataset_train.input_dim
    if dataset == 'creditcard' or dataset == 'paysim':
        net_glob = MLP(dim_in=len_in, dim_out=args.num_classes).to(args.device)
    elif dataset == 'IEEE-CIS':
        net_glob = mlp(dim_in=len_in, dim_out=args.num_classes).to(args.device)
    # net_glob = mlp(dim_in=len_in, dim_out=args.num_classes).to(args.device)

    global_scores = None
    if args.fed_type == 9:
        from models.Nets import convert_to_fedpm

        net_glob = convert_to_fedpm(net_glob).to(args.device)
        global_scores = {n: p.clone().detach() for n, p in net_glob.named_parameters() if 'score' in n}

    print(net_glob)
    net_glob.train()

    # 加载一个固定的初始模型
    # net_glob.load_state_dict(torch.load('./fed_model-mlp120.pt'))
    net_locals = []  # 把初始模型给所有客户端下发一下，以备后续Moon方法使用
    for userid in range(args.num_users):
        net_locals.append(copy.deepcopy(net_glob))

    # copy weights
    w_glob = net_glob.state_dict()
    client_state_prev = {i: copy.deepcopy(net_glob.state_dict()) for i in range(args.num_users)}  # 上一轮次客户端的个性化模型权重
    client_state_dicts = {i: copy.deepcopy(net_glob.state_dict()) for i in range(args.num_users)}  # 当前客户端的个性化模型权重

    list_loss_train = []
    best_threshold = [0.5 for i in range(args.num_users)]

    if args.fed_type < 6:
        last_round_scores = {i: 0.0 for i in range(args.num_users)}
    elif args.fed_type == 6:
        client_masks = {i: init_mask_zeros(net_glob) for i in range(args.num_users)}  # 当前客户端的参数掩码 Mask
        client_masks_prev = {i: init_mask_zeros(net_glob) for i in range(args.num_users)}  # 上一轮次客户端的参数掩码 Mask
        server_accumulate_mask = {}
        server_weights = {}
    elif args.fed_type == 7:
        global_protos = {}
        history_max_diff = []
        is_clp = True
        stable_neighbors = {}
        delta_threshold = 0.5
    elif args.fed_type == 8:
        current_fim_traces = {i: 1.0 for i in range(args.num_users)}

    trainset_val_result = pd.DataFrame(
        columns=['loss', 'accuracy', 'accuracy1', 'precision', 'recall', 'recall_ma', 'tnr', 'fpr', 'gmean', 'ma_f1',
                 'mi_f1', 'f1', 'auc', 'ks', 'prauc', 'auprc', 'cm00', 'cm01', 'cm10', 'cm11', 'amount1',
                 'amount1_pred'])
    testset_val_result = pd.DataFrame(
        columns=['loss', 'accuracy', 'accuracy1', 'precision', 'recall', 'recall_ma', 'tnr', 'fpr', 'gmean', 'ma_f1',
                 'mi_f1', 'f1', 'auc', 'ks', 'prauc', 'auprc', 'cm00', 'cm01', 'cm10', 'cm11', 'amount1',
                 'amount1_pred'])

    for iter in range(args.epochs):  # 进行联合模型的每个轮次迭代
        loss_locals = []
        dw_locals = []
        datanum_locals = []  # 本轮训练的各客户端数据集大小
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        client_diagnostics = {}

        if args.fed_type == 3:
            risk_scores = []
        elif args.fed_type == 7:
            local_protos_list = []
        elif args.fed_type == 8:
            selected_fims = []

        for idx in idxs_users:  # 针对每个客户端执行本地模型训练
            is_selected = (idx in idxs_users)
            # 本地训练
            client_state_prev[idx] = copy.deepcopy(client_state_dicts[idx])
            net_local = copy.deepcopy(net_glob)
            net_local.load_state_dict(client_state_dicts[idx])

            if args.fed_type == 6:
                client_mask = client_masks_prev.get(idx)
            if args.fed_type == 9:
                with torch.no_grad():
                    for (n_g, p_g), (n_l, p_l) in zip(net_glob.named_parameters(), net_local.named_parameters()):
                        if 'weight' in n_g:
                            p_l.copy_(p_g.data)
                            p_l.requires_grad = False

                        if 'score' in n_g:
                            p_l.copy_(global_scores[n_g])
                            p_l.requires_grad = True

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=list(dict_users[idx]), out_round=iter,
                                client_id=idx, net_pre=net_locals[idx])
            print('main_fed: Round {:3d}, client{:3d} start train:'.format(iter, idx))

            if args.fed_type < 6:
                dw, loss, a, score, s, net = local.train(net=net_local.to(args.device), net_global=net_glob)
                last_round_scores[idx] = score
            elif args.fed_type == 6:
                dw, loss, a, score, s, net = local.train(net=net_local.to(args.device), net_global=net_glob,
                                                         mask=client_mask)
            elif args.fed_type == 7:
                dw, loss, local_p, net = local.train_rema(net_local, net_glob, global_protos)
                local_protos_list.append(local_p)
            elif args.fed_type == 8:
                updated_w, fim_val = local.train_fedas(net_locals[idx], copy.deepcopy(net_glob), is_selected=True)

                current_fim_traces[idx] = fim_val
                selected_fims.append(fim_val)
                net_locals[idx].load_state_dict(updated_w)
                client_state_dicts[idx] = copy.deepcopy(updated_w)

                dw = copy.deepcopy(updated_w)
                loss = 0.0
                net = net_locals[idx]
            elif args.fed_type == 9:
                dw, loss, a, net = local.train_fedpm(net=net_local.to(args.device), net_global=net_glob)
            else:
                raise Exception("we don't have this fed type")

            best_threshold[idx] = local.best_threshold
            diag = local.evaluate_diagnostics(net, net_global=net_glob)
            client_diagnostics[idx] = diag
            client_state_dicts[idx] = copy.deepcopy(net.state_dict())

            # 处理训练后数据
            dw_locals.append(copy.deepcopy(dw))  # 收集客户端上传的参数更新
            loss_locals.append(copy.deepcopy(loss))  # 收集客户端上传的本轮训练的损失
            # s_locals.append(s)  # 收集客户端上传的风险数值
            datanum_locals.append(len(dict_users[idx]))
            # client_param_updates[idx] = copy.deepcopy(dw)  # 收集客户端的参数更新量

            if args.fed_type == 3:
                risk_type = args.risk_type
                client_risk = None
                user_data = dataset_train.data.loc[list(dict_users[idx])]
                if risk_type == 1:
                    client_risk = len(user_data)
                elif risk_type == 2:
                    client_risk = len(user_data[user_data.Class == 1])
                elif risk_type == 3:
                    client_risk = sum(user_data[user_data.Class == 1].Amount)
                elif risk_type == 4:
                    client_risk = score
                risk_scores.append(client_risk)  # 修改：将当前客户端的分数追加到列表中
                print(
                    f'[RiskScore] Round {iter} Client {idx} | '
                    f'score:{client_risk:.6f} | '
                    f'loss:{loss:.6f} | '
                    f'data:{len(dict_users[idx])}'
                )
            elif args.fed_type == 6:
                server_accumulate_mask = add_masks(server_accumulate_mask, client_mask)
                server_weights = add_server_weights(
                    server_weights, net.state_dict(), client_mask
                )

                if iter % args.lth_iters == 0 and iter != 0:
                    client_mask, delta_tensor_dict = delta_update(
                        args.prune_percent / 100,
                        client_state_dicts[idx],
                        client_state_prev[idx],
                        client_masks_prev[idx],
                        bound=args.prune_target / 100,
                        invert=True,
                    )
                    client_state_prev[idx] = copy.deepcopy(client_state_dicts[idx])
                    client_masks_prev[idx] = copy.deepcopy(client_mask)
                    client_masks[idx] = copy.deepcopy(client_mask)

            # 保存客户端训练后的最新个性化状态
            net_locals[idx] = copy.deepcopy(net)

            # current_mask = client_masks.get(idx)
            # if args.fed_type == 9:
            #     total_elements = sum(v.numel() for v in dw.values())
            #     active_elements = sum(torch.sum(v).item() for v in dw.values())
            #     actual_ratio = (active_elements / total_elements) * 100 if total_elements > 0 else 0.0
            # else:
            #     if current_mask is None or len(current_mask) == 0 or 'weight' not in list(current_mask.keys())[0]:
            #         actual_ratio = 100.0  # 初始全量保留
            #     else:
            #         total_params = sum(v.numel() for k, v in current_mask.items() if 'weight' in k)
            #         ones_count = sum(torch.count_nonzero(v).item() for k, v in current_mask.items() if 'weight' in k)
            #         actual_ratio = (ones_count / total_params) * 100 if total_params > 0 else 100.0
            # monitor_perf(net, dataset_test, dict_users_test, idx, args, "BEFORE", actual_ratio)

        if args.fed_type == 8:
            print(f"main_fed: Round {iter}, FedAS 补充更新未选中客户端的 Fisher 信息...")
            for idx in range(args.num_users):
                if idx not in idxs_users:
                    local_silent = LocalUpdate(args=args, dataset=dataset_train, idxs=list(dict_users[idx]),
                                               out_round=iter, client_id=idx, net_pre=net_locals[idx])
                    # 执行静默更新：仅 [原型对齐 -> 算FIM]，不进行 SGD 训练
                    fim_val_silent = local_silent.compute_fim_only(
                        net_locals[idx]
                    )
                    # 仅更新 Fisher 追踪器和本地模型状态（对齐后的状态）
                    current_fim_traces[idx] = fim_val_silent
                    # net_locals[idx].load_state_dict(updated_w_silent)
                    # client_state_dicts[idx] = copy.deepcopy(updated_w_silent)

        w_glob_prev = copy.deepcopy(w_glob)

        # update global weights
        if args.fed_type in (1, 4, 5):
            # 第一种：联邦均值算法，根据样本数量进行平均
            w_glob = FedAVG(w_glob, dw_locals, datanum_locals)
        elif args.fed_type == 2:
            # 第二种：联邦平均算法，直接平均
            w_glob = FedMEAN(w_glob, dw_locals)
        elif args.fed_type == 3:
            risk_np = np.array(risk_scores)
            norm_risk = risk_np / (risk_np.sum() + 1e-12)
            w_glob = FedRWA(w_glob, dw_locals, risk_scores)
        elif args.fed_type == 6:
            w_glob = div_server_weights(server_weights, server_accumulate_mask)
            for i in range(args.num_users):
                client_state_dicts[i] = broadcast_server_to_client_initialization(
                    w_glob, client_masks[i], client_state_dicts[i],
                )
            server_accumulate_mask = {k: torch.zeros_like(v) for k, v in net_glob.state_dict().items()}
            server_weights = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in net_glob.state_dict().items()}
        elif args.fed_type == 7:
            global_protos = aggregate_protos(local_protos_list)
            w_glob = FedAVG(w_glob, dw_locals, datanum_locals)
            for i in range(args.num_users):
                for key in w_glob.keys():
                    if "layer_input" in key:
                        client_state_dicts[i][key] = copy.deepcopy(w_glob[key])
            client_state_dicts, avg_diff, round_neighbors = aggregate_rema_heads(
                client_state_dicts, idxs_users, global_protos, args, net_glob,
                is_clp=is_clp, stable_neighbors=stable_neighbors
            )

            if is_clp:
                history_max_diff.append(avg_diff)
                stable_neighbors.update(round_neighbors)

                max_v = max(history_max_diff)
                now_v = history_max_diff[-1]

                max_idx = np.argmax(history_max_diff)
                now_idx = len(history_max_diff) - 1

                if max_idx < now_idx:
                    if now_v / max_v < delta_threshold:
                        print(f"--- CLP Phase Finished at Round {iter} ---")
                        is_clp = False
        elif args.fed_type == 8:
            w_glob = FedAS_Aggregate(w_glob, dw_locals, selected_fims)
            net_glob.load_state_dict(w_glob)

            for i in range(args.num_users):
                for k in w_glob.keys():
                    if "layer_input" in k:
                        client_state_dicts[i][k] = copy.deepcopy(w_glob[k])
        elif args.fed_type == 9:
            from models.Fed import FedPM_Aggregate

            global_scores = FedPM_Aggregate(global_scores, dw_locals, lr_g=0.1)

            for n, p in net_glob.named_parameters():
                if 'score' in n:
                    p.data.copy_(global_scores[n])

            w_glob = net_glob.state_dict()

            with torch.no_grad():
                any_score_name = next(k for k in global_scores.keys() if 'score' in k)
                sparsity = (global_scores[any_score_name] >= 0).float().mean()
                print(f"Global Mask Sparsity (Keep Ratio): {sparsity:.4f}")
        else:
            exit("Error: unrecognized args.fed_type")
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        if args.fed_type < 6 and args.fusion_mode in (1, 2, 3, 4):
            fusion_mode = args.fusion_mode
            for i in range(args.num_users):
                is_selected = (i in idxs_users)
                for key in w_glob.keys():
                    if "weight" not in key and "bias" not in key:
                        client_state_dicts[i][key] = w_glob[key].to(args.device)
                        continue

                    g_new = w_glob[key].to(args.device)
                    l_old = client_state_dicts[i][key].to(args.device)

                    if is_selected:
                        diag = client_diagnostics[i]
                        if fusion_mode == 1:
                            # 基于变动幅度的信任融合 (Bayesian-like Fusion)
                            delta = torch.abs(l_old - client_state_prev[i][key].to(args.device))
                            trust_local = torch.sigmoid(delta * args.alpha)
                            client_state_dicts[i][key] = trust_local * l_old + (1 - trust_local) * g_new
                        elif fusion_mode == 2:
                            trust_local = diag['oracle_trust']
                            client_state_dicts[i][key] = trust_local * l_old + (1 - trust_local) * g_new
                        elif fusion_mode == 3:
                            f_imp = diag['fisher_importance'][key]
                            performance = diag['oracle_trust']
                            trust_local = performance * torch.sigmoid((f_imp - f_imp.mean()) / (f_imp.std() + 1e-9))
                            client_state_dicts[i][key] = trust_local * l_old + (1 - trust_local) * g_new
                        elif fusion_mode == 4:
                            f_imp = diag['fisher_importance'][key]
                            performance = diag['oracle_trust']

                            f_min, f_max = f_imp.min(), f_imp.max()
                            norm_f_imp = (f_imp - f_min) / (f_max - f_min + 1e-9)

                            trust_local = performance + (1 - performance) * norm_f_imp

                            client_state_dicts[i][key] = trust_local * l_old + (1 - trust_local) * g_new
                    else:
                        client_state_dicts[i][key] = 0.9 * l_old + 0.1 * g_new

        # 修改 ：增加本地全局融合机制
        if args.fed_type < 6 and args.fusion_mode == 0:
            for i in range(args.num_users):
                for key in w_glob.keys():
                    # if "layer_hidden" in key:
                    #     client_state_dicts[i][key] = local_param
                    #     continue
                    client_state_dicts[i][key] = w_glob[key].to(args.device)

        # for idx in idxs_users:
        #     net_after = copy.deepcopy(net_glob)
        #     net_after.load_state_dict(client_state_dicts[idx])
        #     current_mask = client_masks[idx]
        #     if args.fed_type == 9:
        #         # dw 此时就是 curr_masks 字典
        #         total_elements = sum(v.numel() for v in dw.values())
        #         active_elements = sum(torch.sum(v).item() for v in dw.values())
        #         actual_ratio = (active_elements / total_elements) * 100 if total_elements > 0 else 0.0
        #     else:
        #         if current_mask is None or len(current_mask) == 0 or 'weight' not in list(current_mask.keys())[0]:
        #             actual_ratio = 100.0
        #         else:
        #             total_params = sum(v.numel() for k, v in current_mask.items() if 'weight' in k)
        #             ones_count = sum(torch.count_nonzero(v).item() for k, v in current_mask.items() if 'weight' in k)
        #             actual_ratio = (ones_count / total_params) * 100 if total_params > 0 else 100.0
        #     monitor_perf(net_after, dataset_test, dict_users_test, idx, args, "AFTER ", actual_ratio)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)  # 所有客户端当轮次训练的损失的平均值
        print('main_fed: Round {:3d}, Clients train average loss {:.3f}'.format(iter, loss_avg))
        list_loss_train.append(loss_avg)

        # --- 修改后的测试部分：汇总所有客户端结果 ---
        print(f"main_fed: Round {iter} 联合训练完成，开始汇总各客户端个性化模型效果：")

        # 1. 准备汇总容器
        all_labels = []
        all_preds = []
        all_probs = []
        total_test_loss = 0
        total_amount1 = 0
        total_amount1_pred = 0
        total_samples = 0

        for idx in range(args.num_users):
            # 加载该客户端的个性化权重
            net_glob.load_state_dict(client_state_dicts[idx])
            net_glob.eval()

            user_test_idxs = list(dict_users_test[idx])
            # 调用我们修改过的 test_model
            result = test_model(net_glob, dataset_test, args, idxs=user_test_idxs)

            # 收集原始预测数据用于全局指标计算
            all_labels.append(result['y_labels'])
            all_preds.append(result['y_preds'])
            all_probs.append(result['y_probs'])

            # 累加损失（按样本量加权）
            total_test_loss += result['loss'] * len(user_test_idxs)
            # 累加金额
            total_amount1 += result['amount1']
            total_amount1_pred += result['amount1_pred']
            total_samples += len(user_test_idxs)

        # 2. 将所有客户端结果合并为大的 Numpy 数组
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)

        final_loss = total_test_loss / total_samples
        final_acc = accuracy_score(all_labels, all_preds)
        final_precision = precision_score(all_labels, all_preds, zero_division=0)
        final_recall = recall_score(all_labels, all_preds, zero_division=0)
        final_f1 = f1_score(all_labels, all_preds, zero_division=0)
        final_auc = roc_auc_score(all_labels, all_probs)
        final_prauc = average_precision_score(all_labels, all_probs)  # 这就是你要的第五个指标

        # 计算混淆矩阵用于特定率值
        cm = confusion_matrix(all_labels, all_preds)
        tnr = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        print(f"--- Round {iter} Diagnostic Report ---")
        print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        if (tp + fp) > 0:
            print(f"Positive Prediction Ratio: {(tp + fp) / len(all_labels):.4%}")
            print(f"Precision Breakdown: {tp} correct out of {tp + fp} predicted positive")
        else:
            print("WARNING: Model predicted ZERO positive cases!")

        # 4. 存入结果 DataFrame (对应你原本的列定义)
        testset_val_result.loc[iter] = [
            final_loss,  # loss
            final_acc * 100,  # accuracy (百分比)
            final_acc,  # accuracy1
            final_precision,  # precision
            final_recall,  # recall
            recall_score(all_labels, all_preds, average='macro'),  # recall_ma
            tnr,  # tnr
            fpr,  # fpr
            (final_recall * tnr) ** 0.5,  # gmean
            f1_score(all_labels, all_preds, average='macro'),  # ma_f1
            f1_score(all_labels, all_preds, average='micro'),  # mi_f1
            final_f1,  # f1
            final_auc,  # auc
            0,  # ks (若需计算: max(tpr-fpr))
            final_prauc,  # prauc (第五个核心指标)
            final_prauc,  # auprc (通常 PRAUC 即可视为 AUPRC)
            cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1],  # 混淆矩阵四项
            total_amount1,  # amount1 (总欺诈金额)
            total_amount1_pred  # amount1_pred (检出总金额)
        ]

        print(
            f"Round {iter} 汇总结果: Recall: {final_recall:.4f}, Precision: {final_precision:.4f}, AUC: {final_auc:.4f}，PRAUC：{final_prauc:.4f}，F1：{final_f1:.4f}")
        # testset_val_result.to_csv(
        #     './log/avg(paysim-local--round300)(1).csv')
