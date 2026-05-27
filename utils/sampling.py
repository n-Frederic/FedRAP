#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def creditcard_iid(dataset, num_users):
    """
    Sample I.I.D. client data from creditcard dataset
    :param dataset:
    :param num_users:
    :return: dict of sample index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, dataset.data.index  # [i for i in range(len(dataset))]
    for i in range(num_users - 1):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    dict_users[i + 1] = set(all_idxs)
    return dict_users


def creditcard_noniid(dataset, num_users, type, list_ratio=None, pre_scaler=None, pre_kmeans=None, pca=None, seed=68):
    """
    Sample non-I.I.D client data from creditcard dataset
    :param pca: pca分析
    :param dataset: 数据集
    :param num_users: 划分的用户数量，即划分为多少份
    :param type: 划分类别，1-按照样本数量划分，2-按照欺诈样本数量划分，3-按照欺诈金额划分
    :param list_ratio: 划分比例
    :return: 划分后每个客户端分配的样本index集合
    """
    if type not in (1, 2, 3, 4, 5):
        raise Exception("creditcard_noniid type param error")

    all_idxs = dataset.data.index
    dict_users = {}
    dict_users0 = {}
    dict_users1 = {}

    if type == 1:  # 按数据集大小比例进行分割
        for i in range(len(list_ratio) - 1):
            num_items = int(len(dataset) * list_ratio[i] / sum(list_ratio))
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i + 1] = set(all_idxs)
    elif type == 2:  # 按欺诈样本比例进行分割
        # if num_users != 3 or len(list_ratio) != 3:
        #     raise Exception("creditcard_noniid scale param error")
        data = dataset.data
        data0 = data[data.Class == 0]
        data1 = data[data.Class == 1]
        idxs_0 = data0.index
        idxs_1 = data1.index
        for i in range(len(list_ratio) - 1):  # 先划分欺诈样本
            num_items1 = int(len(data1) * list_ratio[i] / sum(list_ratio))
            dict_users1[i] = set(np.random.choice(idxs_1, num_items1, replace=False))
            idxs_1 = list(set(idxs_1) - dict_users1[i])
        # dict_users1[i + 1] = set(idxs_1)
        dict_users1[num_users - 1] = set(idxs_1)  # 更安全
        for i in range(len(list_ratio) - 1):  # 再划分正常样本
            num_items0 = int(len(data) / num_users - len(dict_users1[i]))
            dict_users0[i] = set(np.random.choice(idxs_0, num_items0, replace=False))
            idxs_0 = list(set(idxs_0) - dict_users0[i])
        # dict_users0[i + 1] = set(idxs_0)
        dict_users0[num_users - 1] = set(idxs_0)
        dict_users = [set(list(dict_users0[i]) + list(dict_users1[i])) for i in range(num_users)]
    elif type == 3:  # 按欺诈金额进行分割
        data = dataset.data
        data0 = data[data.Class == 0]
        data1 = data[data.Class == 1]

        # 计算总金额和目标分配金额
        all_amount = data1['Amount'].sum()
        list_amount0 = [n / sum(list_ratio) * all_amount for n in list_ratio]

        # 改为降序排列！先分配大额，后用小额填缝
        data1 = data1.sort_values(by='Amount', ascending=False)

        dict_users1 = {i: [] for i in range(num_users)}
        current_amounts = {i: 0.0 for i in range(num_users)}  # 记录实际已分配金额

        # 使用贪心策略一次性遍历
        for idx, row in data1.iterrows():
            amt = row['Amount']
            # 找到当前距离目标金额缺口最大的节点
            best_node = max(range(num_users), key=lambda x: list_amount0[x] - current_amounts[x])

            # 将该笔交易分配给最缺钱的节点
            dict_users1[best_node].append(idx)
            current_amounts[best_node] += amt

        # 获取实际分配的最终结果
        list_amount1 = [current_amounts[i] for i in range(num_users)]
        print("欺诈金额目标比例：", list_ratio)
        print("实际分配到每个节点的欺诈金额:", list_amount1)

        # 正常样本划分
        dict_users0 = {i: [] for i in range(num_users)}
        sample_num = int(len(data) / num_users)
        data0_indexs = data0.index.tolist()  # 转为 list 以支持 remove 或集合操作

    elif type == 4:
        print("正在进行基于 K-Means 的特征聚类划分...")
        data = dataset.data
        drop_list = ['Class', 'Amount', 'TransactionAmt']
        actual_drop = [c for c in drop_list if c in data.columns]
        features = data.drop(actual_drop, axis=1)

        # 根据阶段不同进行训练或预测
        if pre_scaler is None or pre_kmeans is None:
            # 训练集阶段：拟合 (fit)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            kmeans = KMeans(n_clusters=num_users, random_state=seed, n_init=10)
            clusters = kmeans.fit_predict(features_scaled)
            is_train = True
        else:
            # 测试集阶段：直接使用传进来的模型预测 (predict)
            scaler = pre_scaler
            features_scaled = scaler.transform(features)
            kmeans = pre_kmeans
            clusters = kmeans.predict(features_scaled)
            is_train = False

        cluster_dict = {i: [] for i in range(num_users)}
        for idx, cluster_id in enumerate(clusters):
            # 注意：idx 是 features_scaled 的行索引，需要映射回 data 的真实 index
            real_idx = all_idxs[idx]
            cluster_dict[cluster_id].append(real_idx)

        for i in range(num_users):
            cluster_idxs = cluster_dict[i]
            fraud_count = data.loc[cluster_idxs, 'Class'].sum()
            print(f"客户端 {i}: 总样本={len(cluster_idxs)}, 欺诈样本={fraud_count}, "
                  f"欺诈率={fraud_count / len(cluster_idxs):.4%}")

        dict_users = [set(cluster_dict[i]) for i in range(num_users)]

        if is_train:
            return dict_users, scaler, kmeans, None
        else:
            return dict_users
    elif type == 5:
        print("正在进行基于 PCA后的K-Means 的特征聚类划分...")
        data = dataset.data
        drop_list = ['Class', 'Amount', 'TransactionAmt']
        actual_drop = [c for c in drop_list if c in data.columns]
        features = data.drop(actual_drop, axis=1)
        real_clusters = 4

        # 根据阶段不同进行训练或预测
        if pre_scaler is None or pre_kmeans is None:
            # 训练集阶段：拟合 (fit)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            pca = PCA(n_components=50)
            features_scaled = pca.fit_transform(features_scaled)
            print(
                f"PCA: {np.sum(pca.explained_variance_ratio_)}"
            )
            kmeans = KMeans(n_clusters=real_clusters, random_state=68, n_init=10)
            clusters = kmeans.fit_predict(features_scaled)
            is_train = True
        else:
            # 测试集阶段：直接使用传进来的模型预测 (predict)
            scaler = pre_scaler
            features_scaled = scaler.transform(features)
            features_scaled = pca.transform(features_scaled)
            kmeans = pre_kmeans
            clusters = kmeans.predict(features_scaled)
            is_train = False

        cluster_dict = {i: [] for i in range(real_clusters)}
        for idx, cluster_id in enumerate(clusters):
            # 注意：idx 是 features_scaled 的行索引，需要映射回 data 的真实 index
            real_idx = all_idxs[idx]
            cluster_dict[cluster_id].append(real_idx)

        dict_users = [set() for _ in range(num_users)]
        clients_per_cluster = num_users // real_clusters

        for cluster_id in range(real_clusters):
            cluster_idxs = cluster_dict[cluster_id]
            np.random.shuffle(cluster_idxs)

            split_idxs = np.array_split(
                cluster_idxs,
                clients_per_cluster
            )

            for j, part in enumerate(split_idxs):
                client_id = cluster_id * clients_per_cluster + j

                dict_users[client_id] = set(part.tolist())

        for i in range(num_users):
            client_idxs = list(dict_users[i])
            fraud_count = data.loc[client_idxs, 'Class'].sum()

            ratio = (
                fraud_count / len(client_idxs)
                if len(client_idxs) > 0 else 0
            )

            print(
                f"客户端 {i}: "
                f"总样本={len(client_idxs)}, "
                f"欺诈样本={fraud_count}, "
                f"欺诈率={ratio:.4%}"
            )

        if is_train:
            return dict_users, scaler, kmeans, pca
        else:
            return dict_users
    else:
        raise Exception("creditcard数据集non-iid分割，type参数error")
    return dict_users


def get_partition_file_path(args, dataset_name, is_train=True):
    """生成唯一的文件名，确保不同配置不冲突"""
    phase = "train" if is_train else "test"
    # 文件名包含：数据集名、划分类型、用户数、阶段
    filename = (
        f"{dataset_name}_"
        f"type{args.split_dataset_type}_"
        f"u{args.num_users}_"
        f"seed{args.seed}_"
        f"{phase}.pkl"
    )
    # 路径：./data/banksim/banksim_type5_u20_train.pkl
    folder = os.path.join('./data', dataset_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return os.path.join(folder, filename)


def save_partition(dict_users, scaler, model, pca, args, dataset_name, is_train):
    path = get_partition_file_path(args, dataset_name, is_train)

    with open(path, 'wb') as f:
        pickle.dump(
            (dict_users, scaler, model, pca),
            f
        )

    print(f"成功保存划分结果至: {path}")


def load_partition(args, dataset_name, is_train):
    path = get_partition_file_path(args, dataset_name, is_train)

    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        print(f"检测到已有划分文件，成功载入: {path}")

        return data

    return None