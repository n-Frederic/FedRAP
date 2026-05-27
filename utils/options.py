#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    args = parser.parse_args()
    return args

class Args():
    def __init__(self):
        # federated arguments
        self.epochs = 200  #rounds of training
        self.num_users = 20  #number of users: K
        self.frac = 0.4  #the fraction of clients: C
        self.local_ep = 2  #the number of local epochs: E
        self.local_bs = 128  #local batch size: B
        self.bs = 128  #test batch size
        self.lr = 0.01  #learning rate
        self.momentum = 0.5 #SGD momentum (default: 0.5)
        # self.split = 'user'  # train-test split type, user or sample

        # model arguments
        self.model = 'mlp'  # model name

        # other arguments
        self.dataset = 'creditcard'  # name of dataset
        # self.dataset = 'IEEE-CIS'  # name of dataset

        self.iid = False  # whether i.i.d or not
        self.num_classes = 2  # number of classes
        # self.num_channels = 1  # number of channels of imges
        self.gpu = 0  # GPU ID, -1 for CPU
        # self.stopping_rounds = 10  # rounds of early stopping
        self.verbose = True  # verbose print
        self.seed = 1  # random seed (default: 1)
        self.all_clients = False  # aggregation over all clients
        self.local_type = 0

        # new added arguments
        self.split_dataset_type = 4 # 1-根据数据集大小划分，2-根据欺诈样本比例划分(此时各节点数据集大小相同)，3-根据欺诈金额比例划分(此时各节点数据集大小相同)
        self.split_dataset_ratio = [1,2,2,3,3,4,4,1,2,3,1,2,2,3,3,4,3,1,2,3]  #划分数据集时的数据比例
        self.fed_type = 3 #联邦聚合方式  1-fedavg，2-fedmean，3-ours, 4-fedprox, 5-fedmoon， 6-fedselect. 7-fedrema, 8-fedas, 9-fedpm

        self.risk_type = 4 #风险权值类型   1-数据集大小，2-数据集欺诈样本数量，3-数据集欺诈样本金额 4-ours
        self.split_val = True #使用客户端模型性能作为服务器聚合指标时，是否使用验证集进行验证类型

        self.batch_size = 128
        self.lth_iters = 40
        self.prune_percent = 10
        self.prune_target = 80
        self.gamma=0.4#动态掩码上限的惩罚系数
        self.loss = 1           # 1 交叉熵  2 focal_loss 3 sploss 4 esploss

        self.opt_f1 = False#寻找最优F1值
        self.score_mode="power"
        self.score_alpha=1

        self.mask_reverse_when_bad = True

        # arguments for fedrema
        self.training_type = 3
        self.lamda = 1.0
        self.delta = 0.5

        # dataprocess相关参数
        self.dataprocess = True
        self.dp_small_pos_threshold = 10
        self.dp_low_ratio = 0.05
        self.dp_expand_factor = 3.0
        self.dp_normal_ratio = 0.10
        self.dp_smote_max_k = 5

        #measure相关参数
        self.measure = True
        self.score_method = 1
        self.beta_fm = 1
        self.score_grid_T = 40
        self.score_r_grid_T = 101
        self.score_thresh_low_pct = 5
        self.score_thresh_high_pct = 95
        self.score_zero_no_pos = True

        #fusion相关参数
        self.fusion_mode = 1
        self.alpha = 5

