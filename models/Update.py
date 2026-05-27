#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from sklearn.metrics import precision_score, recall_score, average_precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import random
from sklearn import metrics
import copy
import torch.nn.functional as F

from models.Nets import MaskedLinear
from pflopt.optimizers import MaskLocalAltSGD
from utils.dataprocess import DatasetBalance, DatasetFromDataframe
from models.test import test_model


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, out_round=0, client_id=0, net_pre=None):
        self.args = args
        self.client_id = client_id
        self.net_pre = net_pre
        self.out_round = out_round
        self.loss_func = nn.CrossEntropyLoss()
        # self.selected_clients = []
        self.dataset = dataset.data.loc[idxs]
        self.dataset_size = len(idxs)  # 客户端的数据集大小
        self.dataset_0count = dataset.data.loc[idxs].Class.value_counts().get(0, 0)
        self.dataset_1count = dataset.data.loc[idxs].Class.value_counts().get(1, 0)
        self.dataset_1Amount = sum(
            self.dataset[self.dataset['Class'] == 1].Amount) if 'Amount' in self.dataset.columns else 0.0
        self.eps = 1e-9
        self.last_perf_local = {}
        # dataset.data.loc[idxs].loc[:,'Class']
        # self.ldr_train = DataLoader(DatasetBalance(dataset.data.loc[idxs]), batch_size=self.args.local_bs, shuffle=True)  #对数据做平衡处理之后装载

        # 修改 : 增加分支选择：是否拆分数据集
        client_full_data = self.dataset
        self.client_raw_dataset = DatasetFromDataframe(client_full_data.reset_index(drop=True))
        self.best_threshold = 0.5

        if getattr(self.args, 'split_val', False) and self.args.risk_type == 4 and self.args.fed_type == 3:
            # --- 分支 ：严谨模式（划分验证集） ---
            try:
                train_df, val_df = train_test_split(
                    client_full_data, test_size=0.2,
                    stratify=client_full_data['Class'], random_state=42
                )
            except:
                train_df, val_df = train_test_split(client_full_data, test_size=0.2, random_state=42)
            train_df = train_df.reset_index(drop=True)  # 修改:重置索引，避免遍历时出错
            val_df = val_df.reset_index(drop=True)

            df_pos = val_df[val_df['Class'] == 1]
            print(
                f"Client {client_id} validate -- num:{len(df_pos)} sum:{df_pos['Amount'].sum() if 'Amount' in val_df.columns else 0.0}")
            self.ldr_train = DataLoader(DatasetBalance(train_df, args=args), batch_size=self.args.local_bs,
                                        shuffle=True)
            self.ldr_val = DataLoader(DatasetFromDataframe(val_df), batch_size=self.args.local_bs, shuffle=False)
            self.eval_data_name = "Local Validation Set"
        else:
            # --- 分支 ：原始模式（完全一致，不划分） ---
            # 直接使用全部数据进行平衡训练
            self.ldr_train = DataLoader(DatasetBalance(client_full_data, args=args), batch_size=self.args.local_bs,
                                        shuffle=True)
            # 验证时也直接用这个平衡后的数据
            self.ldr_val = self.ldr_train
            self.eval_data_name = "Balanced Training Set (Original Mode)"

    # 修改：增加mask参数
    def train(self, net, net_global, mask=None):
        # 先备份当轮次下发的模型
        net_glob = copy.deepcopy(net_global)
        net_glob.eval()
        # 开始进行训练
        net.train()
        wt0 = copy.deepcopy(net.state_dict())  # 备份本轮训练前的模型参数
        epoch_loss = []
        epoch_accuracy = []
        # train and update
        # 修改 ： 修改mask对应的optimizer+训练过程
        if getattr(self.args, 'local_type', 0) == 1 and mask is not None:
            # 使用支持 Mask 的个性化交替优化器
            optimizer = MaskLocalAltSGD(net.parameters(), mask, lr=self.args.lr)
            for iter in range(self.args.local_ep):
                # 第1次更新: 更新一组参数 (Toggle=True)
                loss1, acc1 = self._run_step_logic(net, net_glob, optimizer)
                optimizer.toggle()

                # 第2次更新: 更新另一组参数 (Toggle=False)
                loss2, acc2 = self._run_step_logic(net, net_glob, optimizer)
                optimizer.toggle()  # 还原状态供下个 Epoch 使用

                epoch_loss.append((loss1 + loss2) / 2)
                epoch_accuracy.append((acc1 + acc2) / 2)
                print(f'Client {self.client_id} [FedSelect Mode] Epoch {iter} Loss: {epoch_loss[-1]:.6f}')
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
            for iter in range(self.args.local_ep):
                # 调用完全一致的步进逻辑（单次遍历数据）
                loss, acc = self._run_step_logic(net, net_glob, optimizer)
                epoch_loss.append(loss)
                epoch_accuracy.append(acc)
                print('Client {} [Original Mode] Local Epoch: {} \tLoss: {:.6f}'.format(
                    self.client_id, iter, loss))

        # for iter in range(self.args.local_ep): #对于本地训练的每一轮
        #     batch_loss = []
        #     correct = 0
        #     for batch_idx, (images, labels) in enumerate(self.ldr_train): #对于本地训练的每一批次，默认每批次样本量为50
        #         images, labels = images.to(self.args.device), labels.to(self.args.device)
        #         net.zero_grad()
        #         code,log_probs = net(images)
        #         loss = self.loss_func(log_probs, labels)
        #         if self.args.fed_type == 4: #FedProx
        #             reg_loss = 0.0
        #             cnt = 0
        #             for name, param in net.named_parameters():
        #                 prox_term = F.smooth_l1_loss(
        #                     param, net_glob.state_dict()[name]
        #                 )
        #                 reg_loss +=  0.01 * prox_term #前面的系数mu，暂时先手动设置一下
        #                 cnt += 1
        #             reg_loss = reg_loss / cnt
        #             loss += reg_loss  #给损失增加一个近似项
        #         if self.args.fed_type == 5:  # Moon
        #             code_glb, _ = net_glob(images)
        #             code_pre, _ = self.net_pre(images)
        #             cs = nn.CosineSimilarity(dim=-1)
        #             sims0 = cs(code, code_glb)
        #             sims1 = cs(code, code_pre)
        #             sims = 2.0 * torch.stack([sims0, sims1], dim=1)  #前面的系数T，暂时先手动设置一下
        #             labels_code = torch.LongTensor([1] * code.shape[0])
        #             labels_code = labels_code.to(code.device)  #这句话用来做什么
        #             criterion = nn.CrossEntropyLoss()
        #             ct_loss = criterion(sims, labels_code)
        #             loss = loss + 0.001 * ct_loss #前面的系数mu，暂时先手动设置一下
        #
        #         loss.backward()
        #         optimizer.step()
        #         # if self.args.verbose and batch_idx % 10 == 0:
        #         #     print('Local Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
        #         #                100. * batch_idx / len(self.ldr_train), loss.item()))
        #         batch_loss.append(loss.item())
        #         #计算本地训练准确率
        #         y_pred = log_probs.data.max(1, keepdim=True)[1]  # 哪个位置的数大，就代表预测的是谁
        #         y_label = labels.data.view_as(y_pred)
        #         correct += y_pred.eq(y_label).long().cpu().sum()
        #     print('Client {} Local Epoch: {} \tLoss: {:.6f}'.format(self.client_id, iter,
        #                                                             sum(batch_loss) / len(batch_loss)))
        #     epoch_loss.append(sum(batch_loss)/len(batch_loss))
        #     epoch_accuracy.append(correct/len(self.ldr_train.dataset))
        #     # torch.save(net.state_dict(), f'./local-model-mlp120-{self.client_id}.pt')

        # 修改 : 在验证集上计算 Precision
        net.eval()
        y_true, y_pred, y_probs = [], [], []
        sample_residuals = []  # 新增：记录预测残差

        with torch.no_grad():
            for images, labels, _ in self.ldr_val:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                _, log_probs = net(images)

                y_preds_value = F.softmax(log_probs, dim=1)[:, 1]
                if self.args.dataset == 'creditcard':
                    preds = log_probs.data.max(1, keepdim=True)[1]
                elif self.args.dataset == 'IEEE-CIS':
                    preds = log_probs.data.max(1, keepdim=True)[1]
                elif self.args.dataset == 'paysim':
                    preds = log_probs.data.max(1, keepdim=True)[1]

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy() if preds.dim() > 0 else [preds.item()])
                y_probs.extend(torch.softmax(log_probs, dim=1)[:, 1].cpu().numpy())

                # 获取真实标签对应的预测概率，计算残差 (1 - P(y_true))
                probs = torch.softmax(log_probs, dim=1)
                true_class_probs = probs.gather(1, labels.view(-1, 1)).squeeze(-1)
                residuals = 1.0 - true_class_probs
                sample_residuals.extend(residuals.cpu().numpy())

        local_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        local_prauc = average_precision_score(y_true, y_probs)
        local_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)

        best_threshold, best_threshold_f1 = self._find_best_threshold(y_true, y_probs)
        self.best_threshold = best_threshold

        if self.args.opt_f1:
            local_f1 = self._compute_optimal_f1(y_true, y_probs)
        else:
            local_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        self.last_perf_local = {
            'recall': local_recall,
            'f1': local_f1
        }

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        recall_val = tp / (tp + fn + 1e-9)
        fpr_val = fp / (fp + tn + 1e-9)

        if self.args.measure:
            score_method = getattr(self.args, 'score_method', 0)

            y_true_np = np.array(y_true)
            val_pos = int(np.sum(y_true_np == 1))

            # 验证集中没有欺诈样本，score 直接置 0
            if getattr(self.args, 'score_zero_no_pos', True) and val_pos == 0:
                final_score = 0.0
                print(
                    f'Client {self.client_id} | no fraud samples on {self.eval_data_name} | '
                    f'score:{final_score:.4f}'
                )

            elif score_method == 0:
                # ===== score_method 0：保留原始 cost measure =====
                # 注意：这里只在 method0 内部对 raw_score 做函数缩放，不再统一 clip final_score
                r = float(np.clip(local_f1, 0.05, 0.95))
                c_pos = 1.0 + self.args.beta_fm ** 2 - r
                c_neg = r

                total_cost = c_pos * (1.0 - recall_val) + c_neg * fpr_val
                raw_score = 1.0 / (total_cost + 1e-6)

                mode = getattr(self.args, 'score_mode', 'power')
                alpha = getattr(self.args, 'score_alpha', 1.0)

                if mode == 'power':
                    # alpha=1.0：完全保留旧逻辑
                    # alpha=0.5：sqrt 压缩
                    final_score = raw_score ** alpha

                elif mode == 'log':
                    # 压缩极端大值，但不强行 clip
                    final_score = np.log1p(alpha * raw_score)

                elif mode == 'sigmoid':
                    # 映射到 (0,1)，适合防止极端 score 主导聚合
                    center = getattr(self.args, 'score_sigmoid_center', 1.0)
                    final_score = 1.0 / (1.0 + np.exp(-alpha * (raw_score - center)))

                elif mode == 'tanh':
                    # 也可以限制到接近 (0,1)，但不是 clip
                    final_score = np.tanh(alpha * raw_score)

                else:
                    raise ValueError(f"Unknown score_mode: {mode}")

                print(
                    f'Client {self.client_id} | METHOD0-cost | '
                    f'local_f:{local_f1:.4f} recall:{recall_val:.4f} fpr:{fpr_val:.4f} | '
                    f'c_pos:{c_pos:.3f} c_neg:{c_neg:.3f} | '
                    f'total_cost:{total_cost:.6f} raw_score:{raw_score:.4f} | '
                    f'mode:{mode} score:{final_score:.4f}'
                )

            elif score_method == 1:
                # ===== score_method 3:level-set / cost-sensitive reduction 风格 =====
                # 核心思想：
                # 对每个阈值产生一个 error profile；
                # 再扫描 r，判断当前 error profile 能达到的最大 F-measure level。
                eps = 1e-9
                beta2 = getattr(self.args, 'beta_fm', 1.0) ** 2

                y_probs_np = np.array(y_probs)

                # 阈值候选
                if len(np.unique(y_probs_np)) > 1:
                    grid_t = getattr(self.args, 'score_grid_T', 40)
                    low_pct = getattr(self.args, 'score_thresh_low_pct', 5)
                    high_pct = getattr(self.args, 'score_thresh_high_pct', 95)
                    thresh_candidates = np.unique(
                        np.percentile(y_probs_np, np.linspace(low_pct, high_pct, grid_t))
                    )
                else:
                    thresh_candidates = [0.5]

                # F-measure level 候选
                r_grid_t = getattr(self.args, 'score_r_grid_T', 101)
                r_candidates = np.linspace(0.0, 1.0, r_grid_t)

                best_r = 0.0
                best_margin = -1e18

                for thresh in thresh_candidates:
                    y_pred_t = (y_probs_np >= thresh).astype(int)
                    tn_t, fp_t, fn_t, tp_t = confusion_matrix(
                        y_true_np, y_pred_t, labels=[0, 1]
                    ).ravel()

                    total_t = tp_t + fp_t + fn_t + tn_t
                    if total_t <= 0:
                        continue

                    P1 = (tp_t + fn_t) / (total_t + eps)
                    e1 = fn_t / (total_t + eps)
                    e2 = fp_t / (total_t + eps)

                    # 如果该阈值下没有正类边际，也跳过
                    if P1 <= 0:
                        continue

                    for r in r_candidates:
                        # 论文 Eq.(5)：
                        # Fβ(e) <= r 可转化为：
                        # (1+β²-r)e1 + r e2 + (1+β²)P1(r-1) >= 0
                        #
                        # 所以 Fβ(e) >= r 等价于：
                        # baseline - cost >= 0
                        cost = (1.0 + beta2 - r) * e1 + r * e2
                        baseline = (1.0 + beta2) * P1 * (1.0 - r)
                        margin = baseline - cost

                        if margin >= -1e-12:
                            if r > best_r:
                                best_r = float(r)
                                best_margin = float(margin)

                final_score = best_r

                print(
                    f'Client {self.client_id} | METHOD3-levelset | '
                    f'pos:{val_pos} best_r:{best_r:.4f} margin:{best_margin:.6f} | '
                    f'score:{final_score:.4f}'
                )

            else:
                raise ValueError(f"Unknown score_method: {score_method}")

        else:
            # measure=False 时，直接把 local_f1 作为聚合 score
            # 如果 opt_f1=True，则这里的 local_f1 已经是阈值扫描 paper-Fβ
            final_score = local_f1

        # 修改 ：跑完后删除， 打印信息增加标识目前是在哪个分支
        print(f'Client {self.client_id} F1 on {self.eval_data_name}: {local_f1:.4f}')

        # 计算本轮训练后本地模型参数更新量
        dw = net.state_dict()
        for k in wt0.keys():
            dw[k] = dw[k] - wt0[k]
        # 风险数值
        if self.args.risk_type == 1:
            s = self.dataset_size
        elif self.args.risk_type == 2:  # P+N/M
            s = self.dataset_1count + (self.dataset_0count / 576)
        elif self.args.risk_type == 3:  # （P+N/M）*平均欺诈金额
            s = self.dataset_1Amount  # (self.dataset_1count + (self.dataset_0count/576)) * (self.dataset_1Amount/self.dataset_1count)
        else:
            s = 1

        # 返回本地训练的网络的所有参数的更新量，及本次训练的平均损失，以及准确率、风险数值、本次训练后的模型
        return dw, sum(epoch_loss) / len(epoch_loss), sum(epoch_accuracy) / len(epoch_accuracy), final_score, s, net

    # 修改 ：新增函数整合核心步进逻辑
    def _run_step_logic(self, net, net_glob, optimizer):
        net.train()
        batch_loss = []
        correct = 0
        for batch_idx, (images, labels, amounts) in enumerate(self.ldr_train):
            images, labels, amounts = images.to(self.args.device), labels.to(self.args.device), amounts.to(
                self.args.device)
            net.zero_grad()
            code, log_probs = net(images)
            loss = None
            if self.args.loss == 2:
                loss = self.focal_loss(log_probs, labels)
            elif self.args.loss == 1:
                loss = self.loss_func(log_probs, labels)
            elif self.args.loss == 3:
                loss = self.symmetric_poly_loss(log_probs, labels)
            elif self.args.loss == 4:
                loss = self.enhanced_spd_loss(log_probs, labels)

            if self.args.fed_type == 4:
                fed_prox_reg = 0.0
                mu = getattr(self.args, 'mu', 0.01)
                global_weight_collector = list(net_glob.parameters())
                for param, global_param in zip(net.parameters(), global_weight_collector):
                    # 使用标准 L2 范数: ||w - w_t||^2
                    fed_prox_reg += (mu / 2) * torch.norm((param - global_param)) ** 2
                loss += fed_prox_reg

            if self.args.fed_type == 5:
                mu_moon = getattr(self.args, 'mu_moon', 1.0)
                temperature = 0.5

                with torch.no_grad():
                    code_glob, _ = net_glob(images)
                    code_prev, _ = self.net_pre(images)

                cs = nn.CosineSimilarity(dim=-1)
                sim_pos = cs(code, code_glob).reshape(-1, 1)
                sim_neg = cs(code, code_prev).reshape(-1, 1)

                logits_con = torch.cat([sim_pos, sim_neg], dim=1) / temperature
                labels_con = torch.zeros(images.size(0)).long().to(self.args.device)

                loss_con = nn.CrossEntropyLoss()(logits_con, labels_con)
                loss += mu_moon * loss_con

            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

        return sum(batch_loss) / len(batch_loss), correct / len(self.ldr_train.dataset)

    def focal_loss(self, logits, labels, alpha=0.25, gamma=2.0):
        """
        Standard Focal Loss
        :param logits: 模型输出 (batch_size, num_classes)
        :param labels: 真实标签 (batch_size)
        :param alpha: 类别平衡因子（通常正类设为 0.25，负类为 0.75）
        :param gamma: 难易样本聚焦参数（通常设为 2.0）
        """
        ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        at = torch.where(labels == 1, alpha, 1 - alpha)
        loss = at * (1 - pt) ** gamma * ce_loss

        return loss.mean()

    def symmetric_poly_loss(self, logits, labels, alpha=0.3, k=5):
        """
        Symmetric Polynomial-Decay Loss (SPD-Loss)
        不再使用对数函数，利用高阶多项式实现对称撤力。
        :param k: 阶数。k越大，p > 0.5 后的撤力越猛。建议取 3 或 4。
        """
        p = torch.softmax(logits, dim=1)[:, 1]
        pt = torch.where(labels == 1, p, 1 - p)
        at = torch.where(labels == 1, alpha, 1 - alpha)

        # 公式：L = at * (1 - pt)^k * (1 + k*pt)
        # 它的导数（梯度）类似于：-k*(k+1) * pt * (1-pt)^(k-1)
        # 这保证了在 pt=0 和 pt=1 时损失都平滑
        loss = at * torch.pow(1 - pt, k) * (1 + k * pt)

        return loss.mean()

    # EUM 扫描阈值计算最佳F-measure
    def _compute_optimal_f1(self, y_true, y_probs):
        y_true = np.array(y_true)
        y_probs = np.array(y_probs)

        eps = 1e-9
        beta2 = getattr(self.args, 'beta_fm', 1.0) ** 2

        if len(y_true) == 0:
            return 0.0

        if getattr(self.args, 'score_zero_no_pos', True) and np.sum(y_true == 1) == 0:
            return 0.0

        if len(np.unique(y_probs)) > 1:
            grid_t = getattr(self.args, 'score_grid_T', 40)
            low_pct = getattr(self.args, 'score_thresh_low_pct', 5)
            high_pct = getattr(self.args, 'score_thresh_high_pct', 95)
            candidates = np.unique(
                np.percentile(y_probs, np.linspace(low_pct, high_pct, grid_t))
            )
        else:
            candidates = np.array([0.5])

        best_f = 0.0

        for thresh in candidates:
            y_pred_t = (y_probs >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t, labels=[0, 1]).ravel()

            total = tp + fp + fn + tn
            if total <= 0:
                continue

            P1 = (tp + fn) / (total + eps)
            e1 = fn / (total + eps)
            e2 = fp / (total + eps)

            paper_f = ((1.0 + beta2) * (P1 - e1)) / (
                    (1.0 + beta2) * P1 - e1 + e2 + eps
            )

            if paper_f > best_f:
                best_f = paper_f

        return float(best_f)

    def evaluate_diagnostics(self, net_local, net_global):
        """
        三合一深度诊断：Oracle (效能), Fisher (敏感度), Sim (对齐度)
        """
        net_local.eval()
        net_global.eval()

        # --- 维度 1: Validation Oracle (效能诊断) ---
        # 目标：直接通过考试分数决定谁该当老师
        perf_local = getattr(self, 'last_perf_local', None)
        if perf_local is None:
            perf_local = self._quick_evaluate(net_local)
        perf_global = self._quick_evaluate(net_global)

        # 性能差距指数 (Gap). 如果 Local 远好于 Global，该值趋近 1
        # 我们重点关注 Recall，因为这是欺诈检测的命门
        f1_gap = perf_local['f1'] - perf_global['f1']
        oracle_trust = torch.sigmoid(torch.tensor(f1_gap * 10.0)).item()

        # --- 维度 2: Fisher Information (敏感度诊断) ---
        # 目标：找出哪些参数是本地数据的“命根子”
        fisher_dict = {}
        net_local.train()  # 必须在 train 模式下计算梯度
        for name, param in net_local.named_parameters():
            fisher_dict[name] = torch.zeros_like(param.data)

        # 只跑一个小批次来估算 Fisher，节省计算成本
        for batch_idx, (images, labels, _) in enumerate(self.ldr_val):
            if batch_idx > 3: break  # 采样前 3 个 batch 足够了
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            net_local.zero_grad()
            _, outputs = net_local(images)
            loss = self.symmetric_poly_loss(outputs, labels)
            loss.backward()

            for name, param in net_local.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += (param.grad.data ** 2) / 3.0  # 平方梯度累计

        # --- 维度 3: Feature Similarity (特征对齐诊断) ---
        # 目标：检查全局和本地在中间层表达上是否冲突
        # 这里简化为 Embedding 层（最后一层前的输出）的余弦相似度
        feature_sim = 0
        with torch.no_grad():
            for images, _, _ in self.ldr_val:
                images = images.to(self.args.device)
                feat_local, _ = net_local(images)
                feat_global, _ = net_global(images)
                feature_sim += F.cosine_similarity(feat_local, feat_global).mean().item()
                break  # 采样一次即可

        return {
            'oracle_trust': oracle_trust,
            'fisher_importance': fisher_dict,
            'feature_sim': feature_sim,
            'perf_local': perf_local,
            'perf_global': perf_global
        }

    def _quick_evaluate(self, net):
        """ 内部快速评估函数 """
        net.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels, _ in self.ldr_val:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                _, outputs = net(images)
                preds = outputs.data.max(1, keepdim=True)[1].cpu().numpy()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.flatten())

        return {
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

    def train_rema(self, net, net_global, global_protos=None):
        """
        修正后的 FedReMa 本地训练逻辑：
        移除所有交替冻结和 MSE 对齐，回归标准 SGD。原型仅用于提取上传。
        """
        net_glob = copy.deepcopy(net_global)
        net_glob.eval()
        wt0 = copy.deepcopy(net.state_dict())

        all_step_losses = []

        # 1. 正常的联合训练 (无冻结，无分阶段)
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        criterion_ce = torch.nn.CrossEntropyLoss()

        for epoch in range(self.args.local_ep):
            batch_losses = []
            for batch_idx, (images, labels, _) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()

                # 直接过前向传播
                _, log_probs = net(images)

                # 只有基础交叉熵损失！绝对不要加 MSE 原型对齐！
                loss = criterion_ce(log_probs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimizer.step()
                batch_losses.append(loss.item())

            epoch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
            all_step_losses.append(epoch_loss)
            print(f'Client {self.client_id} [FedReMa] Epoch {epoch} Loss: {epoch_loss:.6f}')

        # 2. 收集本地原型 (FedReMa 特有：训练结束后，用最新的模型提取特征原型，传给 Server)
        local_protos = self.collect_local_protos(net)
        avg_loss = sum(all_step_losses) / len(all_step_losses) if all_step_losses else 0

        # 3. 本地验证与指标记录 (保持你原有的评估逻辑不变)
        net.eval()
        y_true, y_pred, y_probs = [], [], []
        with torch.no_grad():
            for images, labels, _ in self.ldr_val:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                _, log_probs = net(images)
                preds = log_probs.data.max(1, keepdim=True)[1].squeeze(-1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy() if preds.dim() > 0 else [preds.item()])
                y_probs.extend(torch.softmax(log_probs, dim=1)[:, 1].cpu().numpy())

        local_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        best_threshold, best_threshold_f1 = self._find_best_threshold(y_true, y_probs)
        self.best_threshold = best_threshold

        if self.args.opt_f1:
            local_f1 = self._compute_optimal_f1(y_true, y_probs)
        else:
            local_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        self.last_perf_local = {
            'recall': local_recall,
            'f1': local_f1
        }

        # 4. 计算参数更新 dw
        dw = net.state_dict()
        for k in wt0.keys():
            dw[k] = dw[k] - wt0[k]

        # 注意：请把你在 _run_rema_step 里的旧代码删掉，不再需要它了
        return dw, avg_loss, local_protos, net

    def collect_local_protos(self, net):
        """
        计算本地各个类别的特征均值
        """
        net.eval()
        protos_dict = {}  # 使用 dict: {label: [features]}

        with torch.no_grad():
            for images, labels, _ in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                code, _ = net(images)

                for i, label in enumerate(labels):
                    l = label.item()
                    if l not in protos_dict:
                        protos_dict[l] = []
                    protos_dict[l].append(code[i].detach().cpu())

        # 求平均
        final_protos = {}
        for l, feature_list in protos_dict.items():
            if len(feature_list) > 0:
                final_protos[l] = torch.stack(feature_list).mean(dim=0)

        return final_protos

    # 在 LocalUpdate 类中添加或修改
    def train_fedas(self, net_local, net_glob, is_selected):
        # 源代码逻辑：1. 对齐全局模型 -> 2. 本地模型拉取全局模型 -> 3. 本地训练 -> 4. 算FIM

        # === 步骤 1: 原型对齐 (让全局模型 net_glob 适应本地特征分布) ===
        net_local.eval()
        local_prototypes = [[] for _ in range(self.args.num_classes)]
        ldr_proto = DataLoader(self.client_raw_dataset, batch_size=16, shuffle=False)

        with torch.no_grad():
            for x, y, *args in ldr_proto:
                x, y = x.to(self.args.device), y.to(self.args.device)
                code, _ = net_local(x)  # 提取特征
                for proto, label in zip(code, y):
                    local_prototypes[label.item()].append(proto)

        mean_protos = []
        for protos in local_prototypes:
            mean_protos.append(torch.stack(protos).mean(0) if protos else None)

        # 对齐优化器只作用于全局模型的 Base (layer_input)
        alignment_optimizer = torch.optim.SGD(net_glob.layer_input.parameters(), lr=0.01)
        net_glob.train()
        for _ in range(1):
            for x, y, *args in ldr_proto:
                x, y = x.to(self.args.device), y.to(self.args.device)
                code_glob, _ = net_glob(x)
                loss_alg = 0
                for label in y.unique():
                    if mean_protos[label.item()] is not None:
                        idx = (y == label)
                        target = mean_protos[label.item()].unsqueeze(0).expand(code_glob[idx].size(0), -1)
                        loss_alg += 0.1*F.mse_loss(code_glob[idx], target)
                if torch.is_tensor(loss_alg) and loss_alg > 0:
                    alignment_optimizer.zero_grad()
                    loss_alg.backward()
                    alignment_optimizer.step()

        # === 步骤 2: 同步 (核心修正点：与源代码 set_parameters 对齐) ===
        local_dict = net_local.state_dict()
        global_dict = net_glob.state_dict()

        for k in global_dict.keys():
            if "layer_input" in k:
                local_dict[k] = global_dict[k].clone()

        net_local.load_state_dict(local_dict)

        epoch_loss = []
        epoch_accuracy = []
        # === 步骤 3: 本地训练 (仅选中时) ===
        # for name, param in net_local.named_parameters():
        #     if "layer_input" in name:
        #         param.requires_grad = False

        if is_selected:
            net_local.train()
            # for name, module in net_local.named_modules():
            #     if "layer_input" in name:
            #         module.eval()
            trainable_params = filter(lambda p: p.requires_grad, net_local.parameters())

            optimizer = torch.optim.SGD(
                trainable_params,
                lr=self.args.lr,
                momentum=self.args.momentum
            )
            for epoch in range(self.args.local_ep):
                # 这里调用你原本针对 creditcard 的核心训练逻辑 (如 Focal Loss)
                loss, acc = self._run_step_logic(net_local, net_glob, optimizer)
                epoch_loss.append(loss)
                epoch_accuracy.append(acc)
                print(f'Client {self.client_id} [FedAs Mode] Local Epoch: {epoch} \tLoss: {loss:.6f}')
        else:
            epoch_loss = [0.0]
            epoch_accuracy = [0.0]

        # === 步骤 4: 计算 FIM (对应源代码训练后的 fim_trace_sum) ===
        for param in net_local.layer_input.parameters():
            param.requires_grad = True

        net_local.eval()
        fim_trace_sum = 0
        for x, y, *args in ldr_proto:
            x, y = x.to(self.args.device), y.to(self.args.device)
            _, output = net_local(x)
            nll = -F.log_softmax(output, dim=1)[range(len(y)), y].mean()
            grads = torch.autograd.grad(nll, net_local.layer_input.parameters())
            for g in grads:
                fim_trace_sum += torch.sum(g ** 2).detach()

        y_true, y_pred, y_probs = [], [], []
        with torch.no_grad():
            for images, labels, _ in self.ldr_val:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                _, log_probs = net_local(images)
                preds = log_probs.data.max(1, keepdim=True)[1].squeeze(-1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy() if preds.dim() > 0 else [preds.item()])
                y_probs.extend(torch.softmax(log_probs, dim=1)[:, 1].cpu().numpy())

        local_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        best_threshold, best_threshold_f1 = self._find_best_threshold(y_true, y_probs)
        self.best_threshold = best_threshold

        if self.args.opt_f1:
            local_f1 = self._compute_optimal_f1(y_true, y_probs)
        else:
            local_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        # 执行赋值
        self.last_perf_local = {
            'recall': local_recall,
            'f1': local_f1
        }

        return net_local.state_dict(), fim_trace_sum.item()

    def compute_fim_only(self, net_local):

        net_local.eval()

        ldr_proto = DataLoader(
            self.client_raw_dataset,
            batch_size=64,
            shuffle=False
        )

        fim_trace_sum = 0

        for x, y, *args in ldr_proto:
            x, y = x.to(self.args.device), y.to(self.args.device)

            _, output = net_local(x)

            nll = -F.log_softmax(output, dim=1)[range(len(y)), y].mean()

            grads = torch.autograd.grad(
                nll,
                net_local.layer_input.parameters()
            )

            for g in grads:
                fim_trace_sum += torch.sum(g ** 2).detach()

        return fim_trace_sum.item()

    def train_fedpm(self, net, net_global):
        global_weight_dict = {n: p.clone().detach() for n, p in net_global.named_parameters() if
                              'weight' in n}

        net.train()
        trainable_params = []
        # 只过滤出 score 参数进行优化
        with torch.no_grad():
            for n, p in net.named_parameters():
                if n in global_weight_dict:
                    p.copy_(global_weight_dict[n])
                    p.requires_grad = False  # 严格禁止更新 weight
                if 'score' in n:
                    p.requires_grad = True
                    trainable_params.append(p)
                elif 'bias' in n:
                    p.requires_grad = True
                    trainable_params.append(p)

        optimizer = torch.optim.SGD(trainable_params, lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        epoch_accuracy = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            correct = 0
            for batch_idx, (images, labels,_) in enumerate(self.ldr_train):
                for module in net.modules():
                    if isinstance(module, MaskedLinear):
                        module.sampled_mask = None

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                _, logit = net(images)  # 你的 MLP 返回 (code, y)
                loss = F.cross_entropy(logit, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                batch_loss.append(loss.item())
                y_pred = logit.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

            avg_loss = sum(batch_loss) / len(batch_loss)
            avg_acc = correct / len(self.ldr_train.dataset)
            epoch_loss.append(avg_loss)
            epoch_accuracy.append(avg_acc)

            print(f'Client {self.client_id} [FedPM Mode] Local Epoch: {iter} \tLoss: {avg_loss:.6f}')

        curr_masks = {}
        with torch.no_grad():
            for name, param in net.named_parameters():
                if 'score' in name:
                    probs = torch.sigmoid(param)
                    curr_masks[name] = torch.bernoulli(probs)

        net.eval()
        y_true, y_pred, y_probs = [], [], []
        with torch.no_grad():
            for images, labels, _ in self.ldr_val:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                _, log_probs = net(images)
                preds = log_probs.data.max(1, keepdim=True)[1].squeeze(-1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy() if preds.dim() > 0 else [preds.item()])
                y_probs.extend(torch.softmax(log_probs, dim=1)[:, 1].cpu().numpy())

        local_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        local_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        local_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        best_threshold, best_threshold_f1 = self._find_best_threshold(y_true, y_probs)
        self.best_threshold = best_threshold

        positive_ratio = np.mean(np.array(y_pred) == 1)

        print(
            f'[VAL] Client {self.client_id} | '
            f'pos_pred_ratio:{positive_ratio:.6f} | '
            f'recall:{local_recall:.4f} | '
            f'precision:{local_precision:.4f} | '
            f'f1:{local_f1:.4f}'
        )

        # 更新性能记录
        self.last_perf_local = {'recall': local_recall, 'f1': local_f1}

        return curr_masks, sum(epoch_loss) / len(epoch_loss), sum(epoch_accuracy) / len(
            epoch_accuracy), net

    def _find_best_threshold(self, y_true, y_probs):
        y_true = np.array(y_true)
        y_probs = np.array(y_probs)

        best_f1 = -1
        best_threshold = 0.5

        # 阈值扫描
        thresholds = np.linspace(0.01, 0.99, 99)

        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)

            f1 = f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        return float(best_threshold), float(best_f1)
