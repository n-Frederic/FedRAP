# 在 models/Fed.py 中添加
import copy


def FedAS_Aggregate(w_glob, w_locals, fim_traces):

    new_w = copy.deepcopy(w_glob)

    total_fim = sum(fim_traces) + 1e-9

    for k in new_w.keys():

        if "layer_input" in k:

            new_w[k].zero_()

            for i in range(len(w_locals)):
                weight = fim_traces[i] / total_fim
                new_w[k] += w_locals[i][k] * weight

    return new_w