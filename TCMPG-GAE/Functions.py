import matplotlib.pyplot as plt
import numpy as np
import re
import os
import sys
import datetime
import torch


# def generate_prescription_emb(matrix_p_s, symptom_emb, symptom_size, index_array):
#     prescription_emb = torch.zeros((len(index_array), symptom_size))
#     for i in range(len(index_array)):
#         index = np.where(matrix_p_s[index_array[i]] == 1)
#         index = list(index[0])
#
#         prescription_emb[i] = torch.sum(symptom_emb[index], dim=0)
#
#     return prescription_emb

# def generate_prescription_emb(matrix_p_s, symptom_emb, symptom_size, ctx):
#     prescription_emb = nd.zeros((matrix_p_s.shape[0], symptom_size)).copyto(ctx)
#     for i in range(matrix_p_s.shape[0]):
#         index = np.where(matrix_p_s[i] == 1)
#         index = list(index[0])
#
#         prescription_emb[i] = nd.sum(symptom_emb[index], axis=0)
#
#     return prescription_emb


def get_herb_weight_matrix(sheet, herb_index):
    """
    :param samples_nums:
    :return: weight matries
    """
    herb_weight_matrix = torch.zeros(1603)
    herb_num = [0]*1603

    for i in range(sheet.max_row):
        herb_total = sheet.cell(i + 1, 3).value
        herb_total = herb_total.split(',')
        for h in herb_total:
            h = h.split(' ')
            herb = h[0].strip()  # type:str
            herb_num[herb_index[herb]] += 1

    max_num = max(herb_num)
    for i in range(1603):
        herb_weight_matrix[i] = max_num / herb_num[i]

    return herb_weight_matrix


def DCG(pred, true, k):
    indices = np.argsort(-pred)
    rel = true[indices][:k]
    # true = np.int64(true > 0)
    # start from 1
    dcg = (2 ** rel - 1) / np.log2(np.arange(1, k + 1) + 1)
    dcg = np.sum(dcg)
    return dcg


def NDCG(pred, true, k):
    dcg, idcg = DCG(pred, true, k), DCG(true, true, k)

    return dcg / idcg


def one_DCG(pred, label):
    # ------ cal DCG ------ #
    dcg = 0
    for i in range(len(pred)):
        r_i = 0
        if pred[i] in label:
            r_i = 1
        dcg += (2 ** r_i - 1) / np.log2((i + 1) + 1)
    return dcg


def one_IDCG(pred, label):

    A_temp_1 = []
    A_temp_0 = []
    for a in pred:
        if a in label:
            A_temp_1.append(a)
        else:
            A_temp_0.append(a)
    A_temp_1.extend(A_temp_0)
    idcg = one_DCG(A_temp_1, label)
    return idcg


def one_NDCG(pred, label):
    dcg = one_DCG(pred, label)
    idcg = one_IDCG(pred, label)
    if dcg == 0 or idcg == 0:
        ndcg = 0
    else:
        ndcg = dcg / idcg
    return ndcg


def Precision_Recall_topk(pred, label):
    indices_5 = torch.topk(pred, 5).indices.tolist()
    indices_10 = torch.topk(pred, 10).indices.tolist()
    indices_20 = torch.topk(pred, 20).indices.tolist()

    indices = np.where(label != 0)
    indices = list(indices[0])

    inter_5 = list(set(indices).intersection(set(indices_5)))
    inter_10 = list(set(indices).intersection(set(indices_10)))
    inter_20 = list(set(indices).intersection(set(indices_20)))

    prec_5 = len(inter_5) * 1.0 / 5
    prec_10 = len(inter_10) * 1.0 / 10
    prec_20 = len(inter_20) * 1.0 / 20

    reca_5 = len(inter_5) * 1.0 / len(indices)
    reca_10 = len(inter_10) * 1.0 / len(indices)
    reca_20 = len(inter_20) * 1.0 / len(indices)

    hit_5 = len(inter_5)*1.0
    hit_10 = len(inter_10)*1.0
    hit_20 = len(inter_20)*1.0
    all_num = len(indices)*1.0

    return prec_5, prec_10, prec_20, reca_5, reca_10, reca_20, hit_5, hit_10, hit_20, all_num


def entity_index(path):
    index_entity = {}
    with open(path, encoding='utf-8') as file_obj:
        for line in file_obj:
            line = line.strip()
            line = re.split(r"[ ]+", line)
            index_entity[int(line[0])] = line[1]
    return index_entity


def Precision_k_Curve(precs_5_result, precs_10_result, precs_20_result):
    fold = np.array([1, 2, 3, 4, 5])
    plt.figure(1, figsize=(7, 5))

    # plt.plot(fold, precs_5_result, 'o-', alpha=0.4, label='Mean P@5 = %.4f' % np.mean(precs_5_result))
    # plt.plot(fold, precs_10_result, 'o-', alpha=0.4, label='Mean P@10 = %.4f' % np.mean(precs_10_result))
    # plt.plot(fold, precs_20_result, 'o-', alpha=0.4, label='Mean P@20 = %.4f' % np.mean(precs_20_result))
    width = 0.3
    fontsize = 6
    plt.bar(fold - width, precs_5_result, width=width, alpha=0.4, label='Mean P@5  = %.4f' % np.mean(precs_5_result))
    for a, b in zip(fold - width, precs_5_result):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=fontsize)

    plt.bar(fold, precs_10_result, width=width, alpha=0.4, label='Mean P@10 = %.4f' % np.mean(precs_10_result))
    for a, b in zip(fold, precs_10_result):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=fontsize)

    plt.bar(fold + width, precs_20_result, width=width, alpha=0.4, label='Mean P@20 = %.4f' % np.mean(precs_20_result))
    for a, b in zip(fold + width, precs_20_result):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=fontsize)

    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Precision@K', fontsize=12)
    # plt.xticks(fold)
    plt.ylim([0.00, 0.45])

    plt.legend(loc="upper right")
    plt.savefig("./Experiment result/Precision_K_Curve.svg", dpi=300)
    plt.savefig("./Experiment result/Precision_K_Curve.png")


def Recall_k_Curve(recs_5_result, recs_10_result, recs_20_result):
    fold = np.array([1, 2, 3, 4, 5])
    plt.figure(2, figsize=(7, 5))

    # plt.plot(fold, recs_5_result, 'o-', alpha=0.4, label='Mean R@5 = %.4f' % np.mean(recs_5_result))
    # plt.plot(fold, recs_10_result, 'o-', alpha=0.4, label='Mean R@5 = %.4f' % np.mean(recs_10_result))
    # plt.plot(fold, recs_20_result, 'o-', alpha=0.4, label='Mean R@5 = %.4f' % np.mean(recs_20_result))
    width = 0.3
    fontsize = 6
    plt.bar(fold - width, recs_5_result, width=width, alpha=0.4, label='Mean R@5  = %.4f' % np.mean(recs_5_result))
    for a, b in zip(fold - width, recs_5_result):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=fontsize)

    plt.bar(fold, recs_10_result, width=width, alpha=0.4, label='Mean R@10 = %.4f' % np.mean(recs_10_result))
    for a, b in zip(fold, recs_10_result):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=fontsize)

    plt.bar(fold + width, recs_20_result, width=width, alpha=0.4, label='Mean R@20 = %.4f' % np.mean(recs_20_result))
    for a, b in zip(fold + width, recs_20_result):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=fontsize)

    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Recall@K', fontsize=12)
    # plt.xticks(fold)
    plt.ylim([0.00, 0.70])

    plt.legend(loc="upper right")
    plt.savefig("./Experiment result/Recall_K_Curve.svg", dpi=300)
    plt.savefig("./Experiment result/Recall_K_Curve.png")


def HR_k_Curve(hits_5_result, hits_10_result, hits_20_result):
    fold = np.array([1, 2, 3, 4, 5])
    plt.figure(3, figsize=(7, 5))

    # plt.plot(fold, hits_5_result, 'o-', alpha=0.4, label='Mean HR@5 = %.4f' % np.mean(hits_5_result))
    # plt.plot(fold, hits_10_result, 'o-', alpha=0.4, label='Mean HR@5 = %.4f' % np.mean(hits_10_result))
    # plt.plot(fold, hits_20_result, 'o-', alpha=0.4, label='Mean HR@5 = %.4f' % np.mean(hits_20_result))
    width = 0.3
    fontsize = 6
    plt.bar(fold - width, hits_5_result, width=width, alpha=0.4, label='Mean HR@5  = %.4f' % np.mean(hits_5_result))
    for a, b in zip(fold - width, hits_5_result):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=fontsize)

    plt.bar(fold, hits_10_result, width=width, alpha=0.4, label='Mean HR@10 = %.4f' % np.mean(hits_10_result))
    for a, b in zip(fold, hits_10_result):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=fontsize)

    plt.bar(fold + width, hits_20_result, width=width, alpha=0.4, label='Mean HR@20 = %.4f' % np.mean(hits_20_result))
    for a, b in zip(fold + width, hits_20_result):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=fontsize)

    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('HR@K', fontsize=12)
    # plt.xticks(fold)
    plt.ylim([0.00, 0.70])
    plt.legend(loc="upper right")
    plt.savefig("./Experiment result/HR_K_Curve.svg", dpi=300)
    plt.savefig("./Experiment result/HR_K_Curve.png")


def NDCG_k_Curve(ndcgs_5_result, ndcgs_10_result, ndcgs_20_result):
    fold = np.array([1, 2, 3, 4, 5])
    plt.figure(4, figsize=(7, 5))

    # plt.plot(fold, ndcgs_5_result, 'o-', alpha=0.4, label='Mean NDCG@5 = %.4f' % np.mean(ndcgs_5_result))
    # plt.plot(fold, ndcgs_10_result, 'o-', alpha=0.4, label='Mean NDCG@5 = %.4f' % np.mean(ndcgs_10_result))
    # plt.plot(fold, ndcgs_20_result, 'o-', alpha=0.4, label='Mean NDCG@5 = %.4f' % np.mean(ndcgs_20_result))
    width = 0.3
    fontsize = 6
    plt.bar(fold - width, ndcgs_5_result, width=width, alpha=0.4, label='Mean NDCG@5  = %.4f' % np.mean(ndcgs_5_result))
    for a, b in zip(fold - width, ndcgs_5_result):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=fontsize)

    plt.bar(fold, ndcgs_10_result, width=width, alpha=0.4, label='Mean NDCG@10 = %.4f' % np.mean(ndcgs_10_result))
    for a, b in zip(fold, ndcgs_10_result):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=fontsize)

    plt.bar(fold + width, ndcgs_20_result, width=width, alpha=0.4, label='Mean NDCG@20 = %.4f' % np.mean(ndcgs_20_result))
    for a, b in zip(fold + width, ndcgs_20_result):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=fontsize)

    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('NDCG@K', fontsize=12)
    # plt.xticks(fold)
    plt.ylim([0.00, 0.60])

    plt.legend(loc="upper right")
    plt.savefig("./Experiment result/NDCG_K_Curve.svg", dpi=300)
    plt.savefig("./Experiment result/NDCG_K_Curve.png")


def make_print_to_file(path, p):
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    sys.stdout = Logger(fileName + '_' + str(p) + '.log', path=path)

