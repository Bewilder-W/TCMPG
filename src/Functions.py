import matplotlib.pyplot as plt
import numpy as np
import re
import os
import sys
import datetime
import torch
import torch.nn.functional as F

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


def Precision_Recall_topk(pred, label, ori_label):
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

    pred_5, pred_10, pred_20 = torch.zeros(len(pred)), torch.zeros(len(pred)), torch.zeros(len(pred))
    # 取到top5，top10， top20下标的药材，并把其他药材置为0
    pred_5[indices_5] = pred[indices_5]
    pred_10[indices_10] = pred[indices_10]
    pred_20[indices_20] = pred[indices_20]
    # 计算剂量占比
    pred_5 = pred_5 / torch.sum(pred_5)
    pred_10 = pred_10 / torch.sum(pred_10)
    pred_20 = pred_20 / torch.sum(pred_20)
    # 计算top5，10，20下的余弦相似度
    cosSimi_5 = F.cosine_similarity(pred_5.double().unsqueeze(0), label.double().unsqueeze(0)).detach().numpy()
    cosSimi_10 = F.cosine_similarity(pred_10.double().unsqueeze(0), label.double().unsqueeze(0)).detach().numpy()
    cosSimi_20 = F.cosine_similarity(pred_20.double().unsqueeze(0), label.double().unsqueeze(0)).detach().numpy()
    # cosSimi_5 = F.cosine_similarity(pred[indices_5].double().unsqueeze(0), ori_label[indices_5].unsqueeze(0)).detach().numpy()
    # cosSimi_10 = F.cosine_similarity(pred[indices_10].double().unsqueeze(0), ori_label[indices_10].unsqueeze(0)).detach().numpy()
    # cosSimi_20 = F.cosine_similarity(pred[indices_20].double().unsqueeze(0), ori_label[indices_20].unsqueeze(0)).detach().numpy()

    return prec_5, prec_10, prec_20, reca_5, reca_10, reca_20, hit_5, hit_10, hit_20, all_num, indices_20, cosSimi_5, cosSimi_10, cosSimi_20


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
    plt.savefig("./Experiment result/推荐指标结果/Precision_K_Curve.svg", dpi=300)
    plt.savefig("./Experiment result/推荐指标结果/Precision_K_Curve.png")


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
    plt.savefig("./Experiment result/推荐指标结果/Recall_K_Curve.svg", dpi=300)
    plt.savefig("./Experiment result/推荐指标结果/Recall_K_Curve.png")


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
    plt.savefig("./Experiment result/推荐指标结果/HR_K_Curve.svg", dpi=300)
    plt.savefig("./Experiment result/推荐指标结果/HR_K_Curve.png")


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
    plt.savefig("./Experiment result/推荐指标结果/NDCG_K_Curve.svg", dpi=300)
    plt.savefig("./Experiment result/推荐指标结果/NDCG_K_Curve.png")


def ROC_Curve(auc, fprs, tprs):
    plt.figure(1)
    # mean_fpr = np.linspace(0, 1, 10000)
    # tpr = []

    for i in range(len(fprs)):
        # tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        # tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, label='fold %d (AUC = %.4f)' % (i + 1, auc[i]))
        # plt.plot(fprs[i], tprs[i], '-', label='fold %d (AUC = %.4f)' % (i + 1, auc[i]))
    # mean_tpr = np.mean(tpr, axis=0)
    # # mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    # # auc_std = np.std(auc)
    # plt.plot(mean_fpr, mean_tpr, color='b', alpha=0.8, label='Mean AUC = %.4f)' % (mean_auc))

    # std_tpr = np.std(tpr, axis=0)
    # tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves', fontsize=12)
    plt.legend(loc='lower right')
    plt.savefig("./Experiment result/链接预测指标结果/ROC_curves.svg", dpi=300)
    plt.savefig("./Experiment result/链接预测指标结果/ROC_curves.png")


def PR_Curve(ap, pres, recs):
    plt.figure(2)
    # mean_rec = np.linspace(0, 1, 10000)
    # pre = []

    for i in range(len(recs)):
        # pre.append(interp(mean_rec, recs[i], pres[i]))
        # pre[-1][0] = 0.0
        # plt.plot(recs[i], pres[i], label='fold %d (AUPR = %.4f)' % (i + 1, ap[i]))
        plt.plot(recs[i], pres[i], alpha=0.4, label='fold %d (AUPR = %.4f)' % (i + 1, ap[i]))

    # mean_pre = np.mean(pre, axis=0)
    # # mean_rec[-1] = 1.0
    # mean_ap = metrics.auc(pres, recs)
    # # auc_std = np.std(auc)
    # plt.plot(mean_pre, mean_rec, color='b', alpha=0.8, label='Mean AUC = %.4f)' % (mean_ap))

    # plt.plot([0, 1], [1, 0], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('PR Curves', fontsize=12)
    plt.legend(loc='lower right')
    plt.savefig("./Experiment result/链接预测指标结果/PR_curves.svg", dpi=300)
    plt.savefig("./Experiment result/链接预测指标结果/PR_curves.png")


def make_print_to_file(path, p):
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "w", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    sys.stdout = Logger(str(p) + '.log', path=path)

