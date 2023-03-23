import time
import random
from mxnet import ndarray as nd, gluon, autograd
from mxnet.gluon import loss as gloss
import dgl
from sklearn.model_selection import KFold
from sklearn import metrics
import warnings
from utils import build_graph
from model import TSMPSA_GAE, GraphEncoder, BilinearDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from scipy import sparse
from Functions import *
import openpyxl
import matplotlib.pyplot as plt
import pandas as pd
import math
import mxnet as mx
import numpy as np


class Decoder(nn.Module):
    def __init__(self, in_size, hidden1, hidden2, out_size):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(in_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_size)

    def forward(self, emb):
        out1 = self.fc1(emb)
        output = F.relu(out1)
        # output = F.sigmoid(out1)

        out2 = self.fc2(output)
        output = F.relu(out2)
        # output = F.sigmoid(out2)
        return self.fc3(output)


def get_embedding(directory, embedding_size, layers, dropout, alpha, lr, wd, random_seed, ctx, symptom_emb_path, train_index, valid_index, original_path):
    dgl.load_backend('mxnet')
    random.seed(random_seed)
    np.random.seed(random_seed)
    mx.random.seed(random_seed)
    aggregator = 'GraphSAGE'

    g, sample_prescription_vertices, sample_symptom_vertices, feature_p, feature_s, samples = build_graph(directory,
                                                                                                          random_seed=random_seed,
                                                                                                          ctx=ctx)

    print('## vertices:', g.number_of_nodes())
    # print('## edges:', g.number_of_edges())
    print('## prescription nodes:', nd.sum(g.ndata['type'] == 1).asnumpy())
    print('## symptom nodes:', nd.sum(g.ndata['type'] == 0).asnumpy())

    samples_df = pd.DataFrame(samples, columns=['prescription', 'symptom', 'label'])

    # 先创建好训练的和测试的
    print("开始----构建五折交叉验证所对应的训练图和验证图...........")
    start = time.time()
    for i in range(len(train_index)):
        # print('------------------------------------------------------------------------------------------------------')
        print('Construct Fold ', i + 1)

        samples_df['train' + str(i)] = 0
        samples_df['valid' + str(i)] = 0
        # 非常重要的部分,怎么分开训练的图和验证的图
        # for j in train_index[i]:
        #     samples_df.loc[(samples_df['prescription'] == j) & (samples_df['label'] == 1), 'train' + str(i)] = 1
        #
        # for j in valid_index[i]:
        #     samples_df.loc[(samples_df['prescription'] == j) & (samples_df['label'] == 1), 'valid' + str(i)] = 1
        samples_df.loc[(samples_df['prescription'].isin(train_index[i])) & (samples_df['label'] == 1), 'train' + str(i)] = 1
        samples_df.loc[(samples_df['prescription'].isin(valid_index[i])) & (samples_df['label'] == 1), 'valid' + str(i)] = 1

        train_sample = samples_df.loc[samples_df['train' + str(i)] == 1]
        # valid_sample = samples_df.loc[samples_df['valid' + str(i)] == 1]
        # print("### train_sample num : ", train_sample.shape[0])
        # print("### valid_sample num : ", valid_sample.shape[0])
        neg_sample = samples_df.loc[samples_df['label'] == 0]
        neg_sample_index = neg_sample.index.tolist()
        train_neg_index = random.sample(neg_sample_index, train_sample.shape[0])
        valid_neg_index = list(set(neg_sample_index) ^ set(train_neg_index))

        train_neg_index, valid_neg_index = np.array(train_neg_index), np.array(valid_neg_index)

        col = 3 + (i * 2)
        samples_df.iloc[train_neg_index, col] = 1
        samples_df.iloc[valid_neg_index, col + 1] = 1
    end = time.time()
    print("Time: %.2f" % (end-start))
    print("结束----构建五折交叉验证所对应的训练图和验证图...........")
    # 将pandas中的DataFrame文件保存在csv文件中
    samples_df.to_csv(original_path + 'sample.csv')
    symptom_list = list(range(49493, 52002))

    for i in range(len(train_index)):
        # print('------------------------------------------------------------------------------------------------------')
        # print('Training for Fold ', i + 1)

        samples_df['train'] = 0
        samples_df['valid'] = 0
        samples_df['train'].iloc[train_index[i]] = 1
        samples_df['valid'].iloc[valid_index[i]] = 1
        train_tensor = nd.from_numpy(samples_df['train' + str(i)].values.astype('int32')).copyto(ctx)
        valid_tensor = nd.from_numpy(samples_df['valid' + str(i)].values.astype('int32')).copyto(ctx)
        edge_data = {'train': train_tensor,
                     'valid': valid_tensor}

        g.edges[sample_prescription_vertices, sample_symptom_vertices].data.update(edge_data)
        g.edges[sample_symptom_vertices, sample_prescription_vertices].data.update(edge_data)

        # get the training set
        train_eid = g.filter_edges(lambda edges: edges.data['train']).astype('int64')
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True)
        g_train.copy_from_parent()
        rating_train = g_train.edata['rating']
        src_train, dst_train = g_train.all_edges()
        src_train = src_train.copyto(ctx)
        dst_train = dst_train.copyto(ctx)

        # get the validating set
        valid_eid = g.filter_edges(lambda edges: edges.data['valid']).astype('int64')
        src_valid, dst_valid = g.find_edges(valid_eid)
        rating_valid = g.edges[valid_eid].data['rating']
        src_valid = src_valid.copyto(ctx)
        dst_valid = dst_valid.copyto(ctx)

        # print('## Training edges:', len(train_eid))
        # print('## Validating edges:', len(valid_eid))

        # Train the model
        model = TSMPSA_GAE(GraphEncoder(embedding_size=embedding_size, n_layers=layers, G=g_train, aggregator=aggregator,
                                        dropout=dropout, alpha=alpha, ctx=ctx),
                           BilinearDecoder(feature_size=embedding_size))
        # model.collect_params().initialize(init=mx.init.Xavier(magnitude=math.sqrt(2.0)), ctx=ctx)
        model.collect_params().initialize(init=mx.init.MSRAPrelu(), ctx=ctx)
        # lrs = mx.lr_scheduler.FactorScheduler(step=50, factor=0.9, stop_factor_lr=1e-8, base_lr=lr)
        # trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd, 'lr_scheduler': lrs})
        cross_entropy = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
        trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
        max_auc_val = 0
        best_h = None
        # 端到端训练
        for e in range(20):
            for _ in range(10):
                with mx.autograd.record():
                    score_train = model(g_train, src_train, dst_train)
                    loss_train = cross_entropy(score_train, rating_train).mean()
                    loss_train.backward()
                trainer.step(1)

            h_val = model.encoder(g_train)
            score_val = model.decoder(h_val[src_valid], h_val[dst_valid])
            # loss_val = cross_entropy(score_val, rating_valid).mean()
            # train_auc = metrics.roc_auc_score(np.squeeze(rating_train.asnumpy()), np.squeeze(np.nan_to_num(score_train.asnumpy())))
            # rating = rating_valid.asnumpy()
            # print(np.isnan(rating).any())
            # print(np.isnan(score).any())
            val_auc = metrics.roc_auc_score(np.squeeze(rating_valid.asnumpy()), np.squeeze(np.nan_to_num(score_val.asnumpy(), nan=1e-8)))
            # val_ap = metrics.average_precision_score(np.squeeze(rating_valid.asnumpy()),
            #                                          np.squeeze(np.nan_to_num(score_val.asnumpy())))
            #
            # print('Train Loss: %.4f' % loss_train.asscalar(),
            #       'Val Loss: %.4f' % loss_val.asscalar(), 'Learning_rate: ', trainer.learning_rate,
            #       'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc,
            #       'Val AP: %.4f' % val_ap, 'Time: %.2f' % (end - start))
            if val_auc > max_auc_val:
                max_auc_val = val_auc
                best_h = h_val
        # # 最后让编码器再跑一次，得到融合信息之后的特征向量
        # h = model.encoder(g_train)
        symptom_emb = best_h[symptom_list].asnumpy()
        symptom_emb = torch.from_numpy(symptom_emb)
        torch.save(symptom_emb, symptom_emb_path[i])

    # print('## Training Finished !')
    # print('----------------------------------------------------------------------------------------------------------')


def generate_prescription_emb(matrix_p_s, symptom_emb, symptom_size):
    """
    :param matrix_p_s: 邻接矩阵
    :param symptom_emb: 所有症状的embeddings
    :param symptom_size: 症状embeddings的维度
    :return:
    """

    prescription_emb = torch.zeros((matrix_p_s.shape[0], symptom_size))
    for i in range(matrix_p_s.shape[0]):
        index = np.where(matrix_p_s[i] == 1)
        index = list(index[0])
        prescription_emb[i] = torch.sum(symptom_emb[index], dim=0)

    return prescription_emb


def print_metrics(pred_set, label_set, ori_label_set):
    """
    :param epoch: 此时的eopoch值
    :param pred_set: 模型得到的预测结果
    :param label_set: 药方初始embeddings
    :return: 各种指标在top：5，10，20 情况下的值
    """

    start = time.time()
    precs_5, precs_10, precs_20 = [], [], []
    recas_5, recas_10, recas_20 = [], [], []
    ndcgs_5, ndcgs_10, ndcgs_20 = [], [], []
    hits_5, hits_10, hits_20 = [], [], []
    cosSimis_5, cosSimis_10, cosSimis_20 = [], [], []
    all_nums, indices_20 = [], []

    length = pred_set.shape[0]
    for j in range(length):
        pred, label, ori_label = pred_set[j].cpu(), label_set[j].cpu(), ori_label_set[j].cpu()

        prec_5, prec_10, prec_20, \
        reca_5, reca_10, reca_20, \
        hit_5, hit_10, hit_20, \
        all_num, indice_20, \
        cosSimi_5, cosSimi_10, cosSimi_20 = Precision_Recall_topk(pred, label, ori_label)

        pred, label = pred.detach().numpy(), label.detach().numpy()
        ndcg_5, ndcg_10, ndcg_20 = NDCG(pred, label, 5), NDCG(pred, label, 10), NDCG(pred, label, 20)

        precs_5.append(prec_5)
        precs_10.append(prec_10)
        precs_20.append(prec_20)

        recas_5.append(reca_5)
        recas_10.append(reca_10)
        recas_20.append(reca_20)

        hits_5.append(hit_5)
        hits_10.append(hit_10)
        hits_20.append(hit_20)
        all_nums.append(all_num)
        indices_20.append(indice_20)

        ndcgs_5.append(ndcg_5)
        ndcgs_10.append(ndcg_10)
        ndcgs_20.append(ndcg_20)

        cosSimis_5.append(cosSimi_5)
        cosSimis_10.append(cosSimi_10)
        cosSimis_20.append(cosSimi_20)


    hit_ratio5, hit_ratio10, hit_ratio20 = sum(hits_5) / sum(all_nums), sum(hits_10) / sum(all_nums), sum(
        hits_20) / sum(all_nums)

    end = time.time()

    print('Precision:%.4f' % np.mean(precs_5), ', %.4f' % np.mean(precs_10), ', %.4f' % np.mean(precs_20),
          '| Recall:%.4f' % np.mean(recas_5), ', %.4f' % np.mean(recas_10), ', %.4f' % np.mean(recas_20),
          '| Hit Ratio:%.4f' % hit_ratio5, ', %.4f' % hit_ratio10, ', %.4f' % hit_ratio20,
          '| NDCG:%.4f' % np.mean(ndcgs_5), ', %.4f' % np.mean(ndcgs_10), ', %.4f' % np.mean(ndcgs_20),
          '| Cosine Similarity:%.4f' % np.mean(cosSimis_5), ', %.4f' % np.mean(cosSimis_10), ', %.4f' % np.mean(cosSimis_20),
          'Time: %.2f' % (end - start)
          )

    return np.sum(precs_5), np.sum(precs_10), np.sum(precs_20), \
           np.sum(recas_5), np.sum(recas_10), np.sum(recas_20), \
           sum(hits_5), sum(hits_10), sum(hits_20), \
           np.sum(ndcgs_5), np.sum(ndcgs_10), np.sum(ndcgs_20), \
           indices_20, sum(all_nums), \
           np.nansum(cosSimis_5), np.nansum(cosSimis_10), np.nansum(cosSimis_20),


# 保存五折交叉验证中训练集和验证集的编号
def save_Sample(index, path):
    for i in range(len(index)):
        with open(path + str(i), 'w', encoding='utf-8') as file:
            for j in range(len(index[i])):
                file.write("%s\n" % (str(index[i][j])))


# 通过几个症状预测可用于治疗的新方子
def train(train_index, valid_index, matrix_p_s, feature_p, emb_p, symptom_emb_path, original_path, flag=False):
    """
    :param train_index: K-Fold 输出得到的训练集
    :param valid_index: K-Fold 输出得到的验证集
    :param matrix_p_s: 邻接矩阵
    :param emb_p: 药方的原特征矩阵
    :param symptom_emb_path: 产生的症状embeddings的路径
    :param flag: 是否绘制结果图表
    :return:
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('------------------------------------------------------------------------------------------------------')
    print("Training start............")

    valid_len = []
    precs_5_result, precs_10_result, precs_20_result = [], [], []
    recas_5_result, recas_10_result, recas_20_result = [], [], []
    ndcgs_5_result, ndcgs_10_result, ndcgs_20_result = [], [], []
    hits_5_result, hits_10_result, hits_20_result, hits_all_count = [], [], [], []
    cosSimis_5_result, cosSimis_10_result, cosSimis_20_result = [], [], []

    for i in range(len(train_index)):

        symptom_emb = torch.load(symptom_emb_path[i])  # 2509*emb_size
        symptom_emb = torch.tensor(symptom_emb, dtype=torch.float32)
        emb_size = symptom_emb.shape[1]
        prescription_emb = generate_prescription_emb(matrix_p_s, symptom_emb, emb_size)  # 49493*emb_size

        print('------------------------------------------------------------------------------------------------------')
        print('Training for Fold ', i + 1)

        model = Decoder(emb_size, 600, 1200, 2284)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss_func = torch.nn.MSELoss()
        loss_func = loss_func.to(device)

        train_set = prescription_emb[train_index[i]-1].to(device)
        valid_set = prescription_emb[valid_index[i]-1].to(device)
        train_label = emb_p[train_index[i]-1].to(device)
        valid_label = emb_p[valid_index[i]-1].to(device)
        original_train_label = torch.tensor(feature_p[train_index[i]-1], dtype=torch.float).to(device)
        original_valid_label = torch.tensor(feature_p[valid_index[i]-1], dtype=torch.float).to(device)

        # 保存训练标签和测试标签.n
        # np.savetxt(original_path + 'TrainLabel_' + str(i) + '.csv', original_train_label.cpu().numpy(), fmt='%.8f', delimiter=',')
        # np.savetxt(original_path + 'TestLabel_' + str(i) + '.csv', original_valid_label.cpu().numpy(), fmt='%.8f', delimiter=',')

        for epoch in range(500):
            # start = time.time()
            train_pred = model(train_set)  # len(train_pred)*emb_size -> len(train_pred)*2284
            train_loss = torch.sqrt(loss_func(train_pred, train_label))

            if epoch % 10 == 0:
                valid_pred = model(valid_set)
                valid_loss = torch.sqrt(loss_func(valid_pred, valid_label))

                print("Epoch: " + str(epoch), "Loss : %.4f" % valid_loss.item())
                # print("Train:")
                # _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, = print_metrics(train_pred, train_label, original_train_label)
                # train_cosine_result = F.cosine_similarity(train_pred.cpu().double(), original_train_label)
                # print("Cosine Similaritty: %.4f" % torch.mean(train_cosine_result).item())
                # print("Valid:")
                precs_5, precs_10, precs_20, recas_5, recas_10, recas_20, hit_count5, hit_count10, hit_count20, \
                ndcgs_5, ndcgs_10, ndcgs_20, valid_indices_20, hit_all_count, cosSimi_5, cosSimi_10, cosSimi_20  = print_metrics(valid_pred, valid_label, original_valid_label)
                # valid_cosine_result = F.cosine_similarity(valid_pred.cpu().double(), original_valid_label)
                # print("Cosine Similaritty: %.4f" % torch.mean(valid_cosine_result).item())
                print()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        precs_5_result.append(precs_5)
        precs_10_result.append(precs_10)
        precs_20_result.append(precs_20)

        recas_5_result.append(recas_5)
        recas_10_result.append(recas_10)
        recas_20_result.append(recas_20)

        hits_5_result.append(hit_count5)
        hits_10_result.append(hit_count10)
        hits_20_result.append(hit_count20)
        hits_all_count.append(hit_all_count)

        ndcgs_5_result.append(ndcgs_5)
        ndcgs_10_result.append(ndcgs_10)
        ndcgs_20_result.append(ndcgs_20)
        valid_len.append(len(valid_index[i]))
        # cosines_result.append(torch.sum(valid_cosine_result).item())
        cosSimis_5_result.append(cosSimi_5)
        cosSimis_10_result.append(cosSimi_10)
        cosSimis_20_result.append(cosSimi_20)

        # 将每一折测试集预测得到的实验结果进行保存
        # train_indices_20 = np.array(train_indices_20)
        valid_indices_20 = np.array(valid_indices_20)
        # np.save(original_path + 'TestResult_top20_' + str(i) + '.npy', valid_indices_20)
        # np.savetxt(original_path + 'TestResult_' + str(i) + '.csv', valid_pred.cpu().detach().numpy(), fmt='%.8f', delimiter=',')
        # np.save(original_path + 'Result_test_' + str(i) + '.npy', valid_pred.numpy())

    mean_pre_5, mean_pre_10, mean_pre_20 = np.sum(precs_5_result) / np.sum(valid_len), np.sum(precs_10_result) / np.sum(valid_len), np.sum(precs_20_result) / np.sum(valid_len)
    mean_rec_5, mean_rec_10, mean_rec_20 = np.sum(recas_5_result) / np.sum(valid_len), np.sum(recas_10_result) / np.sum(valid_len), np.sum(recas_20_result) / np.sum(valid_len)
    mean_ht_5, mean_ht_10, mean_ht_20 = np.sum(hits_5_result) / np.sum(hits_all_count), np.sum(hits_10_result) / np.sum(hits_all_count), np.sum(hits_20_result) / np.sum(hits_all_count)
    mean_ndcg_5, mean_ndcg_10, mean_ndcg_20 = np.sum(ndcgs_5_result) / np.sum(valid_len), np.sum(ndcgs_10_result) / np.sum(valid_len), np.sum(ndcgs_20_result) / np.sum(valid_len)
    mean_cosSimi_5, mean_cosSimi_10, mean_cosSimi_20 = np.sum(cosSimis_5_result) / np.sum(valid_len), np.sum(cosSimis_10_result) / np.sum(valid_len), np.sum(cosSimis_20_result) / np.sum(valid_len)
    # mean_cosine_simi = np.sum(cosines_result) / np.sum(valid_len)

    print(
        ' |Mean Precision: [%.4f' % mean_pre_5, ', %.4f' % mean_pre_10, ', %.4f]\n' % mean_pre_20,
        '|Mean Recall : [%.4f' % mean_rec_5, ', %.4f' % mean_rec_10, ', %.4f]\n' % mean_rec_20,
        '|Mean Hit Ratio: [%.4f' % mean_ht_5, ', %.4f' % mean_ht_10, ', %.4f]\n' % mean_ht_20,
        '|Mean NDCG: [%.4f' % mean_ndcg_5, ', %.4f' % mean_ndcg_10, ', %.4f]\n' % mean_ndcg_20,
        '|Mean Cosine Similarity: [%.4f' % mean_cosSimi_5, ', %.4f' % mean_cosSimi_10, ', %.4f]\n' % mean_cosSimi_20,
        # '|Mean Cosine Similarity: %.4f' % mean_cosine_simi
    )

    print('## Training Finished !')
    print('----------------------------------------------------------------------------------------------------------')

    print('[%.4f, %.4f, %.4f, %.4f, %.4f],' % (mean_pre_5, mean_rec_5, mean_ht_5, mean_ndcg_5, mean_cosSimi_5))
    print('[%.4f, %.4f, %.4f, %.4f, %.4f],' % (mean_pre_10, mean_rec_10, mean_ht_10, mean_ndcg_10, mean_cosSimi_10))
    print('[%.4f, %.4f, %.4f, %.4f, %.4f],' % (mean_pre_20, mean_rec_20, mean_ht_20, mean_ndcg_20, mean_cosSimi_20))

    if flag:
        Precision_k_Curve(precs_5_result, precs_10_result, precs_20_result)
        Recall_k_Curve(recas_5_result, recas_10_result, recas_20_result)
        HR_k_Curve(hits_5_result, hits_10_result, hits_20_result)
        NDCG_k_Curve(ndcgs_5_result, ndcgs_10_result, ndcgs_20_result)
        plt.show()


def Train_without_embedding(random_seed, original_path):
    """
    :param random_seed: 随机种子
    :param original_path: embedding保存的路径
    :return:
    """

    prescription_index = list(range(0, 49493))
    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    train_index = []
    valid_index = []
    for train_idx, valid_idx in kf.split(prescription_index):
        # train_idx:array
        train_index.append(train_idx + 1)
        valid_index.append(valid_idx + 1)

    # 将每一折的训练样本和测试样本都保存好
    # save_Sample(train_index, path=original_path + 'train_')
    # save_Sample(valid_index, path=original_path + 'valid_')

    print("generate trained symptoms embedding......")
    symptom_emb_path = [original_path + '0.pt', original_path + '1.pt', original_path + '2.pt', original_path + '3.pt',
                        original_path + '4.pt']

    print("get the adj of prescription-symptom")
    path_matrix_p_s = './data/p_s_adj_matrix.npz'
    matrix_p_s = sparse.load_npz(path_matrix_p_s)
    matrix_p_s = matrix_p_s.toarray()

    print("get the original embeddings of prescriptions")
    path_feature_p = './data/feature_p.npz'
    feature_p = sparse.load_npz(path_feature_p)
    feature_p = feature_p.toarray()
    feature = np.int64(feature_p > 0)
    # feature[feature > 0] = 2
    emb_p = torch.tensor(feature, dtype=torch.float32)

    return train_index, valid_index, matrix_p_s, feature_p, emb_p, symptom_emb_path


def Train_with_embedding(directory, embedding_size, layers, dropout, alpha, lr, wd, random_seed, ctx, original_path):
    """
    :param directory: 数据存放的总文件夹
    :param epochs: 训练得到embeddings所需的
    :param batchsize:
    :param embedding_size:
    :param layers:
    :param dropout:
    :param alpha:
    :param lr:
    :param wd:
    :param random_seed:
    :param ctx:
    :param original_path:
    :return:
    """

    prescription_index = list(range(0, 49493))
    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    train_index = []
    valid_index = []
    for train_idx, valid_idx in kf.split(prescription_index):
        # train_idx:array
        train_index.append(train_idx + 1)
        valid_index.append(valid_idx + 1)

    # 将每一折的训练样本和测试样本都保存好
    save_Sample(train_index, path=original_path + 'train_')
    save_Sample(valid_index, path=original_path + 'valid_')

    print("generate trained symptoms embedding......")
    symptom_emb_path = [original_path + '0.pt', original_path + '1.pt', original_path + '2.pt', original_path + '3.pt',
                        original_path + '4.pt']
    # if not os.path.exists(symptom_emb_path[4]):
    get_embedding(directory, embedding_size, layers, dropout, alpha, lr, wd, random_seed, ctx,
                  symptom_emb_path, train_index, valid_index, original_path)

    print("get the adj of prescription-symptom")
    path_matrix_p_s = './data/p_s_adj_matrix.npz'
    matrix_p_s = sparse.load_npz(path_matrix_p_s)
    matrix_p_s = matrix_p_s.toarray()

    print("get the original embeddings of prescriptions")
    path_feature_p = './data/feature_p.npz'
    feature_p = sparse.load_npz(path_feature_p)
    feature_p = feature_p.toarray()
    feature = np.int64(feature_p > 0)
    emb_p = torch.tensor(feature, dtype=torch.float32)

    return train_index, valid_index, matrix_p_s, feature_p, emb_p, symptom_emb_path


def test_embedding(directory, embedding_size, layers, dropout, alpha, lr, wd, random_seed, ctx, train_index, original_path):
    dgl.load_backend('mxnet')
    random.seed(random_seed)
    np.random.seed(random_seed)
    mx.random.seed(random_seed)
    aggregator = 'GraphSAGE'

    g, sample_prescription_vertices, sample_symptom_vertices, feature_p, feature_s, samples = build_graph(directory,
                                                                                                          random_seed=random_seed,
                                                                                                          ctx=ctx)

    print('## vertices:', g.number_of_nodes())
    # print('## edges:', g.number_of_edges())
    print('## prescription nodes:', nd.sum(g.ndata['type'] == 1).asnumpy())
    print('## symptom nodes:', nd.sum(g.ndata['type'] == 0).asnumpy())

    samples_df = pd.DataFrame(samples, columns=['prescription', 'symptom', 'label'])

    # 先创建好训练的和测试的
    start = time.time()
    for i in range(len(train_index)):
        samples_df['train' + str(i)] = 0
        samples_df.loc[(samples_df['prescription'].isin(train_index[i])) & (samples_df['label'] == 1), 'train' + str(i)] = 1
        train_sample = samples_df.loc[samples_df['train' + str(i)] == 1]
        neg_sample = samples_df.loc[samples_df['label'] == 0]
        neg_sample_index = neg_sample.index.tolist()
        train_neg_index = random.sample(neg_sample_index, train_sample.shape[0])
        train_neg_index= np.array(train_neg_index)
        col = 3 + (i * 2)
        samples_df.iloc[train_neg_index, col] = 1
    end = time.time()
    print("Time: %.2f" % (end-start))
    symptom_list = list(range(49493, 52002))

    for i in range(len(train_index)):

        samples_df['train'] = 0
        samples_df['train'].iloc[train_index[i]] = 1
        train_tensor = nd.from_numpy(samples_df['train' + str(i)].values.astype('int32')).copyto(ctx)
        edge_data = {'train': train_tensor}

        g.edges[sample_prescription_vertices, sample_symptom_vertices].data.update(edge_data)
        g.edges[sample_symptom_vertices, sample_prescription_vertices].data.update(edge_data)

        # get the training set
        train_eid = g.filter_edges(lambda edges: edges.data['train']).astype('int64')
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True)
        g_train.copy_from_parent()
        rating_train = g_train.edata['rating']
        src_train, dst_train = g_train.all_edges()
        src_train = src_train.copyto(ctx)
        dst_train = dst_train.copyto(ctx)

        # Train the model
        model = TSMPSA_GAE(GraphEncoder(embedding_size=embedding_size, n_layers=layers, G=g_train, aggregator=aggregator,
                                        dropout=dropout, alpha=alpha, ctx=ctx),
                           BilinearDecoder(feature_size=embedding_size))
        model.collect_params().initialize(init=mx.init.MSRAPrelu(), ctx=ctx)
        cross_entropy = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
        trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
        max_auc_val = 0
        best_h = None

        for e in range(20):
            for _ in range(10):
                with mx.autograd.record():
                    score_train = model(g_train, src_train, dst_train)
                    loss_train = cross_entropy(score_train, rating_train).mean()
                    loss_train.backward()
                trainer.step(1)
            h_val = model.encoder(g_train)
            train_auc = metrics.roc_auc_score(np.squeeze(rating_train.asnumpy()), np.squeeze(np.nan_to_num(score_train.asnumpy(), nan=1e-8)))
            if train_auc > max_auc_val:
                max_auc_val = train_auc
                best_h = h_val
        symptom_emb = best_h[symptom_list].asnumpy()
        symptom_emb = torch.from_numpy(symptom_emb)
        torch.save(symptom_emb, original_path + 'symptoms_emb.pt')


def before_test(directory, embedding_size, layers, dropout, alpha, lr, wd, random_seed, ctx, original_path):
    index = list(range(0, 49493))
    train_index = []
    train_index.append(index)
    print("generate trained symptoms embedding......")
    test_embedding(directory, embedding_size, layers, dropout, alpha, lr, wd, random_seed, ctx, train_index, original_path)

    print("get the adj of prescription-symptom")
    path_matrix_p_s = './data/p_s_adj_matrix.npz'
    matrix_p_s = sparse.load_npz(path_matrix_p_s)
    matrix_p_s = matrix_p_s.toarray()

    print("get the original embeddings of prescriptions")
    path_feature_p = './data/feature_p.npz'
    feature_p = sparse.load_npz(path_feature_p)
    feature_p = feature_p.toarray()
    feature = np.int64(feature_p > 0)
    emb_p = torch.tensor(feature, dtype=torch.float32)

    return train_index, matrix_p_s, feature_p, emb_p


def test(train_index, matrix_p_s, feature_p, emb_p, original_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('------------------------------------------------------------------------------------------------------')
    print("Training start............")

    for i in range(len(train_index)):

        symptom_emb = torch.load(original_path + 'symptoms_emb.pt')
        symptom_emb = torch.tensor(symptom_emb, dtype=torch.float32)
        emb_size = symptom_emb.shape[1]
        prescription_emb = generate_prescription_emb(matrix_p_s, symptom_emb, emb_size)  # 49493*emb_size

        model = Decoder(emb_size, 600, 1200, 2284)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss_func = torch.nn.MSELoss()
        loss_func = loss_func.to(device)

        train_set = prescription_emb[train_index[i]-1].to(device)
        train_label = emb_p[train_index[i]-1].to(device)
        original_train_label = torch.tensor(feature_p[train_index[i]-1], dtype=torch.float).to(device)

        for epoch in range(500):
            # start = time.time()
            train_pred = model(train_set)  # len(train_pred)*emb_size -> len(train_pred)*2284
            train_loss = torch.sqrt(loss_func(train_pred, train_label))

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        torch.save(model, original_path + 'trained_model')
        torch.save(symptom_emb, original_path + 'trained_symptom_emb.pt')


# 不加GNN-encoder层的情况
def without_encoder(original_path, random_seed, flag=False):
    """
    :param random_seed: 随机种子
    :param flag: 是否绘制结果图表
    :return:
    """

    prescription_index = list(range(0, 49493))
    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    train_index = []
    valid_index = []
    for train_idx, valid_idx in kf.split(prescription_index):
        # train_idx:array
        train_index.append(train_idx + 1)
        valid_index.append(valid_idx + 1)

    # 将每一折的训练样本和测试样本都保存好
    save_Sample(train_index, path=original_path + 'train_')
    save_Sample(valid_index, path=original_path + 'valid_')

    print("get the adj of prescription-symptom")
    path_matrix_p_s = './data/p_s_adj_matrix.npz'
    matrix_p_s = sparse.load_npz(path_matrix_p_s)
    matrix_p_s = matrix_p_s.toarray()

    print("get the original embeddings of prescriptions")
    path_feature_p = './data/feature_p.npz'
    feature_p = sparse.load_npz(path_feature_p)
    feature_p = feature_p.toarray()
    # 将药材剂量占比改为1
    feature = np.int64(feature_p > 0)
    emb_p = torch.tensor(feature, dtype=torch.float32)

    print("get the embeddings of symptoms")
    path_feature_s = './data/feature_s_none_w2v.npz'
    feature_s = sparse.load_npz(path_feature_s)
    feature_s = feature_s.toarray()
    # path_feature_s = './data/feature_s.npz'
    # feature_object = np.load(path_feature_s)
    # feature_s = feature_object["feature_s"]
    symptom_emb = torch.tensor(feature_s, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('------------------------------------------------------------------------------------------------------')
    print("Training start............")

    valid_len = []
    precs_5_result, precs_10_result, precs_20_result = [], [], []
    recas_5_result, recas_10_result, recas_20_result = [], [], []
    ndcgs_5_result, ndcgs_10_result, ndcgs_20_result = [], [], []
    hits_5_result, hits_10_result, hits_20_result, hits_all_count = [], [], [], []
    cosSimis_5_result, cosSimis_10_result, cosSimis_20_result = [], [], []

    for i in range(len(train_index)):

        emb_size = symptom_emb.shape[1]
        prescription_emb = generate_prescription_emb(matrix_p_s, symptom_emb, emb_size)

        print('------------------------------------------------------------------------------------------------------')
        print('Training for Fold ', i + 1)

        model = Decoder(emb_size, 600, 1200, 2284)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss_func = torch.nn.MSELoss()
        loss_func = loss_func.to(device)

        train_set = prescription_emb[train_index[i]-1].to(device)
        valid_set = prescription_emb[valid_index[i]-1].to(device)
        train_label = emb_p[train_index[i]-1].to(device)
        valid_label = emb_p[valid_index[i]-1].to(device)

        original_train_label = torch.tensor(feature_p[train_index[i]-1], dtype=torch.float).to(device)
        original_valid_label = torch.tensor(feature_p[valid_index[i]-1], dtype=torch.float).to(device)

        # 保存训练标签和测试标签.n
        # np.savetxt(original_path + 'TrainLabel_' + str(i) + '.csv', original_train_label.cpu().numpy(), fmt='%.8f', delimiter=',')
        # np.savetxt(original_path + 'TestLabel_' + str(i) + '.csv', original_valid_label.cpu().numpy(), fmt='%.8f', delimiter=',')

        for epoch in range(500):
            # start = time.time()
            train_pred = model(train_set)  # len(train_pred)*emb_size -> len(train_pred)*2284
            train_loss = torch.sqrt(loss_func(train_pred, train_label))

            if epoch % 10 == 0:
                valid_pred = model(valid_set)
                valid_loss = torch.sqrt(loss_func(valid_pred, valid_label))
                # valid_loss = torch.sqrt(loss_func(valid_pred, valid_label, weight_matrix))

                print("Epoch: " + str(epoch), "Loss : %.4f" % valid_loss.item())
                # print("Train:")
                # _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, = print_metrics(train_pred, train_label, original_train_label)
                # train_cosine_result = F.cosine_similarity(train_pred.cpu().double(), original_train_label)
                # print("Cosine Similaritty: %.4f" % torch.mean(train_cosine_result).item())
                # print("Valid:")
                precs_5, precs_10, precs_20, recas_5, recas_10, recas_20, hit_count5, hit_count10, hit_count20, \
                ndcgs_5, ndcgs_10, ndcgs_20, valid_indices_20, hit_all_count, cosSimi_5, cosSimi_10, cosSimi_20 = print_metrics(
                    valid_pred, valid_label, original_valid_label)
                # valid_cosine_result = F.cosine_similarity(valid_pred.cpu().double(), original_valid_label)
                # print("Cosine Similaritty: %.4f" % torch.mean(valid_cosine_result).item())
                print()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        precs_5_result.append(precs_5)
        precs_10_result.append(precs_10)
        precs_20_result.append(precs_20)

        recas_5_result.append(recas_5)
        recas_10_result.append(recas_10)
        recas_20_result.append(recas_20)

        hits_5_result.append(hit_count5)
        hits_10_result.append(hit_count10)
        hits_20_result.append(hit_count20)
        hits_all_count.append(hit_all_count)

        ndcgs_5_result.append(ndcgs_5)
        ndcgs_10_result.append(ndcgs_10)
        ndcgs_20_result.append(ndcgs_20)
        valid_len.append(len(valid_index[i]))
        # cosines_result.append(torch.sum(valid_cosine_result).item())
        cosSimis_5_result.append(cosSimi_5)
        cosSimis_10_result.append(cosSimi_10)
        cosSimis_20_result.append(cosSimi_20)

        # 将每一折测试集预测得到的实验结果进行保存
        # train_indices_20 = np.array(train_indices_20)
        valid_indices_20 = np.array(valid_indices_20)
        # np.save(original_path + 'TestResult_top20_' + str(i) + '.npy', valid_indices_20)
        # np.savetxt(original_path + 'TestResult_' + str(i) + '.csv', valid_pred.cpu().detach().numpy(), fmt='%.8f',
        #            delimiter=',')
        # np.save(original_path + 'Result_test_' + str(i) + '.npy', valid_pred.numpy())

    mean_pre_5, mean_pre_10, mean_pre_20 = np.sum(precs_5_result) / np.sum(valid_len), np.sum(precs_10_result) / np.sum(
        valid_len), np.sum(precs_20_result) / np.sum(valid_len)
    mean_rec_5, mean_rec_10, mean_rec_20 = np.sum(recas_5_result) / np.sum(valid_len), np.sum(recas_10_result) / np.sum(
        valid_len), np.sum(recas_20_result) / np.sum(valid_len)
    mean_ht_5, mean_ht_10, mean_ht_20 = np.sum(hits_5_result) / np.sum(hits_all_count), np.sum(hits_10_result) / np.sum(
        hits_all_count), np.sum(hits_20_result) / np.sum(hits_all_count)
    mean_ndcg_5, mean_ndcg_10, mean_ndcg_20 = np.sum(ndcgs_5_result) / np.sum(valid_len), np.sum(
        ndcgs_10_result) / np.sum(valid_len), np.sum(ndcgs_20_result) / np.sum(valid_len)
    mean_cosSimi_5, mean_cosSimi_10, mean_cosSimi_20 = np.sum(cosSimis_5_result) / np.sum(valid_len), np.sum(
        cosSimis_10_result) / np.sum(valid_len), np.sum(cosSimis_20_result) / np.sum(valid_len)
    # mean_cosine_simi = np.sum(cosines_result) / np.sum(valid_len)

    print(
        ' |Mean Precision: [%.4f' % mean_pre_5, ', %.4f' % mean_pre_10, ', %.4f]\n' % mean_pre_20,
        '|Mean Recall : [%.4f' % mean_rec_5, ', %.4f' % mean_rec_10, ', %.4f]\n' % mean_rec_20,
        '|Mean Hit Ratio: [%.4f' % mean_ht_5, ', %.4f' % mean_ht_10, ', %.4f]\n' % mean_ht_20,
        '|Mean NDCG: [%.4f' % mean_ndcg_5, ', %.4f' % mean_ndcg_10, ', %.4f]\n' % mean_ndcg_20,
        '|Mean Cosine Similarity: [%.4f' % mean_cosSimi_5, ', %.4f' % mean_cosSimi_10, ', %.4f]\n' % mean_cosSimi_20,
        # '|Mean Cosine Similarity: %.4f' % mean_cosine_simi
    )

    print('## Training Finished !')
    print('----------------------------------------------------------------------------------------------------------')

    print('[%.4f, %.4f, %.4f, %.4f, %.4f],' % (mean_pre_5, mean_rec_5, mean_ht_5, mean_ndcg_5, mean_cosSimi_5))
    print('[%.4f, %.4f, %.4f, %.4f, %.4f],' % (mean_pre_10, mean_rec_10, mean_ht_10, mean_ndcg_10, mean_cosSimi_10))
    print('[%.4f, %.4f, %.4f, %.4f, %.4f],' % (mean_pre_20, mean_rec_20, mean_ht_20, mean_ndcg_20, mean_cosSimi_20))

    if flag:
        Precision_k_Curve(precs_5_result, precs_10_result, precs_20_result)
        Recall_k_Curve(recas_5_result, recas_10_result, recas_20_result)
        HR_k_Curve(hits_5_result, hits_10_result, hits_20_result)
        NDCG_k_Curve(ndcgs_5_result, ndcgs_10_result, ndcgs_20_result)
        plt.show()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    path = ['weight_decay/0/']
    for i in path:
        original_path = './data/trained_symptom_emb/' + i
        train_index, valid_index, matrix_p_s, emb_p, symptom_emb_path = Train_without_embedding(random_seed=123, original_path=original_path)

        train(train_index, valid_index, matrix_p_s, emb_p, symptom_emb_path, flag=False)

