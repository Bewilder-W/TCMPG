import time
import random
import mxnet as mx
from mxnet import ndarray as nd, gluon, autograd
from mxnet.gluon import loss as gloss
import dgl
from sklearn.model_selection import KFold
from sklearn import metrics
import warnings
from utils import build_graph
from model import TCMPG_GAE, GraphEncoder, BilinearDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from scipy import sparse
from Functions import *
import openpyxl
import matplotlib.pyplot as plt
import pandas as pd


class Decoder(nn.Module):
    def __init__(self, in_size, hidden1, hidden2, out_size):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(in_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_size)

    def forward(self, emb):
        out1 = self.fc1(emb)
        output = F.relu(out1)

        out2 = self.fc2(output)
        output = F.relu(out2)
        return self.fc3(output)


def get_embedding(directory, epochs, batchsize, embedding_size, layers, dropout, alpha, lr, wd, random_seed, ctx, symptom_emb_path, train_index, valid_index):
    dgl.load_backend('mxnet')
    random.seed(random_seed)
    np.random.seed(random_seed)
    mx.random.seed(random_seed)
    aggregator = 'GraphSAGE'

    g, sample_prescription_vertices, sample_symptom_vertices, feature_p, feature_s, samples = build_graph(directory,
                                                                                                          random_seed=random_seed,
                                                                                                          ctx=ctx)

    print('## vertices:', g.number_of_nodes())
    print('## edges:', g.number_of_edges())
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

    symptom_list = list(range(20998, 23492))

    for i in range(len(train_index)):
        print('------------------------------------------------------------------------------------------------------')
        print('Training for Fold ', i + 1)

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

        print('## Training edges:', len(train_eid))
        print('## Validating edges:', len(valid_eid))

        # Train the model
        model = TCMPG_GAE(GraphEncoder(embedding_size=embedding_size, n_layers=layers, G=g_train, aggregator=aggregator,
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
        for epoch in range(epochs):
            start = time.time()
            for _ in range(batchsize):
                with mx.autograd.record():
                    score_train = model(g_train, src_train, dst_train)
                    loss_train = cross_entropy(score_train, rating_train).mean()
                    loss_train.backward()
                trainer.step(1)

            h_val = model.encoder(g_train)
            score_val = model.decoder(h_val[src_valid], h_val[dst_valid])
            loss_val = cross_entropy(score_val, rating_valid).mean()

            train_auc = metrics.roc_auc_score(np.squeeze(rating_train.asnumpy()), np.squeeze(score_train.asnumpy()))
            val_auc = metrics.roc_auc_score(np.squeeze(rating_valid.asnumpy()), np.squeeze(score_val.asnumpy()))
            val_ap = metrics.average_precision_score(np.squeeze(rating_valid.asnumpy()),
                                                     np.squeeze(score_val.asnumpy()))

            end = time.time()

            print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.asscalar(),
                  'Val Loss: %.4f' % loss_val.asscalar(), 'Learning_rate: ', trainer.learning_rate,
                  'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc,
                  'Val AP: %.4f' % val_ap, 'Time: %.2f' % (end - start))

            if val_auc > max_auc_val:
                max_auc_val = val_auc
                best_h = h_val

        symptom_emb = best_h[symptom_list].asnumpy()
        symptom_emb = torch.from_numpy(symptom_emb)
        torch.save(symptom_emb, symptom_emb_path[i])

    print('## Training Finished !')
    print('----------------------------------------------------------------------------------------------------------')


# 用学习到的symptom的embedding，计算得到每个药方的embedding
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


def print_metrics(epoch, pred_set, label_set):
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
    hits_5, hits_10, hits_20, all_nums = [], [], [], []

    for j in range(pred_set.shape[0]):
        pred, label = pred_set[j].cpu(), label_set[j].cpu()
        prec_5, prec_10, prec_20, \
        reca_5, reca_10, reca_20, \
        hit_5, hit_10, hit_20, all_num = Precision_Recall_topk(pred, label)

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

        ndcgs_5.append(ndcg_5)
        ndcgs_10.append(ndcg_10)
        ndcgs_20.append(ndcg_20)

    hit_ratio5, hit_ratio10, hit_ratio20 = sum(hits_5) / sum(all_nums), sum(hits_10) / sum(all_nums), sum(
        hits_20) / sum(all_nums)
    end = time.time()
    print('Epoch:', epoch, '| Top [5-10-20] Precision: [%.4f' % np.mean(precs_5), ', %.4f' % np.mean(precs_10), ', %.4f]' % np.mean(precs_20),
          '| Recall : [%.4f' % np.mean(recas_5), ', %.4f' % np.mean(recas_10), ', %.4f]' % np.mean(recas_20),
          '| Hit Ratio: [%.4f' % hit_ratio5, ', %.4f' % hit_ratio10, ', %.4f]' % hit_ratio20,
          '| NDCG: [%.4f' % np.mean(ndcgs_5), ', %.4f' % np.mean(ndcgs_10), ', %.4f]' % np.mean(ndcgs_20),
          'Time: %.2f' % (end-start)
          )

    return precs_5, precs_10, precs_20, recas_5, recas_10, recas_20, hit_ratio5, hit_ratio10, hit_ratio20, ndcgs_5, ndcgs_10, ndcgs_20


# 通过几个症状预测可用于治疗的新方子
def train(train_index, valid_index, matrix_p_s, emb_p, symptom_emb_path, flag=False):
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

    precs_5_result, precs_10_result, precs_20_result = [], [], []
    recas_5_result, recas_10_result, recas_20_result = [], [], []
    ndcgs_5_result, ndcgs_10_result, ndcgs_20_result = [], [], []
    hits_5_result, hits_10_result, hits_20_result = [], [], []

    for i in range(len(train_index)):

        symptom_emb = torch.load(symptom_emb_path[i])  # 2494*emb_size
        symptom_emb = torch.tensor(symptom_emb, dtype=torch.float32)
        emb_size = symptom_emb.shape[1]
        prescription_emb = generate_prescription_emb(matrix_p_s, symptom_emb, emb_size)  # 20998*emb_size

        print('------------------------------------------------------------------------------------------------------')
        print('Training for Fold ', i + 1)

        model = Decoder(emb_size, 400, 800, 1603)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        loss_func = torch.nn.MSELoss()
        loss_func = loss_func.to(device)

        train_set = prescription_emb[train_index[i]-1].to(device)
        valid_set = prescription_emb[valid_index[i]-1].to(device)
        train_label = emb_p[train_index[i]-1].to(device)
        valid_label = emb_p[valid_index[i]-1].to(device)

        start = time.time()

        for epoch in range(1600):
            # start = time.time()
            train_pred = model(train_set)  # len(train_pred)*emb_size -> len(train_pred)*1603
            train_loss = torch.sqrt(loss_func(train_pred, train_label))  # RMSE

            if epoch % 10 == 0:
                valid_pred = model(valid_set)
                # valid_loss = torch.sqrt(loss_func(valid_pred, valid_label))
                # valid_loss = torch.sqrt(loss_func(valid_pred, valid_label, weight_matrix))

                # precs_5, precs_10, precs_20 = [], [], []
                # recas_5, recas_10, recas_20 = [], [], []
                # ndcgs_5, ndcgs_10, ndcgs_20 = [], [], []
                # hits_5, hits_10, hits_20, all_nums = [], [], [], []
                # print("Training Set : ")
                # _, _, _, _, _, _, _, _, _, _, _, _, = print_metrics(epoch, train_pred, train_label)
                # print("Validating Set : ")
                precs_5, precs_10, precs_20, recas_5, recas_10, recas_20, hit_ratio5, \
                hit_ratio10, hit_ratio20, ndcgs_5, ndcgs_10, ndcgs_20 = print_metrics(epoch, valid_pred, valid_label)
                # for j in range(valid_pred.shape[0]):
                #     pred, label = valid_pred[j].cpu(), valid_label[j].cpu()
                #     prec_5, prec_10, prec_20, \
                #     reca_5, reca_10, reca_20, \
                #     hit_5, hit_10, hit_20, all_num = Precision_Recall_topk(pred, label)
                #
                #     pred, label = pred.detach().numpy(), label.detach().numpy()
                #     ndcg_5, ndcg_10, ndcg_20 = NDCG(pred, label, 5), NDCG(pred, label, 10), NDCG(pred, label, 20)
                #     precs_5.append(prec_5)
                #     precs_10.append(prec_10)
                #     precs_20.append(prec_20)
                #
                #     recas_5.append(reca_5)
                #     recas_10.append(reca_10)
                #     recas_20.append(reca_20)
                #
                #     hits_5.append(hit_5)
                #     hits_10.append(hit_10)
                #     hits_20.append(hit_20)
                #     all_nums.append(all_num)
                #
                #     ndcgs_5.append(ndcg_5)
                #     ndcgs_10.append(ndcg_10)
                #     ndcgs_20.append(ndcg_20)
                #
                # # end = time.time()
                # hit_ratio5, hit_ratio10, hit_ratio20 = sum(hits_5)/sum(all_nums), sum(hits_10)/sum(all_nums), sum(hits_20)/sum(all_nums)
                # # print('Epoch:', epoch, 'Train Loss: %.8f' % train_loss, 'Valid Loss: %.8f' % valid_loss,
                # #       '| Top [5-10-20] Precision: [%.4f' % np.mean(precs_5), ', %.4f' % np.mean(precs_10), ', %.4f]' % np.mean(precs_20),
                # #       '| Recall : [%.4f' % np.mean(recas_5), ', %.4f' % np.mean(recas_10), ', %.4f]' % np.mean(recas_20),
                # #       '| Hit Ratio: [%.4f' % hit_ratio5, ', %.4f' % hit_ratio10, ', %.4f]' % hit_ratio20,
                # #       '| NDCG: [%.4f' % np.mean(ndcgs_5), ', %.4f' % np.mean(ndcgs_10), ', %.4f]' % np.mean(ndcgs_20),
                # #       '| Time: %.2f' % (end - start)
                # #       )

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        precs_5_result.append(np.mean(precs_5))
        precs_10_result.append(np.mean(precs_10))
        precs_20_result.append(np.mean(precs_20))

        recas_5_result.append(np.mean(recas_5))
        recas_10_result.append(np.mean(recas_10))
        recas_20_result.append(np.mean(recas_20))

        hits_5_result.append(hit_ratio5)
        hits_10_result.append(hit_ratio10)
        hits_20_result.append(hit_ratio20)

        ndcgs_5_result.append(np.mean(ndcgs_5))
        ndcgs_10_result.append(np.mean(ndcgs_10))
        ndcgs_20_result.append(np.mean(ndcgs_20))
        end = time.time()

        print('Fold:', i+1,
              'Precision: %.4f' % np.mean(precs_5), ', %.4f' % np.mean(precs_10), ', %.4f' % np.mean(precs_20),
              'Recall : %.4f' % np.mean(recas_5), ', %.4f' % np.mean(recas_10), ', %.4f' % np.mean(recas_20),
              'Hit Ratio: %.4f' % hit_ratio5, ', %.4f' % hit_ratio10, ', %.4f' % hit_ratio20,
              'NDCG: %.4f' % np.mean(ndcgs_5), ', %.4f' % np.mean(ndcgs_10), ', %.4f' % np.mean(ndcgs_20),
              '| Time: %.2f' % (end - start)
              )

    print(
          ' |Mean Precision: [%.4f' % np.mean(precs_5_result), ', %.4f' % np.mean(precs_10_result), ', %.4f]\n' % np.mean(precs_20_result),
          '|Mean Recall : [%.4f' % np.mean(recas_5_result), ', %.4f' % np.mean(recas_10_result), ', %.4f]\n' % np.mean(recas_20_result),
          '|Mean Hit Ratio: [%.4f' % np.mean(hits_5_result), ', %.4f' % np.mean(hits_10_result), ', %.4f]\n' % np.mean(hits_20_result),
          '|Mean NDCG: [%.4f' % np.mean(ndcgs_5_result), ', %.4f' % np.mean(ndcgs_10_result), ', %.4f]' % np.mean(ndcgs_20_result),
          )

    print('## Training Finished !')
    print('----------------------------------------------------------------------------------------------------------')

    print('[%.4f, %.4f, %.4f, %.4f],' % (np.mean(precs_5_result), np.mean(recas_5_result), np.mean(hits_5_result), np.mean(ndcgs_5_result)))
    print('[%.4f, %.4f, %.4f, %.4f],' % (np.mean(precs_10_result), np.mean(recas_10_result), np.mean(hits_10_result), np.mean(ndcgs_10_result)))
    print('[%.4f, %.4f, %.4f, %.4f],' % (np.mean(precs_20_result), np.mean(recas_20_result), np.mean(hits_20_result), np.mean(ndcgs_20_result)))

    if flag:
        Precision_k_Curve(precs_5_result, precs_10_result, precs_20_result)
        Recall_k_Curve(recas_5_result, recas_10_result, recas_20_result)
        HR_k_Curve(hits_5_result, hits_10_result, hits_20_result)
        NDCG_k_Curve(ndcgs_5_result, ndcgs_10_result, ndcgs_20_result)
        plt.show()


def Train_with_embedding(directory, epochs, batchsize, embedding_size, layers, dropout, alpha, lr, wd, random_seed, ctx, original_path):
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

    prescription_index = list(range(0, 20998))
    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    train_index = []
    valid_index = []
    for train_idx, valid_idx in kf.split(prescription_index):
        # train_idx:array
        train_index.append(train_idx + 1)
        valid_index.append(valid_idx + 1)

    print("generate trained symptoms embedding......")
    symptom_emb_path = [original_path + '0.pt', original_path + '1.pt', original_path + '2.pt', original_path + '3.pt',
                        original_path + '4.pt']
    # if not os.path.exists(symptom_emb_path[4]):
    get_embedding(directory, epochs, batchsize, embedding_size, layers, dropout, alpha, lr, wd, random_seed, ctx, symptom_emb_path, train_index, valid_index)

    print("get the adj of prescription-symptom")
    path_matrix_p_s = './data/p_i_adj_matrix.npz'
    matrix_p_s = sparse.load_npz(path_matrix_p_s)
    matrix_p_s = matrix_p_s.toarray()

    print("get the original embeddings of prescriptions")
    path_feature_p = './data/feature_p.npz'
    feature_p = sparse.load_npz(path_feature_p)
    feature_p = feature_p.toarray()
    # 将药材剂量占比改为1
    feature_p = np.int64(feature_p > 0)
    emb_p = torch.tensor(feature_p, dtype=torch.float32)

    return train_index, valid_index, matrix_p_s, emb_p, symptom_emb_path



