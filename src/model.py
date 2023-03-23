import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
import dgl
import numpy as np
import torch
import torch.nn as tnn
from layers import Layer
import torch.nn.functional as F


class TSMPSA_GAE(nn.Block):
    def __init__(self, encoder, decoder):
        super(TSMPSA_GAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, G, prescription, symptom):
        h = self.encoder(G)
        h_prescription = h[prescription]
        h_symptom = h[symptom]
        return self.decoder(h_prescription, h_symptom)


class GraphEncoder(nn.Block):
    def __init__(self, embedding_size, n_layers, G, aggregator, dropout, alpha, ctx):
        super(GraphEncoder, self).__init__()

        self.G = G

        # G.filter_nodes ： 返回具有满足给定谓词的给定节点类型的节点的 ID。 返回的是一维向量
        self.prescription_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1).astype(np.int64).copyto(ctx)
        self.symptom_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0).astype(np.int64).copyto(ctx)

        self.layers = nn.Sequential()
        for i in range(n_layers):
            if aggregator == 'GraphSAGE':
                self.layers.add(
                    Layer(embedding_size, G, self.prescription_nodes, self.symptom_nodes, dropout, alpha, ctx))
            else:
                raise NotImplementedError

        self.prescription_emb = PrescriptionEmbedding(embedding_size, dropout)
        self.symptom_emb = SymptomEmbedding(embedding_size, dropout)

    def forward(self, G):
        # project the embedding sizes to the same vector space
        assert G.number_of_nodes() == self.G.number_of_nodes()
        G.apply_nodes(lambda nodes: {'h': self.prescription_emb(nodes.data)}, self.prescription_nodes)
        G.apply_nodes(lambda nodes: {'h': self.symptom_emb(nodes.data)}, self.symptom_nodes)

        # Message propagation and Node update
        for layer in self.layers:
            layer(G)

        return G.ndata['h']


class PrescriptionEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(PrescriptionEmbedding, self).__init__()

        seq = nn.Sequential()
        with seq.name_scope():
            # seq.add(nn.Dense(512, use_bias=False))
            seq.add(nn.Dense(embedding_size, use_bias=False))
            seq.add(nn.Dropout(dropout))
        self.proj_prescription = seq

    def forward(self, ndata):
        emb = self.proj_prescription(ndata['p_features'])

        return emb


class SymptomEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(SymptomEmbedding, self).__init__()

        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(embedding_size, use_bias=False))
            seq.add(nn.Dropout(dropout))
        self.proj_symptom = seq

    def forward(self, ndata):
        emb = self.proj_symptom(ndata['s_features'])

        return emb


class BilinearDecoder(nn.Block):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()

        self.activation = nn.Activation('sigmoid')
        with self.name_scope():
            self.W = self.params.get('dot_weights', shape=(feature_size, feature_size))

    def forward(self, h_prescription, h_symptom):
        results_adj = self.activation((nd.dot(h_prescription, self.W.data()) * h_symptom).sum(1))

        return results_adj


class TestDecoder(nn.Block):
    def __init__(self, original_prescription_sizes):
        super(TestDecoder, self).__init__()
        # self.G = G
        # self.prescription_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1).astype(np.int64).copyto(ctx)
        # self.prescription_update = Embedding()

        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(600, activation='relu'))
            seq.add(nn.Dense(1200, activation='relu'))
            seq.add(nn.Dense(original_prescription_sizes))
        self.proj = seq

    def forward(self, G, prescription_index=[], symptom_index=[]):
        """
        :param prescription_index:  提供给训练的时候
        :param symptom_index: 提供给进行单个药方验证结果使用，一个症状下标列表， 需要注意下标是从49493开始
        :return: 两个index不同时为空，也不同时有值，返回所需要的结果
        """
        h_res = None
        if prescription_index != []:
            G.update_all(fn.copy_src('h', 'h'), fn.sum('h', 'h_sum'))
            h_sum = G.ndata['h_sum']
            h_prescription = h_sum[prescription_index]
            h_res = self.proj(h_prescription)

        if symptom_index != []:
            h = G.ndata['h']
            h_symptom = nd.sum(h[symptom_index], axis=0)
            h_res = self.proj(h_symptom)

        return h_res


class Decoder(tnn.Module):
    def __init__(self, in_size, hidden1, hidden2, out_size):
        super(Decoder, self).__init__()

        self.fc1 = tnn.Linear(in_size, hidden1)
        self.fc2 = tnn.Linear(hidden1, hidden2)
        self.fc3 = tnn.Linear(hidden2, out_size)

    def forward(self, emb):
        out1 = self.fc1(emb)
        output1 = F.relu(out1)

        out2 = self.fc2(output1)
        output2 = F.relu(out2)

        return self.fc3(output2)
