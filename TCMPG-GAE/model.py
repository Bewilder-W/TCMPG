import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
import dgl
import numpy as np
import torch
import torch.nn as tnn
from layers import Layer
import torch.nn.functional as F


class TCMPG_GAE(nn.Block):
    def __init__(self, encoder, decoder):
        super(TCMPG_GAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, G, prescription, symptom):
        h = self.encoder(G)
        h_prescription = h[prescription]
        h_symptom = h[symptom]
        return self.decoder(h_prescription, h_symptom)
#
# class TCMPG_GAE(nn.Block):
#     def __init__(self, encoder, decoder):
#         super(TCMPG_GAE, self).__init__()
#
#         self.encoder = encoder
#         self.decoder = decoder
#
#     def forward(self, G, symptom_list, embedding_size, matrix_p_s, ctx):
#         """
#         :param G: 子图
#         :param symptom_list: symptom序号
#         :param embedding_size: encoder得到的向量维度
#         :param matrix_p_s: 邻接矩阵
#         :return: 输出药方的原embedding
#         """
#
#         h = self.encoder(G)
#         symptom_emb = h[symptom_list]
#         # symptom_emb = symptom_emb.asnumpy()
#         # symptom_emb = torch.from_numpy(symptom_emb)
#         # prescription_emb = generate_prescription_emb(matrix_p_s, symptom_emb, embedding_size, index_array)
#         prescription_emb = generate_prescription_emb(matrix_p_s, symptom_emb, embedding_size, ctx)
#         return self.decoder(prescription_emb)


class GraphEncoder(nn.Block):
    def __init__(self, embedding_size, n_layers, G, aggregator, dropout, alpha, ctx):
        super(GraphEncoder, self).__init__()

        self.G = G
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
