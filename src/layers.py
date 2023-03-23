import math
import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
import dgl
import dgl.function as fn
import numpy as np
# from mxnet import np, npx


class Layer(nn.Block):
    def __init__(self, feature_size, G, prescription_nodes, symptom_nodes, dropout, alpha, ctx):
        super(Layer, self).__init__()

        self.feature_size = feature_size
        self.G = G
        self.prescription_nodes = prescription_nodes
        self.symptom_nodes = symptom_nodes
        self.ctx = ctx

        self.prescription_update = NodeUpdate(feature_size, dropout, alpha)
        self.symptom_update = NodeUpdate(feature_size, dropout, alpha)

        all_nodes = mx.nd.arange(G.number_of_nodes(), dtype=np.int64)
        self.deg = G.in_degrees(all_nodes).astype(np.float32).copyto(ctx)

    def forward(self, G):
        assert G.number_of_nodes() == self.G.number_of_nodes()
        G.ndata['deg'] = self.deg

        # fn.copy_src = fn.copy_u(src, out) copy_u将源节点数据复制为消息
        # fn.sum(msg, out) 输出为out，输出为节点特征信息， 聚合函数，只执行聚合消息任务
        # G.update_all(message_func, reduce_func, apply_node_func) 分别为消息函数，聚合函数，更新函数（一般在下面指明）
        G.update_all(fn.copy_src('h', 'h'), fn.sum('h', 'h_agg'))  # mean, sum，max, min
        # G.update_all(fn.copy_src('h', 'h'), fn.mean('h', 'h_agg'))
        # G.update_all(fn.copy_src('h', 'h'), fn.max('h', 'h_agg'))
        # G.update_all(fn.copy_src('h', 'h'), fn.prod('h', 'h_agg'))

        # G.apply_nodes(更新函数， 更新的节点) 更新节点默认为图中所有节点
        G.apply_nodes(self.prescription_update, self.prescription_nodes)
        G.apply_nodes(self.symptom_update, self.symptom_nodes)


class NodeUpdate(nn.Block):
    def __init__(self, feature_size, dropout, alpha):
        super(NodeUpdate, self).__init__()

        self.feature_size = feature_size
        self.leakyrelu = nn.LeakyReLU(alpha)
        # self.leakyrelu = nn.PReLU()
        self.W = nn.Dense(feature_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes):
        h = nodes.data['h']
        h_agg = nodes.data['h_agg']
        deg = nodes.data['deg'].expand_dims(1)

        # h_concat = nd.concat(h, h_agg / nd.maximum(deg, 1e-6), dim=1)
        h_concat = nd.concat(h, h_agg, dim=1)
        # h_concat = nd.concat(h, h_agg, dim=1)
        h_new = self.dropout(self.leakyrelu(self.W(h_concat)))
        # h_new = self.dropout(self.leakyrelu(self.W()))
        return {'h': h_new}
