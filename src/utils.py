import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import ndarray as nd
import dgl
from scipy import sparse


def load_data():
    """
    :return: 返回numpy格式的特征矩阵p和s
    """
    path_feature_p = './data/feature_p.npz'
    path_feature_s = './data/feature_s.npz'
    feature_p = sparse.load_npz(path_feature_p)
    feature_object = np.load(path_feature_s)
    feature_s = feature_object["feature_s"]
    feature_p = feature_p.toarray()

    return feature_p, feature_s


def load_data_none_w2v():
    """
    :return: 返回numpy格式的特征矩阵p和s
    """
    path_feature_p = './data/feature_p.npz'
    feature_p = sparse.load_npz(path_feature_p)
    feature_p = feature_p.toarray()

    path_feature_s = './data/feature_s_none_w2v.npz'
    feature_s = sparse.load_npz(path_feature_s)
    feature_s = feature_s.toarray()

    return feature_p, feature_s


def load_data_none_dose():
    """
    :return: 返回numpy格式的特征矩阵p和s
    """
    path_feature_p = './data/feature_p_none_dose.npz'
    feature_p = sparse.load_npz(path_feature_p)
    feature_p = feature_p.toarray()

    path_feature_s = './data/feature_s.npz'
    feature_object = np.load(path_feature_s)
    feature_s = feature_object["feature_s"]

    return feature_p, feature_s


def load_data_none_both():
    """
    :return: 返回numpy格式的特征矩阵p和s
    """
    path_feature_p = './data/feature_p_none_dose.npz'
    feature_p = sparse.load_npz(path_feature_p)
    feature_p = feature_p.toarray()

    path_feature_s = './data/feature_s_none_w2v.npz'
    feature_s = sparse.load_npz(path_feature_s)
    feature_s = feature_s.toarray()

    return feature_p, feature_s


def sample(directory, random_seed):
    all_associations = pd.read_csv(directory + '/all_prescription_symptom_pairs.csv', names=['prescription', 'symptom', 'label'])
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)

    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)

    return sample_df.values


# 新方子治疗症状
def case_sample1(directory, random_seed):
    all_associations = pd.read_csv(directory + '/all_prescription_symptom_pairs.csv', names=['prescription', 'symptom', 'label'])
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)
    sample_df = known_associations.append(random_negative)

    new_associations = pd.read_csv(directory + '/new_all_prescription_symptom_pairs.csv', names=['prescription', 'symptom', 'label'])
    sample_df = sample_df.append(new_associations)

    sample_df.reset_index(drop=True, inplace=True)

    return sample_df.values


# 症状预测新方子
def case_sample2(directory, random_seed, symptom_index):
    all_associations = pd.read_csv(directory + '/all_prescription_symptom_pairs.csv',
                                   names=['prescription', 'symptom', 'label'])
    known_associations = all_associations.loc[all_associations['label'] == 1]

    symptoms_unknown_associatios = all_associations.iloc[:, :][(all_associations.symptom == symptom_index[0]) & (all_associations.label == 0)]
    for i in range(len(symptom_index) - 1):
        symptom_unknown_associatios = all_associations.iloc[:, :][(all_associations.symptom == symptom_index[i + 1]) & (all_associations.label == 0)]
        symptoms_unknown_associatios = symptoms_unknown_associatios.append(symptom_unknown_associatios)
    unknown_associations = all_associations.iloc[:, :][(all_associations.label == 0) & (all_associations.symptom not in symptom_index)]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)

    sample_df = known_associations.append(random_negative)
    sample_df = sample_df.append(symptoms_unknown_associatios)
    sample_df.reset_index(drop=True, inplace=True)

    return sample_df.values


def build_graph(directory, random_seed, ctx):
    # dgl.load_backend('mxnet')
    feature_p, feature_s = load_data()
    # feature_p, feature_s = load_data_none_w2v()
    # feature_p, feature_s = load_data_none_dose()
    # feature_p, feature_s = load_data_none_both()
    samples = sample(directory, random_seed)

    print('Building graph ...')
    g = dgl.DGLGraph(multigraph=True)
    g.add_nodes(feature_p.shape[0] + feature_s.shape[0])
    node_type = nd.zeros(g.number_of_nodes(), dtype='float32', ctx=ctx)
    node_type[:feature_p.shape[0]] = 1
    g.ndata['type'] = node_type

    print('Adding prescription features ...')
    p_data = nd.zeros(shape=(g.number_of_nodes(), feature_p.shape[1]), dtype='float32', ctx=ctx)
    p_data[: feature_p.shape[0], :] = nd.from_numpy(feature_p)
    g.ndata['p_features'] = p_data
    print("## prescription feature dimensions: " + str(p_data.shape))

    print('Adding symptom features ...')
    s_data = nd.zeros(shape=(g.number_of_nodes(), feature_s.shape[1]), dtype='float32', ctx=ctx)
    s_data[feature_p.shape[0]: feature_p.shape[0]+feature_s.shape[0], :] = nd.from_numpy(feature_s)
    g.ndata['s_features'] = s_data
    print("## symptom feature dimensions: " + str(s_data.shape))

    print('Adding edges ...')
    prescription_ids = list(range(1, feature_p.shape[0] + 1))
    symptom_ids = list(range(1, feature_s.shape[0] + 1))

    prescription_ids_invmap = {id_: i for i, id_ in enumerate(prescription_ids)}
    symptom_ids_invmap = {id_: i for i, id_ in enumerate(symptom_ids)}

    sample_prescription_vertices = [prescription_ids_invmap[id_] for id_ in samples[:, 0]]
    sample_symptom_vertices = [symptom_ids_invmap[id_] + feature_p.shape[0] for id_ in samples[:, 1]]

    g.add_edges(sample_prescription_vertices, sample_symptom_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
    g.add_edges(sample_symptom_vertices, sample_prescription_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
    #
    # # 删除边的label信息
    # g.add_edges(sample_prescription_vertices, sample_symptom_vertices,
    #             data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx)})
    # g.add_edges(sample_symptom_vertices, sample_prescription_vertices,
    #             data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx)})

    g.readonly()
    print('Successfully build graph !!')

    return g, sample_prescription_vertices, sample_symptom_vertices, feature_p, feature_s, samples

