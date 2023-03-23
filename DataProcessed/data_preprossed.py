import numpy as np
from scipy import sparse
import csv


def generate_csv():
    path_matrix_p_s = './data/p_s_adj_matrix.npz'
    # path_feature_p = './data/feature_p.npz'
    # path_feature_i = './data/feature_i.npz'
    matrix_p_s = sparse.load_npz(path_matrix_p_s)
    # feature_p = sparse.load_npz(path_feature_p)
    # feature_object = np.load(path_feature_i)
    # feature_i = feature_object["feature_i"]
    matrix_p_s = matrix_p_s.toarray()
    # feature_p = feature_p.toarray()

    pair_list = []
    for i in range(matrix_p_s.shape[0]):
        for j in range(matrix_p_s.shape[1]):
            if matrix_p_s[i][j] == 1:
                tupl = (i+1, j+1, 1)
            else:
                tupl = (i+1, j+1, 0)
            pair_list.append(tupl)

    # pair = []
    # for i in range(1):
    #     for j in range(matrix_p_s.shape[1]):
    #         if matrix_p_s[i+20998][j] == 1:
    #             print(j)
    #             tupl = (i+20999, j+1, 1)
    #         else:
    #             tupl = (i+20999, j+1, 0)
    #         pair.append(tupl)

    with open('./data/all_prescription_symptom_pairs.csv', 'w', encoding='utf-8', newline='') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerows(pair_list)
    #
    # with open('./data/new_all_prescription_symptom_pairs.csv', 'w', encoding='utf-8', newline='') as file_obj:
    #     writer = csv.writer(file_obj)
    #     writer.writerows(pair)


if __name__ == '__main__':
    generate_csv()