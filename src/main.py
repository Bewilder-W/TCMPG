import warnings
from Functions import *
import mxnet as mx
from predict_prescription import Train_without_embedding, Train_with_embedding, train, test, before_test, without_encoder
# from test_predict_prescription import  Train_without_embedding, before_test


def main(original_path):
    warnings.filterwarnings("ignore")

    train_index, valid_index, matrix_p_s, feature_p, emb_p, symptom_emb_path = Train_with_embedding(directory='data',
                                                                                         embedding_size=300,
                                                                                         layers=1,
                                                                                         dropout=0.5,
                                                                                         alpha=0,
                                                                                         lr=0.001,
                                                                                         wd=0.1,
                                                                                         random_seed=123,
                                                                                         ctx=mx.gpu(0),
                                                                                         original_path=original_path)

    # train_index, valid_index, matrix_p_s, feature_p, emb_p, symptom_emb_path = Train_without_embedding(random_seed=123,
    #                                                                                         original_path=original_path)
    train(train_index, valid_index, matrix_p_s, feature_p, emb_p, symptom_emb_path, original_path, flag=False)
    # without_encoder(original_path=original_path, random_seed=123, flag=False)


def test_main(original_path):
    warnings.filterwarnings("ignore")
    train_index, matrix_p_s, feature_p, emb_p = before_test(directory='data',
                                                             embedding_size=300,
                                                             layers=1,
                                                             dropout=0.5,
                                                             alpha=0,
                                                             lr=0.001,
                                                             wd=0.1,
                                                             random_seed=123,
                                                             ctx=mx.gpu(0),
                                                             original_path=original_path)
    # test(train_index, matrix_p_s, feature_p, emb_p, original_path)


if __name__ == '__main__':

    original_path = './data/trained_symptom_emb/'
    # make_print_to_file(original_path, p='Encoder')
    make_print_to_file(original_path, p='Experiment')
    main(original_path=original_path)
    # test_main(original_path=original_path)
