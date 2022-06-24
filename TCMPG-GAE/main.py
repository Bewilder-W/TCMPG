import warnings
import mxnet as mx
from predict_prescription import Train_with_embedding, train


def main():
    warnings.filterwarnings("ignore")
    original_path = './data/'
    train_index, valid_index, matrix_p_s, emb_p, symptom_emb_path = Train_with_embedding(directory='data',
                                                                                         epochs=20,
                                                                                         batchsize=10,
                                                                                         embedding_size=200,
                                                                                         layers=1,
                                                                                         dropout=0.1,
                                                                                         alpha=0,
                                                                                         lr=0.001,
                                                                                         wd=0,
                                                                                         random_seed=123,
                                                                                         ctx=mx.gpu(0),
                                                                                         original_path=original_path)

    train(train_index, valid_index, matrix_p_s, emb_p, symptom_emb_path, flag=False)


if __name__ == '__main__':
    main()

