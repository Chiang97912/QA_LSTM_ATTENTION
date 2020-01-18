# -*- coding: utf-8 -*-
import tensorflow as tf
import data_helpers
from qalstm import QALSTM
from utils import eval_map_mrr


def main():
    trained_model = "checkpoints/model.ckpt"
    embedding_size = 100  # Word embedding dimension
    batch_size = 128  # Batch data size
    sequence_length = 300  # Sentence length
    rnn_size = 50  # Number of hidden layer neurons
    attention_matrix_size = 100
    margin = 0.1
    gpu_mem_usage = 0.75
    gpu_device = "/gpu:0"

    embeddings, word2idx = data_helpers.load_embedding('vectors.nobin')
    voc = data_helpers.load_vocab('D:\\DataMining\\Datasets\\insuranceQA\\V1\\vocabulary')
    all_answers = data_helpers.load_answers('D:\\DataMining\\Datasets\\insuranceQA\\V1\\answers.label.token_idx', voc)
    questions, answers, labels, qids, aids = data_helpers.load_test_data('D:\\DataMining\\Datasets\\insuranceQA\\V1\\question.test1.label.token_idx.pool', all_answers, voc, word2idx, 300)
    with tf.Graph().as_default(), tf.device(gpu_device):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_usage)
        session_conf = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        model = QALSTM(batch_size, sequence_length, embeddings, embedding_size, rnn_size, margin, attention_matrix_size)
        with tf.Session(config=session_conf).as_default() as sess:  # config=session_conf
            saver = tf.train.Saver()
            print("Start loading the model")
            saver.restore(sess, trained_model)
            print("The model is loaded")
            scores = []
            for question, answer in data_helpers.test_batch_iter(questions, answers, batch_size):
                feed_dict = {
                    model.qtest: question,
                    model.atest: answer
                }
                score = sess.run([model.scores], feed_dict)
                scores.extend(score[0].tolist())
            MAP, MRR = eval_map_mrr(qids, aids, scores, labels)
            print('MAP %2.3f\tMRR %2.3f' % (MAP, MRR))


if __name__ == '__main__':
    main()
