# -*- coding: utf-8 -*-
import data_helpers
from qalstm import QALSTM
import tensorflow as tf


def main():
    trained_model = "checkpoints/model.ckpt"
    embedding_size = 100  # Word embedding dimension
    epochs = 10
    batch_size = 64  # Batch data size
    rnn_size = 50  # Number of hidden layer neurons
    sequence_length = 300  # Sentence length
    learning_rate = 0.01  # Learning rate
    lrdownRate = 0.9
    margin = 0.1
    attention_matrix_size = 100
    gpu_mem_usage = 0.75
    # gpu_device = "/gpu:0"
    cpu_device = "/cpu:0"

    embeddings, word2idx = data_helpers.load_embedding('vectors.nobin')
    voc = data_helpers.load_vocab('D:\\DataMining\\Datasets\\insuranceQA\\V1\\vocabulary')
    all_answers = data_helpers.load_answers('D:\\DataMining\\Datasets\\insuranceQA\\V1\\answers.label.token_idx', voc)
    questions, pos_answers, neg_answers = data_helpers.load_train_data('D:\\DataMining\\Datasets\\insuranceQA\\V1\\question.train.token_idx.label', all_answers, voc, word2idx, sequence_length)

    with tf.Graph().as_default(), tf.device(cpu_device):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_usage)
        # session_conf = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        model = QALSTM(batch_size, sequence_length, embeddings, embedding_size, rnn_size, margin, attention_matrix_size)
        with tf.Session().as_default() as sess:  # config=session_conf
            saver = tf.train.Saver()

            print("Start training")
            sess.run(tf.global_variables_initializer())  # Initialize all variables
            for epoch in range(epochs):
                print("The training of the %s iteration is underway" % (epoch + 1))
                for question, pos_answer, neg_answer in data_helpers.batch_iter(questions, pos_answers, neg_answers, batch_size):
                    feed_dict = {
                        model.q: question,
                        model.ap: pos_answer,
                        model.an: neg_answer,
                        model.lr: learning_rate
                    }
                    _, loss, acc = sess.run([model.train_op, model.loss, model.acc], feed_dict)
                    print("loss:%s\tacc:%s" % (loss, acc))
                learning_rate *= lrdownRate
            print("End of the training")
            saver.save(sess, trained_model)


if __name__ == '__main__':
    main()
