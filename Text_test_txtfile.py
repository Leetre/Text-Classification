import jieba, re
import numpy as np
import get_data as raw
import Text_train
import tensorflow as tf

# Test_CNN_NET is almost the same as CNN_NET
class Test_CNN_NET:
    def __init__(self, config):
        # config parameter
        self.config = config
        self.word_embedding_length = self.config.word_embedding_length
        self.sentence_length = self.config.sentence_length
        self.filter_sizes = self.config.filter_size
        self.num_class = self.config.num_class
        self.regularization_rate = self.config.regularization_rate

        # add placeholder
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.sentence_length, self.word_embedding_length], name='input_x')
        self.input_label = tf.placeholder(dtype=tf.int32, shape=[None, self.num_class], name='input_label')  # input labels vector
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        self.regularize = tf.contrib.layers.l2_regularizer(self.regularization_rate)

        # inference_result and accuracy
        self.prediction = self.inference()
        self.accuracy = self.cal_acc()

    def inference(self):
        output_list = list()
        sentence_data_reshaped = tf.reshape(self.input_x, [-1, self.sentence_length, self.word_embedding_length, 1])
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope('filter{}'.format(filter_size)):
                # convolution layer
                with tf.variable_scope('conv'):
                    conv_w = tf.get_variable('conv_w', [filter_size, self.word_embedding_length, 1, 1], initializer=tf.random_normal_initializer(stddev=0.1))
                    conv_b = tf.get_variable('conv_b', [1], initializer=tf.constant_initializer(0.1))
                    conv = tf.nn.conv2d(sentence_data_reshaped, conv_w, strides=[1,1,1,1], padding='VALID')
                    output_conv = tf.nn.relu(tf.nn.bias_add(conv, conv_b))
                # max_pooling layer
                with tf.name_scope('pool'):
                    output_pool = tf.nn.max_pool(output_conv, ksize=[1,self.sentence_length-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID')

                output_list.append(output_pool)
        # tf.concat() can get [None, 1, 1, concat_dim], we use tf.reshape() to get [None, concat_dim]
        concat_dim = len(self.filter_sizes)
        conv_total_out = tf.reshape(tf.concat(output_list, 3), [-1, concat_dim])
        #drop_out layer
        with tf.name_scope('drop_out'):
            conv_total_drop = tf.nn.dropout(conv_total_out, self.keep_prob)
        # fully connected layer
        with tf.variable_scope('fc'):
            fc_w = tf.get_variable('fc_w', [concat_dim, self.num_class], initializer=tf.random_normal_initializer(stddev=0.1))
            if self.regularize != None:
                tf.add_to_collection('losses', self.regularize(fc_w))
            fc_b = tf.get_variable('fc_b', [self.num_class], initializer=tf.constant_initializer(0.1))
            fc_out = tf.matmul(conv_total_drop, fc_w) + fc_b
            prediction = tf.nn.softmax(fc_out)
        return prediction

    def cal_acc(self):
        # Use argmax function to get index of the largest item in each line of the matrix and to check if it equals to the standard.
        # Compute average value of matrix as the accuracy.
        correct_prediction = tf.equal(tf.argmax(self.input_label, 1), tf.argmax(self.prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return accuracy

def test():
    # Construct a Test_CNN_NET obj.
    text_cnn = Test_CNN_NET(Text_train.config)
    with tf.Session() as sess:
        # Load the model.
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./old_model/')
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Get sentences and convert them into vector
        for i in range(text_cnn.config.test_steps):
            num = i
            # Load training data and labels
            content_file = open('./jieba_treat_test/' + str(num) + '.txt', 'r')
            label_file = open('./jieba_treat_test/' + str(num) + '_l.txt', 'r')
            all_sentence_words = raw.get_all_words(content_file)
            embeddings = raw.get_embeddings(all_sentence_words)
            embeddings = np.array(embeddings)
            labels = raw.get_labels(label_file)
            labels = np.array(labels)
            # Feed the data into the network
            feed_dict = {
                text_cnn.input_x: embeddings,
                text_cnn.input_label: labels,
                text_cnn.keep_prob: text_cnn.config.keep_prob
            }
            # Compute the loss and accuracy
            acc = sess.run(text_cnn.accuracy, feed_dict)
            print("step {}, acc {:g}".format(i, acc))
            content_file.close()
            label_file.close()

if __name__ == '__main__':
    test()