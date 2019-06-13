import jieba, re
import numpy as np
import get_data, Text_train
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
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        self.regularize = tf.contrib.layers.l2_regularizer(self.regularization_rate)

        # inference_result
        self.prediction = self.inference()

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
        # Use argmax function to treat the result of the neural network and get a simple labels representation
        prediction = tf.argmax(prediction, 1)
        return prediction

# Input a sentence
def get_input():
    input_sentence = input('Please enter a sentence: ')
    return input_sentence

# use jieba to split sentences
def jieba_cut(sentences):
    all_sentence_words = []
    for sentence in sentences:
        sentence = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "", sentence)
        sentence = re.sub(r"[^\u4e00-\u9fff]", " ", sentence)
        sentence_words = jieba.lcut(sentence)
        all_sentence_words.append(sentence_words)
    return all_sentence_words

# convert sentences into vector
def get_embeddings(all_sentence_words, max_length=500):
    all_words_vec = get_data.get_embeddings(all_sentence_words, max_length)
    return all_words_vec

def test():
    # Construct a Test_CNN_NET obj.
    text_cnn = Test_CNN_NET(Text_train.config)
    with tf.Session() as sess:
        # Load the model.
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./old_model/')
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Get sentences and convert them into vector
        sentences = []
        input_sentence = get_input()
        sentences.append(input_sentence)
        all_sentence_words = jieba_cut(sentences)
        embeddings = get_embeddings(all_sentence_words)
        embeddings = np.array(embeddings)
        # Feed all data into network
        feed_dict = {
            text_cnn.input_x: embeddings,
            text_cnn.keep_prob: text_cnn.config.keep_prob
        }
        # Get the result
        prediction = sess.run(text_cnn.prediction, feed_dict)
        # Make decisions according to prediction
        Class = ''
        if prediction[0] == 0:
            Class = '体育'
        elif prediction[0] == 1:
            Class = '财经'
        elif prediction[0] == 2:
            Class = '房产'
        elif prediction[0] == 3:
            Class = '家居'
        elif prediction[0] == 4:
            Class = '教育'
        elif prediction[0] == 5:
            Class = '科技'
        elif prediction[0] == 6:
            Class = '时尚'
        elif prediction[0] == 7:
            Class = '时政'
        elif prediction[0] == 8:
            Class = '游戏'
        elif prediction[0] == 9:
            Class = '娱乐'

        print('The result is {}'.format(Class))

if __name__ == '__main__':
    test()