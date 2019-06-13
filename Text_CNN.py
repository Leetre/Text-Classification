import tensorflow as tf

class CNN_NET:
    def __init__(self, word_embedding_length, sentence_length, learning_rate, filter_sizes, num_class, regularization_rate):
        # config parameter
        self.word_embedding_length = word_embedding_length  # length of a word vector
        self.sentence_length = sentence_length  # the num of words in one sentence
        self.learning_rate = learning_rate  # learning rate for training
        self.filter_sizes = filter_sizes # all of the filter sizes, represented as a list
        self.num_class = num_class  # the number of class
        self.regularization_rate = regularization_rate  # regularization factor

        # add placeholder
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.sentence_length, self.word_embedding_length], name='input_x') # input sentences in the form of word embeddings
        self.input_label = tf.placeholder(dtype=tf.int32, shape=[None, self.num_class], name='input_label') # input labels vector
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob') # to decide drop out, prevent overfitting(drop_out = 1 - keep_prob)

        self.regularize = tf.contrib.layers.l2_regularizer(self.regularization_rate)    # regularization for preventing overfitting

        # train_result
        self.prediction = self.inference()  # get the prediction(the output of the neural network n*10)
        self.loss = self.add_loss() # loss value
        self.train_op = self.optimize() # train step
        self.accuracy = self.cal_acc()  # compute accuracy of each train step

    def inference(self):
        output_list = list()
        sentence_data_reshaped = tf.reshape(self.input_x, [-1, self.sentence_length, self.word_embedding_length, 1])    # reshape input tensor
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope('filter{}'.format(filter_size)):
                # convolution layer
                with tf.variable_scope('conv'):
                    # Initialize weight and bias of convolution layer
                    conv_w = tf.get_variable('conv_w', [filter_size, self.word_embedding_length, 1, 1], initializer=tf.random_normal_initializer(stddev=0.1))
                    conv_b = tf.get_variable('conv_b', [1], initializer=tf.constant_initializer(0.1))
                    # Implement convolution step
                    conv = tf.nn.conv2d(sentence_data_reshaped, conv_w, strides=[1,1,1,1], padding='VALID')
                    # Add bias and use relu function as the activation function
                    output_conv = tf.nn.relu(tf.nn.bias_add(conv, conv_b))
                # max_pooling layer
                with tf.name_scope('pool'):
                    output_pool = tf.nn.max_pool(output_conv, ksize=[1,self.sentence_length-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID')
                # to record all the results by using each filter treated
                output_list.append(output_pool)
        # tf.concat() can get [None, 1, 1, concat_dim], we use tf.reshape() to get [None, concat_dim]
        concat_dim = len(self.filter_sizes)
        conv_total_out = tf.reshape(tf.concat(output_list, 3), [-1, concat_dim])
        #drop_out layer
        with tf.name_scope('drop_out'):
            conv_total_drop = tf.nn.dropout(conv_total_out, self.keep_prob)
        # fully connected layer
        with tf.variable_scope('fc'):
            # Initialize weight and bias of fully connected layer and add regularization
            fc_w = tf.get_variable('fc_w', [concat_dim, self.num_class], initializer=tf.random_normal_initializer(stddev=0.1))
            if self.regularize != None:
                tf.add_to_collection('losses', self.regularize(fc_w))
            fc_b = tf.get_variable('fc_b', [self.num_class], initializer=tf.constant_initializer(0.1))
            fc_out = tf.matmul(conv_total_drop, fc_w) + fc_b
            # Use softmax function to Normalize
            prediction = tf.nn.softmax(fc_out)
        return prediction

    def add_loss(self, l2_lambda=0.0001):
        with tf.name_scope('loss'):
            # compute cross entropy by using standard labels and predicted labels
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.input_label, 1), logits=self.prediction)
            cross_entropy_mean = tf.reduce_mean(cross_entropy)  # compute average value of the matrix
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            total_loss = cross_entropy_mean + l2_losses
        return total_loss

    def optimize(self):
        with tf.name_scope('train_op'):
            # Use adam optimizer to optimal loss function
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        return train_op

    def cal_acc(self):
        # Use argmax function to get index of the largest item in each line of the matrix and to check if it equals to the standard.
        # Compute average value of matrix as the accuracy.
        correct_prediction = tf.equal(tf.argmax(self.input_label, 1), tf.argmax(self.prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return accuracy