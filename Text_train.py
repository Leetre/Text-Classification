from Text_CNN import CNN_NET
import tensorflow as tf
import numpy as np
import get_data as raw

loss_file = open('loss.txt', 'w')
acc_file = open('acc.txt', 'w')
# Define parameters
flags, FLAGS = tf.app.flags, tf.app.flags.FLAGS
flags.DEFINE_integer('word_embedding_length', 300, 'length of each word')
flags.DEFINE_integer('sentence_length', 500, 'sentence length')
flags.DEFINE_integer('num_class', 10, 'the num of class')
flags.DEFINE_integer('num_epochs', 20, 'epochs')
flags.DEFINE_integer('steps', 500, 'total steps in one epoch')
flags.DEFINE_integer('test_steps', 100, 'test steps')
flags.DEFINE_float('regularization_rate', 0.0001, 'regularization rate')
flags.DEFINE_float('l2_reg_lambda', 0.0001, 'L2 regularization factor')
flags.DEFINE_float('keep_prob', 1.0, 'Dropout keep rate')
flags.DEFINE_float('lr', 0.001, 'learning rate')
MODELSAVEPATH = './model/'

class Config:
    def __init__(self):
        # Config parameters
        self.word_embedding_length = FLAGS.word_embedding_length
        self.sentence_length = FLAGS.sentence_length
        self.num_class = FLAGS.num_class
        self.num_epochs = FLAGS.num_epochs
        self.steps = FLAGS.steps
        self.test_steps = FLAGS.test_steps
        self.regularization_rate = FLAGS.regularization_rate
        self.l2_reg_lambda = FLAGS.l2_reg_lambda
        self.keep_prob = FLAGS.keep_prob
        self.learning_rate = FLAGS.lr
        self.filter_size = [2, 3, 4, 5]

config = Config()
def train():
    # Construct a CNN_NET obj.
    text_cnn = CNN_NET(config.word_embedding_length, config.sentence_length, config.learning_rate,
                       config.filter_size, config.num_class, config.regularization_rate)
    with tf.Session() as sess:
        # Define saver and constraint the num of models can save.
        saver = tf.train.Saver(max_to_keep=5)
        # Initialize all of the parameters.
        sess.run(tf.global_variables_initializer())
        for i in range(config.num_epochs):
            print('epoch {}'.format(i))
            for j in range(config.steps):
                num = j
                # Load training data and labels
                content_file = open('./jieba_treat/'+str(num)+'.txt', 'r')
                label_file = open('./jieba_treat/'+str(num)+'_l.txt', 'r')
                all_sentence_words = raw.get_all_words(content_file)
                embeddings = raw.get_embeddings(all_sentence_words)
                embeddings = np.array(embeddings)
                labels = raw.get_labels(label_file)
                labels = np.array(labels)
                # Feed the data into the network
                feed_dict = {
                    text_cnn.input_x: embeddings,
                    text_cnn.input_label: labels,
                    text_cnn.keep_prob: config.keep_prob
                }
                # Compute the loss and accuracy
                loss, _, acc = sess.run([text_cnn.loss, text_cnn.train_op, text_cnn.accuracy], feed_dict)
                loss_file.write(str(loss)+'\n')
                acc_file.write(str(acc)+'\n')
                print("step {}, loss {:g}, acc {:g}".format(j, loss, acc))
                content_file.close()
                label_file.close()
            # Save models
            saver.save(sess, MODELSAVEPATH+'epoch_'+str(i)+'.ckpt')

if __name__ == '__main__':
    train()
