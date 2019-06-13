import re
import gensim
# load word2vec model
model = gensim.models.KeyedVectors.load_word2vec_format('./w2v_vec/sgns.wiki.word',binary=False)

# This function is to get all sentences in the form of list.
def get_all_words(content_file):
    all_sentence_words = []
    for lines in content_file.readlines():
        sentence_words = lines.split(',')
        all_sentence_words.append(sentence_words)
    return all_sentence_words

# This function is to convert all words in sentences into vector by using w2v model
# and adjust the num of words in a sentence by padding or cutting off.
def get_embeddings(all_sentence_words, max_length=500):
    list_pad = [0]*300  # vector which is filled with 0, used to pad
    all_words_vec = []
    print('get embeddings...')
    for words in all_sentence_words:
        tmp_words_vec = []
        num_need_to_pad = max_length - len(words)   # compute if the number of words in the sentence need to pad or cut off
        if num_need_to_pad > 0: # need to pad
            for word in words:
                word = re.sub('[\n]', '', word) # get rid of '\n' in a word
                if word in model:
                    tmp_words_vec.append(model[word].tolist()) # convert numpy array to list for appending operation
                else:
                    tmp_words_vec.append(list_pad)
            for i in range(num_need_to_pad):
                tmp_words_vec.append(list_pad)
        elif num_need_to_pad <= 0:  # need to cut off
            for num in range(max_length):
                word = words[num]
                word = re.sub('[\n]', '', word) # get rid of '\n' in a word
                if word in model:
                    tmp_words_vec.append(model[word].tolist())  # convert numpy array to list for appending operation
                else:
                    tmp_words_vec.append(list_pad)
        all_words_vec.append(tmp_words_vec)
    return all_words_vec

# This function is used to get labels which are stored in txt files
def get_labels(label_file):
    all_labels = []
    for lines in label_file.readlines():
        str_list = lines.split(',')
        label = [int(item) for item in str_list]
        all_labels.append(label)
    return all_labels

if __name__ == '__main__':
    content_file = open('./jieba_treat/11.txt', 'r')
    label_file = open('./jieba_treat/11_l.txt', 'r')
    # all_sentence_words = get_all_words(content_file)
    # embeddings = get_embeddings(all_sentence_words)
    # for item in embeddings:
    #     print(len(item))
    # embeddings = np.array(embeddings)
    label = get_labels(label_file)
    for i in range(100):
        print(label[i])
    # print(embeddings.shape)