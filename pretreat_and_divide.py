import jieba, re

def pretreat(data, save_path):
    i = 0
    for lines in data.readlines():
        # divide the each line of the data into twp parts: label and content
        label_and_content = lines.split('\t')
        label = label_and_content[0]
        # convert the label into vector
        label_vec = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
        if label == '体育':
            label_vec = ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0']
        elif label == '财经':
            label_vec = ['0', '1', '0', '0', '0', '0', '0', '0', '0', '0']
        elif label == '房产':
            label_vec = ['0', '0', '1', '0', '0', '0', '0', '0', '0', '0']
        elif label == '家居':
            label_vec = ['0', '0', '0', '1', '0', '0', '0', '0', '0', '0']
        elif label == '教育':
            label_vec = ['0', '0', '0', '0', '1', '0', '0', '0', '0', '0']
        elif label == '科技':
            label_vec = ['0', '0', '0', '0', '0', '1', '0', '0', '0', '0']
        elif label == '时尚':
            label_vec = ['0', '0', '0', '0', '0', '0', '1', '0', '0', '0']
        elif label == '时政':
            label_vec = ['0', '0', '0', '0', '0', '0', '0', '1', '0', '0']
        elif label == '游戏':
            label_vec = ['0', '0', '0', '0', '0', '0', '0', '0', '1', '0']
        elif label == '娱乐':
            label_vec = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '1']
        # use re to get rid of some specific symbol
        content = label_and_content[1]
        content = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "", content)
        content = re.sub(r"[^\u4e00-\u9fff]", " ", content)
        # use jieba to split sentences as a list
        content_l = jieba.lcut(content)
        # save sentences into files, and one file consists of 100 sentences
        num = int(i/100)
        file_content = open(save_path+str(num)+'.txt', 'a')
        file_label = open(save_path+str(num)+'_l.txt', 'a')
        seg = ','
        file_content.write(seg.join(content_l) + '\n')  # convert list into string and save it
        file_label.write(seg.join(label_vec) + '\n')
        file_content.close()
        file_label.close()
        i = i + 1

if __name__ == '__main__':
    save_path = './jieba_treat/'
    train_data = open('disorganize_train.txt', 'r+', encoding='utf8')
    pretreat(train_data, save_path)

    # save_path = './jieba_treat_test/'
    # test_data = open('cnews_test.txt', 'r+', encoding='utf8')
    # pretreat(test_data, save_path)
