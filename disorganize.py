import random

# store each line of the data in a list respectively
def repermulate(input_data, output_data):
    str_list = []
    for lines in input_data.readlines():
        str_list.append(lines)

    # mess up the order of items in the list
    random.shuffle(str_list)

    # write the data back to a new file
    for item in str_list:
        output_data.write(item)

if __name__ == '__main__':
    train_data = open('cnews_train.txt', 'r+', encoding='utf8')
    output_data = open('disorganize_train.txt', 'w+', encoding='utf8')
    repermulate(train_data, output_data)