import numpy as np

import pickle



print("hello world")


"cifar-10 expression"

cifar = open('E:\graduateWorks\CodeExperience\PythonCode\KNN\cifar-10-batches-py\data_batch_1','rb')
dict1 = pickle.load(cifar, encoding='bytes')
# print(dict1.items())
print(dict1[b'data'].shape)
print(dict1[b'data'])


batchfile = "E:/graduateWorks/CodeExperience/PythonCode/KNN/cifar-10-batches-py/"

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        #fo.close()
    return dict

def load_cifar10(file):

    "confusion data_batch"


    data_train = []
    label_train = []
    for i in range(1,6):
        train_dict = unpickle(file + 'data_batch_' + str(i))
        for i_datas in train_dict[b'data']:
            data_train.append(i_datas)
        for i_lables in train_dict[b'labels']:
            label_train.append(i_lables)

    "confusion test_batch"

    data_test = []
    label_test = []
    test_filepath = file + 'test_batch'
    test_dict = unpickle(test_filepath)
    for test_datas in test_dict[b'data']:
        data_test.append(test_datas)
    for test_labels in test_dict[b'labels']:
        label_test.append(test_labels)

    return np.array(data_train), np.array(label_train), np.array(data_test), np.array(label_test)


[data_train, label_train, data_test, label_test] = load_cifar10(batchfile)

print(data_train.shape)
print(data_test.shape)
print(label_train.shape)
print(label_test.shape)






















"""

digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
def char2num(s):
    digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    return digits[s]


print(list(map(char2num, ('1','3','5','7','9'))))
print(list(map(char2num, ['1','3','5','7','9'])))
print(list(map(char2num, {'1','3','5','7','9'})))
print(list(map(char2num, '13579')))

"""




