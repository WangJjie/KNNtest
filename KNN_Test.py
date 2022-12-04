import numpy as np
import matplotlib.pyplot as plt

from intro import load_cifar10

class KnnNearestNeighbour(object):
    def __init__(self, data_train, data_label, data_test):
        self.data_train = data_train
        self.data_label = data_label
        self.data_test = data_test

    def predict(self, k=1):
        #num_test = 10000
        num_test = self.data_test.shape[0]
        label_predict = []
        for i in range(num_test):
            """使用l1距离"""
            distance = np.sum(np.abs(self.data_train - self.data_test[i, :]), axis=1) ##此处运用了array的广播机制
            closest_index = np.argsort(distance) ##排序
            label_area = self.data_label[closest_index]
            label_area_final = label_area[:k]  ##筛选出前k个
            label_predict.append(np.argmax(np.bincount(label_area_final)))
            if i%100 == 0:
                print('这是k={}'.format(k))
                print('运行到第{}步'.format(i))

        return label_predict

batchfile = "E:/graduateWorks/CodeExperience/PythonCode/KNN/cifar-10-batches-py/"
[data_train, label_train, data_test, label_test] = load_cifar10(batchfile)

train_num = 20000
test_num = 5000

data_train = data_train[:train_num, ]
label_train = label_train[:train_num, ]
data_test = data_test[:test_num, ]
label_test = label_test[:test_num, ]



nn = KnnNearestNeighbour(data_train, label_train, data_test)
print(data_test.shape[0])




k_choice = [1, 3, 5, 7, 9]




"""


"""

label_predict = np.zeros((len(k_choice), test_num))
num = 0
for k in k_choice:
    predict_temp = nn.predict(k)
    label_predict[num,:] = predict_temp
    num = num + 1


print(label_predict.shape)


accuracy = np.zeros((len(k_choice),1))
for i in range(len(k_choice)):
    accuracy[i,:] = np.mean(label_test == label_predict[i,:])

plt.errorbar(k_choice, accuracy)
plt.title('accuracy on k')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()



"""
num = 0

for i in k_choice:
    num = num + 1
    if num == 1:
        label_predict1 = nn.predict(i)
    elif num == 2:
        label_predict2 = nn.predict(i)
    elif num == 3:
        label_predict3 = nn.predict(i)
    elif num == 4:
        label_predict4 = nn.predict(i)
    else:
        label_predict5 = nn.predict(i)

"""









