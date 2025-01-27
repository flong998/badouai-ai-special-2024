import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_imgs, train_labels), (test_imgs, test_labels) = mnist.load_data()
print("train_imgs.shape = ", train_imgs.shape)
print("test_imgs.shape = ", test_imgs.shape)
print("train_labels.shape = ", train_labels.shape)
print("test_labels.shape = ", test_labels.shape)

# 定义网络
net = models.Sequential()
net.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
net.add(layers.Dense(10, activation="softmax"))

net.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# 预处理数据
train_imgs = (train_imgs.reshape((60000, 28 * 28)).astype("float32")) / 255
test_imgs = (test_imgs.reshape((10000, 28 * 28)).astype("float32")) / 255

print("before change:", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])

# 训练
net.fit(train_imgs, train_labels, epochs=5, batch_size=128)

# 计算损失与准确率
test_loss, test_acc = net.evaluate(test_imgs, test_labels, verbose=1)
print(test_loss)
print("test_acc", test_acc)

# 测试
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.pause(0)
plt.show()
test_images = test_images.reshape((10000, 28 * 28))
res = net.predict(test_images)

for i in range(res[1].shape[0]):
    if res[1][i] == 1:
        print("the number for the picture is : ", i)
        break
print(res[1])
