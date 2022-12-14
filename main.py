import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from PIL import Image
import numpy as np

from keras import backend as K
# the data, split between train and test sets

def read_folder(start_code, end_code, dict, train_x, train_y, correction):
    for code in range(start_code, end_code+1):
        files = os.listdir("training_set\\" + str(code))
        training_hist[code] = len(files)
        for file in files:
            pic_array = read_picture("training_set\\" + str(code) + "\\" + file)
            train_x.append(pic_array)
            train_y.append(code-correction)


def read_picture(picture):
    pic = Image.open(picture)
    pix = np.array(pic)
    return pix

def create_test(test_x, test_y, test_dict):
    files = os.listdir("test_set\\")
    for file in files:
        pic = Image.open( "test_set\\" + file)
        pix = np.array(pic)
        test_x.append(pix)
        test_y.append(ord(file[0]))

        if  48 <= ord(file[0]) <= 57:
            code = ord(file[0])-48
        elif 97 <= ord(file[0]) <= 122:
            code = ord(file[0]) - 87
        elif 65 <= ord(file[0]) <= 90:
            code = ord(file[0]) - 55

        if code in test_dict.keys():
            test_dict[code] += 1
        else:
            test_dict[code] = 1


x_train = []
y_train = []

training_hist = {}
#read_folder(48, 57, training_hist,x_train, y_train, 48)
#read_folder(97, 122, training_hist,x_train, y_train, 87)
#read_folder(65, 90, training_hist,x_train, y_train, 55)
x_train = np.array(x_train)
y_train = np.array(y_train)


print(training_hist)
print(type(x_train), x_train.shape)
print(type(y_train), y_train.shape)
print(y_train)


x_test = []
y_test = []
test_hist = {}
create_test(x_test, y_test, test_hist)
x_test = np.array(x_test)
y_test = np.array(y_test)


print(test_hist)
print(type(x_test), x_test.shape)
print(type(y_test), y_test.shape)



# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
#
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
# input_shape = (28, 28, 1)
#
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
#
# print(y_test)
# print(type(y_test))
#
# #histogram of the test set
# count = [0]*10
# for item in y_test:
#     count [item] += 1
#
# digits = []
# for i in range(10):
#     digits.append(str(i))
#
# fig, ax = plt.subplots(1,1, figsize = (10,10))
# ax.barh(digits, count)
#
# plt.xlabel("Number of elements in the test")
# plt.ylabel('Digits')
# plt.grid()
# plt.show()
#
# #histogram of the training set
# count = [0]*10
# for item in y_train:
#     count [item] += 1
#
# digits = []
# for i in range(10):
#     digits.append(str(i))
#
# fig, ax = plt.subplots(1,1, figsize = (10,10))
# ax.barh(digits, count)
#
# plt.xlabel("Number of elements in the training set")
# plt.ylabel('Digits')
# plt.grid()
# plt.show()
#
#
# y_train = keras.utils.to_categorical(y_train, num_classes=10, dtype='int')
# y_test = keras.utils.to_categorical(y_test, num_classes=10, dtype='int')
#
# print(y_test)
# print(type(y_test))
#
# print(y_train)
#
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
#
#
# batch_size = 128
# num_classes = 10
# epochs = 10
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
#
# hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
# print("The model has successfully trained")
# model.save('mnist.h5')
# print("Saving the model as mnist.h5")