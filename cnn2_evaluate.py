from keras.models import load_model
import keras
from keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle

model = load_model('cnn2.h5')

def read_folder(start_code, end_code, dict, train_x, train_y, correction):
    for code in range(start_code, end_code+1):
        files = os.listdir("training_set_full\\" + str(code))
        training_hist[code] = len(files)
        for file in files:
            pic_array = read_picture("training_set_full\\" + str(code) + "\\" + file)
            train_x.append(pic_array)
            train_y.append(code-correction)

def read_picture(picture):
    pic = Image.open(picture)   #.convert('L')
    pix = np.array(pic)
    return pix

def create_test(test_x, test_y, test_dict):
    files = os.listdir("test_set3\\")
    #files = os.listdir("paint\\")
    for file in files:
        pic = Image.open( "test_set3\\" + file)
        #pic = Image.open("paint\\" + file).convert('L')
        pix = np.array(pic)
        test_x.append(pix)

        if  48 <= ord(file[0]) <= 57:
            code = ord(file[0])-48
        elif 97 <= ord(file[0]) <= 122:
            code = ord(file[0]) - 87
        elif 65 <= ord(file[0]) <= 90:
            code = ord(file[0]) - 55

        test_y.append(code)

        if code in test_dict.keys():
            test_dict[code] += 1
        else:
            test_dict[code] = 1




x_train = []
y_train = []

training_hist = {}
read_folder(48, 57, training_hist,x_train, y_train, 48)
read_folder(97, 122, training_hist,x_train, y_train, 87)
x_train = np.array(x_train)
y_train = np.array(y_train)


x_test = []
y_test = []
test_hist = {}
create_test(x_test, y_test, test_hist)
x_test = np.array(x_test)
y_test = np.array(y_test)


x_train,y_train = shuffle(x_train,y_train)
x_test, y_test = shuffle(x_test, y_test)



x_train = x_train.reshape(x_train.shape[0], 64, 64, 1)
x_test = x_test.reshape(x_test.shape[0], 64, 64,1)
input_shape = (64, 64, 1)

y_train = keras.utils.to_categorical(y_train, num_classes=36, dtype='int')
y_test = keras.utils.to_categorical(y_test, num_classes=36, dtype='int')


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

from numpy import argmax
y_pred = model.predict(x_test)
y_test_conv = [argmax(y, axis=None, out=None) for y in y_test]

y_pred_conv = [argmax(y, axis=None, out=None) for y in y_pred]


confusion_mtx = tf.math.confusion_matrix(y_test_conv, y_pred_conv)
#plt.figure(figsize=(10, 8))
plt.figure(figsize=(18, 14))
sns.heatmap(confusion_mtx,
            xticklabels=[0,1,2,3,4,5,6,7,8,9, 'a','b','c','d','e','f','g','h','i','j','k','l','m',
                         'n','o','p','q','r','s','t','u','v','w','x','y','z',
                         ],
            yticklabels=[0,1,2,3,4,5,6,7,8,9, 'a','b','c','d','e','f','g','h','i','j','k','l','m',
                         'n','o','p','q','r','s','t','u','v','w','x','y','z',
                         ],
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

print(y_pred, type(y_pred))
print(y_test, type(y_test))
