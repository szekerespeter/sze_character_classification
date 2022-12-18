import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, LSTM
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import os
from PIL import Image
import numpy as np
from sklearn.utils import shuffle
import pandas as pd


def read_folder(start_code, end_code, dict, train_x, train_y, correction):
    for code in range(start_code, end_code+1):
        files = os.listdir("training_set_full\\" + str(code))
        training_hist[code] = len(files)
        for file in files:
            pic_array = read_picture("training_set_full\\" + str(code) + "\\" + file)
            train_x.append(pic_array)
            train_y.append(code-correction)


def read_picture(picture):
    pic = Image.open(picture).convert('L')
    pix = np.array(pic)
    return pix

def create_test(test_x, test_y, test_dict):
    files = os.listdir("test_set3\\")
    for file in files:
        pic = Image.open( "test_set3\\" + file).convert('L')
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


num_classes = 62


x_train = []
y_train = []

training_hist = {}

read_folder(48, 57, training_hist,x_train, y_train, 48)
read_folder(97, 122, training_hist,x_train, y_train, 87)
#read_folder(65, 90, training_hist,x_train, y_train, 55)
read_folder(65, 90, training_hist,x_train, y_train, 39)

x_train = np.array(x_train)
y_train = np.array(y_train)





print(training_hist)
print(type(x_train), x_train.shape)
print(type(y_train), y_train.shape)
print(y_train)


x_test = []
y_test = []
test_hist = {}

x_test = np.array(x_test)
y_test = np.array(y_test)

x_train_shuffled,y_train_shuffled = shuffle(x_train,y_train)

x_train = x_train_shuffled[:50000]
x_test = x_train_shuffled[50001:]

y_train = y_train_shuffled[:50000]
y_test = y_train_shuffled[50001:]



print(test_hist)
print(type(x_test), x_test.shape)
print(type(y_test), y_test.shape)


x_train = x_train.reshape(x_train.shape[0], 64, 64, 1)
x_test = x_test.reshape(x_test.shape[0], 64, 64, 1)


input_shape = (64, 64, 1)



y_train = keras.utils.to_categorical(y_train, num_classes=num_classes, dtype='int')
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes, dtype='int')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print("x_train:" ,x_train[0])
print("y_train:", y_train[0])

batch_size = 32

epochs = 10

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(64, 64, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dense(32))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

opt = SGD(learning_rate=0.02, momentum=0.5)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))


print("The model has successfully trained")
model.save('mnist.h5')
print("Saving the model as mnist.h5")