from keras.models import load_model
import keras
from keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


model = load_model('mnist.h5')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], x_test.shape[2],1)


y_train = keras.utils.to_categorical(y_train, num_classes=10, dtype = 'int')
y_test = keras.utils.to_categorical(y_test, num_classes=10, dtype='int')

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

digits = []
for i in range(10):
    digits.append(str(i))

confusion_mtx = tf.math.confusion_matrix(y_test_conv, y_pred_conv)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=digits,
            yticklabels=digits,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

print(y_pred, type(y_pred))
print(y_test, type(y_test))

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test_conv, y_pred_conv)
# from sklearn.metrics import ConfusionMatrixDisplay
# disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=digits)
# disp.plot(cmap = plt.cm.Blues)
# plt.show()