import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint

## load datasets
filepath = 'datasets/Medical_MNIST/'
_,labels,_ = next(os.walk(filepath))
print('Labels :', labels)

# Converts numbers to 6 figure strings
# e.g. '24' to '000024', '357' to '000357'
def to_6sf(n):
    new = str(n)
    new = '0'*(6-len(new))+new
    return new

x_all = []
y_all = []
for x in labels:
    num_images = len(os.listdir(filepath+x+'/'))
    for i in range(num_images):
        X = Image.open(filepath+x+'/'+to_6sf(i)+'.jpeg')
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1],1))
        x_all.append(X)
        y_all.append(x)
x_all = np.array(x_all)
y_all = np.array(y_all)

print('Total Samples :',len(x_all))

## preprocess datasets

# Prepare Encoder-Decoder Dict
encode_y = dict()
decode_y = dict()
for x in enumerate(labels):
    encode_y[x[1]] = x[0]
    decode_y[x[0]] = x[1]
print('Encoder :',encode_y)
print('Decoder :',decode_y)

# Apply Encoder dict
for x in range(len(x_all)):
    y_all[x] = encode_y[y_all[x]]
y_all = np.array(y_all, dtype=np.int16)

## Split to Train & Test
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, shuffle=True)
print('Total Train Samples :',len(x_train))
print('Total Test Samples :',len(x_test))

# Load pretrained model
model = load_model('CNN-Medical_MNIST-saved.h5')
loss, acc = model.evaluate(x_test, y_test)
print('Test Loss :',loss)
print('Test Accuracy :',acc)

# Predict
IMAGE_INDEX = 100
print("x_test.shape",x_test.shape)
img = x_test[IMAGE_INDEX]

plt.imshow(img.squeeze())
plt.show()

print('Actual :', decode_y[y_test[IMAGE_INDEX]])

img = np.reshape(img,(1,64,64,1))
predicted = model.predict(img)
predicted = np.argmax(predicted)
predicted = decode_y[predicted]
print('Predicted :', predicted)

