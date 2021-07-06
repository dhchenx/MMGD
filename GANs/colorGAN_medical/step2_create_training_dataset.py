import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
import h5py
import cv2
import scipy
import io

label_ids=["0","1","2","3"]
concepts=["lung","heart","cxr","hand"]

current_concept='hand'
#导入必要的包
def get_files(label_id,file_dir):
    cats = []

    label_cats = []
    #dogs = []
    #label_dogs = []
 
    for file in os.listdir(file_dir + '/'+label_id):
        cats.append(file_dir + '/'+label_id + '/' + file)
        label_cats.append(0)  # 添加标签，该类标签为0，此为2分类例子，多类别识别问题自行添加
    #for file in os.listdir(file_dir + '/1'):
    #    dogs.append(file_dir + '/1' + '/' + file)
    #    label_dogs.append(1)
         
    # 把cat和dog合起来组成一个list（img和lab）
    #image_list = np.hstack((cats, dogs))
    #label_list = np.hstack((label_cats, label_dogs))
    image_list=cats
    label_list=label_cats

    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # 从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
     
    return image_list, label_list
    # 返回两个list 分别为图片文件名及其标签 顺序已被打乱

label_id=label_ids[concepts.index(current_concept)]
train_dir = 'dataset/train'
image_list, label_list = get_files(label_id,train_dir)

print(len(image_list))
print(len(label_list))

# 450为数据长度的20%
percent20=int(len(image_list)*0.2)
dim=64
Train_image = np.random.rand(len(image_list) - percent20, dim, dim, 3).astype('float32')
Train_label = np.random.rand(len(image_list) - percent20, 1).astype('float32')

Test_image = np.random.rand(percent20, dim, dim, 3).astype('float32')
Test_label = np.random.rand(percent20, 1).astype('float32')

for i in range(len(image_list) - percent20):
    img=plt.imread(image_list[i])
    shrink_img = cv2.resize(img, (dim,dim), interpolation=cv2.INTER_AREA)
    Train_image[i] = np.array(shrink_img)
    Train_label[i] = np.array(label_list[i])

for i in range(len(image_list) - percent20, len(image_list)):
    img = plt.imread(image_list[i])
    shrink_img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)
    Test_image[i + percent20 - len(image_list)] = np.array(shrink_img)
    Test_label[i + percent20 - len(image_list)] = np.array(label_list[i])

# Create a new file
h5_path='data_'+current_concept+'_64.h5'
if os.path.exists(h5_path):
    os.remove(h5_path)
f = h5py.File(h5_path, 'w')
f.create_dataset('X_train', data=Train_image)
f.create_dataset('y_train', data=Train_label)
f.create_dataset('X_test', data=Test_image)
f.create_dataset('y_test', data=Test_label)
f.close()
