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
#导入必要的包
def get_files(file_dir):

    labels=['AbdomenCT','BreastMRI','ChestCT','CXR','Hand','HeadCT']
    files_set=[]
    lbs_set=[]
    for idx,label in enumerate(labels):
        files=[]
        lbs=[]
        for file in os.listdir(file_dir + '/'+label):
            files.append(file_dir + '/'+label + '/' + file)
            lbs.append(idx)  # 添加标签，该类标签为0，此为2分类例子，多类别识别问题自行添加
        files_set.append(files)
        lbs_set.append(lbs)

    files_tuple=()
    lbs_tuple=()
    for idx, label in enumerate(labels):
        files_tuple=files_tuple+tuple(files_set[idx])
        lbs_tuple=lbs_tuple+tuple(lbs_set[idx])

    # 把cat和dog合起来组成一个list（img和lab）
    image_list = np.hstack(files_tuple)
    label_list = np.hstack(lbs_tuple)

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

train_dir = 'datasets/Medical_MNIST'
image_list, label_list = get_files(train_dir)

print(len(image_list))
print(len(label_list))

# 450为数据长度的20%
percent20=int(len(image_list)*0.2)
dim=28
Train_image = np.random.rand(len(image_list) - percent20, dim, dim).astype('float32')
Train_label = np.random.rand(len(image_list) - percent20, 1).astype('float32')

Test_image = np.random.rand(percent20, dim, dim).astype('float32')
Test_label = np.random.rand(percent20, 1).astype('float32')

for i in range(len(image_list) - percent20):
    img=plt.imread(image_list[i])
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shrink_img = cv2.resize(img, (dim,dim), interpolation=cv2.INTER_AREA)
    Train_image[i] = np.array(shrink_img)
    Train_label[i] = np.array(label_list[i])

for i in range(len(image_list) - percent20, len(image_list)):
    img = plt.imread(image_list[i])
    # img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shrink_img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)
    Test_image[i + percent20 - len(image_list)] = np.array(shrink_img)
    Test_label[i + percent20 - len(image_list)] = np.array(label_list[i])

# Create a new file
f = h5py.File('med_minist_data.h5', 'w')
f.create_dataset('X_train', data=Train_image)
f.create_dataset('y_train', data=Train_label)
f.create_dataset('X_test', data=Test_image)
f.create_dataset('y_test', data=Test_label)
f.close()
