import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 导入相关包
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, LeakyReLU, Dropout, Input
from tensorflow.keras.layers import Reshape, Conv2DTranspose, Flatten
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import keras
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import warnings
# warning.filterwarnings('ignore')

"""设置相关参数"""
# 潜变量维度
latent_dim = 100
# 输入像素维度
dim=64
height = 64
width = 64
channels = 3

concept='heart'

# 搭建生成器网络（Model方式）
generator_input = Input(shape=(latent_dim,))
print("shape1",generator_input.shape)
x = Dense(128*32*32)(generator_input)
print("shape2",x.shape)
x = LeakyReLU()(x)
print("shape3",x.shape)
x = Reshape((32, 32, 128))(x)
print("shape4",x.shape)

# IN: 16*16*128  OUT: 16*16*256
x = Conv2D(256, 5, padding='same')(x)
print("shape5",x.shape)
x = LeakyReLU()(x)
print("shape6",x.shape)

# IN: 16*16*256  OUT: 32*32*256
x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = LeakyReLU()(x)
print("shape7",x.shape)

# 通过反卷积操作，把维度变得和图片一样大了，就别用反卷积了
x = Conv2D(256, 5, padding='same')(x)
x = LeakyReLU()(x)
x = Conv2D(256, 5, padding='same')(x)
x = LeakyReLU()(x)
print("shape8",x.shape)

# 把通道变回来
x = Conv2D(channels, 7, activation='tanh', padding='same')(x)
print("shape9",x.shape)
generator = Model(generator_input, x)
#generator.summary()

# 搭建判别器，判别器就真的是一个卷积神经网络了
discriminator_input = Input(shape=(height, width, channels))
# IN:32*32*3   OUT: 30*30*128
x = Conv2D(128, 3)(discriminator_input)
x = LeakyReLU()(x)

# IN: 30*30*128  OUT:14*14*128
x = Conv2D(128, 4, strides=2)(x)
x = LeakyReLU()(x)

# IN:14*14*128   OUT:6*6*128
x = Conv2D(128, 4, strides=2)(x)
x = LeakyReLU()(x)

# IN:6*6*128  OUT:2*2*128
x = Conv2D(128, 4, strides=2)(x)
x = LeakyReLU()(x)

# 展开成512个神经元
x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(1, activation='sigmoid')(x)

discriminator = Model(discriminator_input, x)
#discriminator.summary()

discriminator_optimizer = optimizers.RMSprop(lr=0.0008,
                                             clipvalue=1.0,
                                             decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer,
                      loss='binary_crossentropy')

# 将上面两个连起来组成生成对抗网络
# 将判别器参数设置为不可训练
discriminator.trainable = False

gan_input = Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
# 搭建对抗网络
gan = Model(gan_input, gan_output)
gan_optimizer = optimizers.RMSprop(lr=0.0004,
                                   clipvalue=1.0,
                                   decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

## 下面导入数据集进行训练，数据集选用的是Keras自带的CIFAR-10数据集
import os
from keras.preprocessing import image

# 加载数据集（手写数字的数据集也可以这样加载）
# (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
# 指定青蛙图像（编号为6）
# x_train = x_train[y_train.flatten() == 6]
# Load hdf5 dataset

import h5py
train_dataset = h5py.File('data_'+concept+'_'+str(dim)+'.h5', 'r')
train_set_x_orig = np.array(train_dataset['X_train'][:]) # your train set features
train_set_y_orig = np.array(train_dataset['y_train'][:]) # your train set labels
test_set_x_orig = np.array(train_dataset['X_test'][:]) # your train set features
test_set_y_orig = np.array(train_dataset['y_test'][:]) # your train set labels
train_dataset.close()

print(train_set_x_orig.shape)
print(train_set_y_orig.shape)

print(train_set_x_orig.max())
print(train_set_x_orig.min())

print(test_set_x_orig.shape)
print(test_set_y_orig.shape)
x_train=train_set_x_orig


print("data size:",x_train.shape)
x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

print("data size(adjusted):",x_train.shape)

# 下面开始训练
iterations = 5100
batch_size = 32
save_dir = 'outputs/image'+str(dim)+'_'+concept

start = 0
f_log=open("generated/"+concept+"_loss.txt",'w',encoding='utf-8')
f_log.write("step\td_loss\ta_loss\n")
for step in range(iterations):

    # 先通过噪声生成64张伪造图片
    noise = np.random.normal(size=(batch_size, latent_dim))
    images_fake = generator.predict(noise)

    # 从真实图片中抽64张
    end = start + batch_size
    images_train = x_train[start:end]

    # 两者混合作为训练集，并打上标签
    x = np.concatenate([images_fake, images_train])
    y = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size, 1))])

    # 向标签中添加噪声
    y += 0.05 * np.random.random(y.shape)

    # 训练判别器
    d_loss = discriminator.train_on_batch(x, y)

    # 训练生成对抗网络
    noise = np.random.normal(size=(batch_size, latent_dim))
    labels = np.ones((batch_size, 1))

    # 通过gan模型来训练生成器模型，冻结判别器模型权重
    a_loss = gan.train_on_batch(noise, labels)

    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0

    # 每50步绘图并保存

    if step % 50 ==0:
        print('discriminator loss:', d_loss)
        print('adversarial loss:', a_loss)
        f_log.write(str(step)+"\t"+str(d_loss)+"\t"+str(a_loss)+"\n")
        img = image.array_to_img(images_fake[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_sample' + str(step) + '.png'))
        img = image.array_to_img(images_train[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_sample' + str(step) + '.png'))
        generator.save('models/model' + str(dim) + '_' + concept + '/GAN_model_' + str(dim) + '_' + concept + '_' + str(
            step) + '.h5')
    if step % 200 ==0:
        f_log.flush()
f_log.close()



