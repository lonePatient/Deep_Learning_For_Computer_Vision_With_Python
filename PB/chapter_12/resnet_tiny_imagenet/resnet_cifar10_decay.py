#encoding:utf-8
# 加载所需模块
import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import resnet
from pyimagesearch.callbacks import epochcheckpoint as EPO
from pyimagesearch.callbacks import trainingmonitor as TM
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.models import load_model
import keras.backend as K
import numpy as np
import argparse
import sys
import os

# 初始化迭代次数和学习率
NUM_EPOCHS = 100
INIT_LR = 1e-1

def ploy_decay(epoch):
    # 初始化相关参数
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    # 根据多项式衰减更新学习率
    alpha = baseLR * (1- (epoch / float(maxEpochs))) ** power
    return alpha

# 解析命令行参数
ap = argparse.ArgumentParser()
ap.add_argument('-m','--model',required=True,help='path to  output model ')
ap.add_argument('-o','--output',required=True,help='path to output directory (logs,plots,etc.)')
args = vars(ap.parse_args())

# 加载train和test数据集
print('[INFO] loading CIFAR-10 data...')
((trainX,trainY),(testX,testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")
# 计算RGB通道均值
mean = np.mean(trainX,axis =0)
# 零均值化
trainX -= mean
testX -= mean
# 标签编码
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)
# 数据增强
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')
# 回调函数列表
figPath = os.path.sep.join([args['output'],"{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args['output'],"{}.json".format(os.getpid())])
callbacks = [
    TM.TrainingMonitor(figPath,jsonPath),
    LearningRateScheduler(ploy_decay)
]
# 初始化模型和优化器
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR,momentum=0.9)
model = resnet.ResNet.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# 训练网络
print("[INFO] training network.....")
model.fit_generator(
    aug.flow(trainX,trainY,batch_size=128),
    validation_data = (testX,testY),
    steps_per_epoch = len(trainX) // 128,
    epochs = 10,
    callbacks = callbacks,
    verbose =1
)
# 将模型序列化道磁盘
print("[INFO] serializing network...")
model.save(args['model'])