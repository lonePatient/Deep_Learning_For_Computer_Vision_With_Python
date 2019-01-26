# -*- coding: utf-8 -*-
#加载所需模块
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import minigooglenet as MGN
from pyimagesearch.callbacks import trainingmonitor as TM
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import os

# 总的迭代次数
NUM_EPOCHS = 70
#初始学习率
INIT_LR = 5e-3

def poly_decay(epoch):
    # 初始最大迭代次数和学习率
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    # 以多项式的方式衰减学习率
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return alpha
    
# 定义命令行参数
ap = argparse.ArgumentParser()
ap.add_argument('-m','--model',required=True,
                help = 'path to output model')
ap.add_argument('-o','--output',required=True,
                help='path to output directory (logs,plots,etc.)')
args = vars(ap.parse_args())

# 加载训练数据集和测试数据集
print("[INFO] loading CIFAR-10 data...")
((trainX,trainY),(testX,testY))  = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

#计算均值
mean = np.mean(trainX,axis = 0)
trainX -= mean
testX -= mean

# 标签编码化
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# 数据增强
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range = 0.1,
                         horizontal_flip = True,
                         fill_mode = 'nearest')

# 回调,监控
figPath = os.path.sep.join([args['output'],'{}.png'.format(os.getpid())])
jsonPath = os.path.sep.join([args['output'],"{}.json".format(os.getpid())])
callbacks = [TM.TrainingMonitor(figPath,jsonPath = jsonPath),
             LearningRateScheduler(poly_decay)]

# 初始化优化器和模型
print("[INFO] compiling model...")
opt= SGD(lr = INIT_LR,momentum=0.9)
model = MGN.MiniGoogLeNet.build(width=32,height = 32,depth=3,classes = 10)
model.compile(loss = 'categorical_crossentropy',optimizer=opt,
              metrics = ['accuracy'])
# 训练网络
print("[INFO] training network...")
model.fit_generator(aug.flow(trainX,trainY,batch_size = 64),
                    validation_data = (testX,testY),steps_per_epoch = len(trainX) // 64,
                    epochs = NUM_EPOCHS,callbacks = callbacks,verbose = 1)

# 保存模型到磁盘中
print("[INFO] serializing network...")
model.save(args['model'])
