# -*- coding: utf-8 -*-
# 加载所需模块
import matplotlib
matplotlib.use("Agg")
from config import tiny_imagenet_config as config
from pyimagesearch.preprocessing import imagetoarraypreprocessor as ITAP
from pyimagesearch.preprocessing import simplespreprocessor as SIP
from pyimagesearch.preprocessing import meanpreprocessor as MP
from pyimagesearch.callbacks import epochcheckpoint as EPO
from pyimagesearch.callbacks import trainingmonitor as TM
from pyimagesearch.io import hdf5datasetgenerator as HDFG
from pyimagesearch.nn.conv import resnet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam,SGD
from keras.models import load_model
import keras.backend as K
import argparse
import json
import sys
import os
# 初始化迭代次数和学习率
NUM_EPOCHS = 75
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

# 数据增强
aug = ImageDataGenerator(rotation_range=18,zoom_range=0.15,
                         width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,
                         horizontal_flip=True,fill_mode='nearest')

# 加载 RGB均值文件
means = json.loads(open(config.DATASET_MEAN).read())

# 初始化预处理
sp = SIP.SimplePreprocessor(64,64)
mp = MP.MeanPreprocessor(means['R'],means['G'],means['B'])
iap = ITAP.ImageToArrayPreprocessor()

# 训练数据集和验证书籍生成器
trainGen = HDFG.HDF5DatasetGenerator(config.TRAIN_HDF5,64,aug = aug,
                                preprocessors=[sp,mp,iap],classes=config.NUM_CLASSES)
valGen = HDFG.HDF5DatasetGenerator(config.VAL_HDF5,64,
                              preprocessors=[sp,mp,iap],classes=config.NUM_CLASSES)

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
model = resnet.ResNet.build(32, 32, 3, config.NUM_CLASSES, (3, 4, 6), (64, 128, 256, 512), reg=0.0005,dataset="tiny_imagenet")
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


#训练网络
print("[INFO] training network....")
model.fit_generator(
        trainGen.generator(),
        steps_per_epoch = trainGen.numImages // 64,
        validation_data = valGen.generator(),
        validation_steps = valGen.numImages // 64,
        epochs = NUM_EPOCHS,
        max_queue_size = 64 * 2,
        callbacks = callbacks,
        verbose = 1)

# 保存模型道磁盘
print("[INFO] serializing network...")
model.save(args['model'])

# 关闭数据库
trainGen.close()
valGen.close()
