# -*- coding: utf-8 -*-
# 加载所需模块

import matplotlib
matplotlib.use("Agg")
from config import tiny_imagenet_config as config
from pyimagesearch.preprocessing import imagetoarraypreprocessor as ITA
from pyimagesearch.preprocessing import simplespreprocessor as SP
from pyimagesearch.preprocessing import meanpreprocessor as MP
from pyimagesearch.callbacks import epochcheckpoint as ECP
from pyimagesearch.callbacks import trainingmonitor as TM
from pyimagesearch.io import hdf5datasetgenerator as HDFG
from pyimagesearch.nn.conv import deepergooglenet as DGN
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.models import load_model
import keras.backend as K
import argparse
import json

# 解析命令行参数
ap = argparse.ArgumentParser()
ap.add_argument('-c','--checkpoints',required=True,
                help='path to output checkpoint directory')
ap.add_argument('-m','--model',type=str,
                help='path to *specific* model checkpoint to load')
ap.add_argument('-s','--start_epoch',type = int,default=0,
                help='epoch to restart training at')
args=vars(ap.parse_args())

# 数据增强
aug = ImageDataGenerator(rotation_range=18,zoom_range=0.15,
                         width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,
                         horizontal_flip=True,fill_mode='nearest')

# 加载 RGB均值文件
means = json.loads(open(config.DATASET_MEAN).read())

# 初始化预处理
sp = SP.SimplePreprocessor(64,64)
mp = MP.MeanPreprocessor(means['R'],means['G'],means['B'])
iap = ITA.ImageToArrayPreprocessor()

# 训练数据集和验证书籍生成器
trainGen = HDFG.HDF5DatasetGenerator(config.TRAIN_HDF5,64,aug = aug,
                                preprocessors=[sp,mp,iap],classes=config.NUM_CLASSES)
valGen = HDFG.HDF5DatasetGenerator(config.VAL_HDF5,64,
                              preprocessors=[sp,mp,iap],classes=config.NUM_CLASSES)

# 如果不存在checkpoints 模型，则直接初始化模型
if args['model'] is None:
    print("[INFO] compiling model....")
    model = DGN.DeeperGoogLeNet.build(width=64,height=64,depth=3,classes=config.NUM_CLASSES,reg=0.0002)
    # 优化器
    #opt=Adam(1e-3)
    opt = SGD(lr=1e-4,momentum=0.9)
    # 编译模型
    model.compile(loss='categorical_crossentropy',optimizer=opt,
                  metrics=['accuracy'])
# 否则，直接从磁盘中加载checkpoint模型，接着训练
else:
    print("[INFO] loading {}...".format(args['model']))
    model = load_model(args['model'])
    
    # 更新学习率
    print("[INFO] old learning rate:{}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr,1e-5)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))
    
# 回调函数
callbacks = [
        ECP.EpochCheckpoint(args['checkpoints'],every=5,startAt = args['start_epoch']),
        TM.TrainingMonitor(config.FIG_PATH,jsonPath=config.JSON_PATH,startAt = args['start_epoch'])
        ]

#训练网络
model.fit_generator(
        trainGen.generator(),
        steps_per_epoch = trainGen.numImages // 64,
        validation_data = valGen.generator(),
        validation_steps = valGen.numImages // 64,
        epochs = 30,
        max_queue_size = 64 * 2,
        callbacks = callbacks,
        verbose = 1)

# 关闭数据库
trainGen.close()
valGen.close()