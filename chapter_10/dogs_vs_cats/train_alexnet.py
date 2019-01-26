# -*- coding: utf-8 -*-
#加载所需模块
import matplotlib
matplotlib.use('Agg')
from config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing import imagetoarraypreprocessor as IAP
from pyimagesearch.preprocessing import simplespreprocessor as SP
from pyimagesearch.preprocessing import patchpreprocessor as PP
from pyimagesearch.preprocessing import meanpreprocessor as MP
from pyimagesearch.callbacks import trainingmonitor as TM
from pyimagesearch.io import hdf5datasetgenerator as HDF
from pyimagesearch.nn.conv import alexnet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os

# 数据增强
aug = ImageDataGenerator(rotation_range = 20,zoom_range = 0.15,
                         width_shift_range = 0.2,height_shift_range = 0.2,
                         shear_range=0.15,horizontal_flip=True,
                         fill_mode='nearest')

#加载ＲＧＢ均值文件
means = json.loads(open(config.DATASET_MEAN).read())

# 预处理
sp = SP.SimplePreprocessor(227,227)
pp = PP.PatchPreprocessor(227,227)
mp = MP.MeanPreprocessor(means['R'],means['G'],means['B'])
iap = IAP.ImageToArrayPreprocessor()

#初始化训练数据集和验证数据集生成器
trainGen = HDF.HDF5DatasetGenerator(dbPath=config.TRAIN_HDF5,batchSize=128,aug=aug,preprocessors= [pp,mp,iap],classes = 2)
valGen = HDF.HDF5DatasetGenerator(config.VAL_HDF5,128,preprocessors=[sp,mp,iap],classes =2)
# 初始化优化器
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = alexnet.AlexNet.build(width=227,height=227,depth=3,
                      classes=2,reg=0.0002)
model.compile(loss = 'binary_crossentropy',optimizer=opt,
              metrics = ['accuracy'])
# callbacks
path = os.path.sep.join([config.OUTPUT_PATH,"{}.png".format(os.getpid())])
callbacks = [TM.TrainingMonitor(path)]

# 训练网络
model.fit_generator(
        trainGen.generator(),
        steps_per_epoch = trainGen.numImages // 128,
        validation_data = valGen.generator(),
        validation_steps = valGen.numImages // 128,
        epochs = 75,
        max_queue_size = 128 * 2,
        callbacks = callbacks,
        verbose = 1)

# 保存模型文件
print("[INFO] serializing model ....")
model.save(config.MODEL_PATH,overwrite = True)

# 关闭HDF5数据
trainGen.close()
valGen.close()
