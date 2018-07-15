# -*- coding: utf-8 -*-
from config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing import imagetoarraypreprocessor as IAP
from pyimagesearch.preprocessing import simplespreprocessor as SP
from pyimagesearch.preprocessing import patchpreprocessor as PP
from pyimagesearch.preprocessing import meanpreprocessor as MP
from pyimagesearch.preprocessing import croppreprocessor as CP
from pyimagesearch.io import hdf5datasetgenerator as HDF
from pyimagesearch.utils.ranked import rank5_accuracy
from keras.models import load_model
import numpy as np
import progressbar
import json

# 加载RGB均值数据
means = json.loads(open(config.DATASET_MEAN).read())

# 初始化预处理
sp = SP.SimplePreprocessor(227,227)
mp = MP.MeanPreprocessor(means['R'],means['G'],means['B'])
cp = CP.CropPreprocessor(227,227)
iap = IAP.ImageToArrayPreprocessor()

# 加载训练好的模型
print("[INFO] loading model ...")
model = load_model(config.MODEL_PATH)

# 初始化测试数据集生成器，并进行预测
print("[INFO] predicting on test data (no crops)...")
testGen = HDF.HDF5DatasetGenerator(config.TEST_HDF5,64,
                               preprocessors = [sp,mp,iap],
                               classes = 2)
predictions = model.predict_generator(testGen.generator(),
                                      steps = testGen.numImages // 64,
                                      max_queue_size = 64 * 2)
#计算rank-1和rank-5准确度
(rank1,_) = rank5_accuracy(predictions,testGen.db['labels'])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()

# 重新初始化生成器
# 'SimplePreprocessor'
testGen = HDF.HDF5DatasetGenerator(config.TEST_HDF5,64,
                               preprocessors = [mp],classes = 2)
predictions = []

# 初始化进度条
widgets = ['Evaluating: ',progressbar.Percentage()," ",
           progressbar.Bar()," ",progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval = testGen.numImages // 64,
                               widgets=widgets).start()

# 遍历测试集
for (i,(images,labels)) in enumerate(testGen.generator(passes=1)):
    #遍历图像数据集
    for image in images:
        #
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(c) for c in crops],
                          dtype = 'float32')
        pred = model.predict(crops)
        predictions.append(pred.mean(axis = 0))
    # 跟新进度条
    pbar.update(i)

# 计算rank-1准确度
pbar.finish()
print("[INFO] predicting on test data (with crops)....")
(rank1,_) = rank5_accuracy(predictions,testGen.db['labels'])
print("[INFO'] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()
