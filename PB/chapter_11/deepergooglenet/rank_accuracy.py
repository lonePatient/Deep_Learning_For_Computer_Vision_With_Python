# -*- coding: utf-8 -*-
# 加载所需模块
from config import tiny_imagenet_config as config
from pyimagesearch.preprocessing import imagetoarraypreprocessor as ITA
from pyimagesearch.preprocessing import simplespreprocessor as SP
from pyimagesearch.preprocessing import meanpreprocessor as MP
from pyimagesearch.io import hdf5datasetgenerator as HDFG
from pyimagesearch.utils.ranked import rank5_accuracy
from keras.models import load_model
import json

# 加载RGB均值文件
means = json.loads(open(config.DATASET_MEAN).read())

# 初始化预处理
sp = SP.SimplePreprocessor(64,64)
mp = MP.MeanPreprocessor(means['R'],means['G'],means['B'])
iap = ITA.ImageToArrayPreprocessor()


# 初始化测试数据集生成器
testGen = HDFG.HDF5DatasetGenerator(config.TEST_HDF5,64,preprocessors = [sp,mp,iap],classes=config.NUM_CLASSES)

# 加载预训练好的模型
print("[INFO] loading model ...")
model = load_model(config.MODEL_PATH)

# 对测试集进行预测
print("[INFO] predicting on test data...")
predictions = model.predict_generator(testGen.generator(),steps = testGen.numImages // 64,max_queue_size = 64 * 2)

# 计算rank-1和rank5准确度
(rank1,rank5) = rank5_accuracy(predictions,testGen.db['labels'])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))

#关闭数据库
testGen.close()
