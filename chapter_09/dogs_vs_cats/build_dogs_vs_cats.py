# -*- coding: utf-8 -*-

# 加载模块
from config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import AspectAwarePreprocessor as AAP
from pyimagesearch.io import hdf5datasetwriter as HDF
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

# 图像路径
trainPaths = list(paths.list_images(config.IMAGES_PATH))
# 获取标签
trainLabels = [p.split(os.path.sep)[2].split(".")[0] for p in trainPaths]
# 标签编码化
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# 将原始的train分割成train和test两份
split = train_test_split(trainPaths, trainLabels,test_size=config.NUM_TEST_IMAGES, stratify=trainLabels,random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split
#　将新的train分割成train和val两份
split = train_test_split(trainPaths, trainLabels,test_size=config.NUM_VAL_IMAGES, stratify=trainLabels,random_state=42)
(trainPaths, valPaths, trainLabels, valLabels) = split


# 将数据构建一个list，方便写入HDF5文件中
datasets = [
 ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
 ("val", valPaths, valLabels, config.VAL_HDF5),
 ("test", testPaths, testLabels, config.TEST_HDF5)]

# 数据预处理
aap = AAP.AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])

# 遍历数据集
for (dType, paths, labels, outputPath) in datasets:
    # HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    writer = HDF.HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)
    
    # 初始化进度条
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
    progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),
    widgets=widgets).start()
    
    # 遍历路径
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # 读取数据并预处理
        image = cv2.imread(path)
        image = aap.preprocess(image)
        # 如果是train，则计算RGB均值
        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        # 写入数据
        writer.add([image], [label])
        pbar.update(i)
    
    pbar.finish()
    writer.close()
    
# 保存成json文件
print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()