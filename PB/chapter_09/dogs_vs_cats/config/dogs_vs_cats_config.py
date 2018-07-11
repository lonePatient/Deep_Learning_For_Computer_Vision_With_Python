# -*- coding: utf-8 -*-

# 原始图像路径
IMAGES_PATH = "../datasets/kaggle_dogs_vs_cats/train"

#类别总数
NUM_CLASSES = 2
# 验证数据集大小
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
# 测试数据集代销
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

# hdf5数据保存路径
TRAIN_HDF5 = "../datasets/kaggle_dogs_vs_cats/hdf5/train.hdf5"
VAL_HDF5 = "../datasets/kaggle_dogs_vs_cats/hdf5/val.hdf5"
TEST_HDF5 = "../datasets/kaggle_dogs_vs_cats/hdf5/test.hdf5"

# 模型保存路径
MODEL_PATH = "output/alexnet_dogs_vs_cats.model"

# 数据均值保存路径
DATASET_MEAN = "output/dogs_vs_cats_mean.json"

# 其余输出保存路径
OUTPUT_PATH = "output"