# -*- coding: utf-8 -*-
from os import path

#训练数据集和验证数据集路径
TRAIN_IMAGES = "../datasets/tiny-imagenet-200/train"
VAL_IMAGES = "../datasets/tiny-imagenet-200/val/images"

# 验证数据集与标签映射文件
VAL_MAPPINGS = "../datasets/tiny-imagenet-200/val/val_annotations.txt"

# WordNet hierarchy文件路径
WORDNET_IDS = '../datasets/tiny-imagenet-200/wnids.txt'
WORD_LABELS = '../datasets/tiny-imagenet-200/words.txt'

# 从train数据中构造test数据
NUM_CLASSES = 100
NUM_TEST_IMAGES = 30 * NUM_CLASSES

# 定义输出路径
TRAIN_HDF5 = "../datasets/tiny-imagenet-200/hdf5/train.hdf5"
VAL_HDF5 = "../datasets/tiny-imagenet-200/hdf5/val.hdf5"
TEST_HDF5 = "../datasets/tiny-imagenet-200/hdf5/test.hdf5"

# 数据均值文件
DATASET_MEAN = "output/tiny-image-net-200-mean.json"

# 输出路径和性能结果
OUTPUT_PATH = "output"
MODEL_PATH = path.sep.join([OUTPUT_PATH,
                            "checkpoints/epoch_70.hdf5"])
FIG_PATH = path.sep.join([OUTPUT_PATH,
                          'deepergooglenet_tinyimagenet.png'])
JSON_PATH = path.sep.join([OUTPUT_PATH,
                           'deepergooglenet_tinyimagenet.json'])


