# -*- coding: utf-8 -*-
# 加载所需要模块
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io import hdf5datasetwriter as HDF
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os


# 命令行参数
ap = argparse.ArgumentParser()
ap.add_argument('-d',"--dataset",required = True,
                help = 'path to input dataset')
ap.add_argument("-0",'--output',required = True,
                help = 'path ot output hdf5 file')
ap.add_argument('-b','--batch_size',type = int,default = 16,
                help='batch size of images to ba passed through network')
ap.add_argument('-s','--buffer_size',type=int,default=1000,
                help = 'size of feature extraction buffer')
args = vars(ap.parse_args())
# batch
bs = args['batch_size']

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args['dataset']))
# 混洗数据
random.shuffle(imagePaths)
# 标签获取
labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
# 编码编码化
le = LabelEncoder()
labels = le.fit_transform(labels)
print("[INFO] loading network...")
# imagenet上训练的权重
model = ResNet50(weights = 'imagenet',include_top=False)
#ResNet50的最后一个average pooling层的维度是2048
dataset = HDF.HDF5DatasetWriter((len(imagePaths),2048),
                            args['output'],dataKey='feature',buffSize=args['buffer_size'])
dataset.storeClassLabels(le.classes_)

# 初始化进度条
widgets = ['Extracting Features: ',progressbar.Percentage(),' ',
           progressbar.Bar(),' ',progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval = len(imagePaths),
                               widgets = widgets).start()

for i in np.arange(0,len(imagePaths),bs):
    #提取图像和标签
    batchPaths = imagePaths[i:i+bs]
    batchLabels = labels[i:i+bs]
    batchImages = []
    # 对每张图像进行处理
    for (j,imagePath) in enumerate(batchPaths):
        image = load_img(imagePath,target_size = (224,224))
        image = img_to_array(image)
        # 增加一维度
        image = np.expand_dims(image,axis = 0)
        # 提取imageNet数据集中RGB均值
        image = imagenet_utils.preprocess_input(image)
        batchImages.append(image)
    batchImages = np.vstack(batchImages)
    # 提取最后一个pool层特征图像
    features = model.predict(batchImages,batch_size=bs)
    # 拉平
    features = features.reshape((features.shape[0],2048))
    # hdf5数据集中增加特征和标签
    dataset.add(features, batchLabels)
    pbar.update(i)
# 关闭数据
dataset.close()
pbar.finish()

