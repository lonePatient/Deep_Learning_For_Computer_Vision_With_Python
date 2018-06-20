#encoding:utf-8
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image  import img_to_array
from keras.preprocessing.image import  load_img
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io import hdf5datasetwriter as hdf5DW
from imutils import paths
import numpy as np
import progressbar # 安装：pip install progressbar2
import argparse
import random
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d",'--dataset',required=True,help='path to input dataset')
ap.add_argument('-o','--output',required=True,help='path to output HDF5 file')
ap.add_argument("-b",'--batch_size',type =int,default=32,help='batch size of image to be passed through network')
ap.add_argument('-s','--buffer_size',type=int,default=1000,help='size of feature extraction buffer')
args = vars(ap.parse_args())

bs = args['batch_size']
print("[INFO] loading image....")
imagePaths = list(paths.list_images(args['dataset']))
# 混洗图像路径集
random.shuffle(imagePaths)
# 从图像路径中提取标签
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# 加载 VGG16网络
print("[INFO] loading network.....")
model = VGG16(weights='imagenet',include_top=False)
# 初始化 HDF5数据写入模块
dataset = hdf5DW.HDF5DatasetWriter((len(imagePaths),512*7*7),
                                   args['output'],dataKey='features',bufSize=args['buffer_size'])
dataset.storeClassLabels(le.classes_)
# 初始化进度条
widgets = ['EXtracting Features: ',progressbar.Percentage()," ",progressbar.Bar(),' ',progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),widgets=widgets).start()
# 每batch_size遍历全量图像数据
for i in np.arange(0,len(imagePaths),bs):
    # 提取batch_size的图像以及标签数据
    batchPaths = imagePaths[i:i+bs]
    batchLabels = labels[i:i+bs]
    batchImages = []

    for (j,imagePath) in enumerate(batchPaths):
        # 加载图像，并调整大小为224x224
        image = load_img(imagePath,target_size=(224,224))
        image = img_to_array(image)
        # 图像预处理
        # 1. 增加一个维度
        # 2. 利用imagenet信息进行标准化处理
        image = np.expand_dims(image,axis =0)
        image = imagenet_utils.preprocess_input(image)
        # 将处理玩的图像加入batch
        batchImages.append(image)
    # 使用模型的预测值作为特征向量
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages,batch_size=bs)
    # 将最后一个池化层拉平，调整特征的大小
    features = features.reshape((features.shape[0],512 * 7* 7))
    # 将得到的特征和标签加入HDF5数据集中
    dataset.add(features,batchLabels)
    pbar.update(i)
dataset.close()
pbar.finish()