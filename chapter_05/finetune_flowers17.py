#encoding:utf-8
# 加载所需要模块
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor as ITAP
from pyimagesearch.preprocessing import AspectAwarePreprocessor as AAP
from pyimagesearch.datasets import SimpleDatasetLoader as SDL
from pyimagesearch.nn.conv import fcheadnet as FCN
from keras.preprocessing.image   import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.layers import Input
from keras.models import Model
from keras .applications import VGG16
from keras.optimizers import SGD
from keras.models import Model
from imutils import paths
import numpy as np
import argparse
import os

# 解析命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help='path to input dataset')
ap.add_argument('-m','--model',required=True,help='path to output model')
args = vars(ap.parse_args())

# 数据增强
aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,
                         shear_range=0.2,zoom_range=0.2,
                         horizontal_flip=True,fill_mode='nearest')

# 从磁盘中加载图片，并提取标签
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args['dataset']))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# 初始化图像预处理
aap = AAP.AspectAwarePreprocesser(224,224)
iap= ITAP.ImageToArrayPreprocess()
# 加载图像数据，并进行图像数据预处理
sdl = SDL.SimpleDatasetLoader(preprocessors=[aap,iap])
(data,labels)  = sdl.load(imagePaths,verbose=500)
data = data.astype("float") / 255.0

# 数据划分
(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.25,random_state=42)
# 标签进行编码化处理
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# 加载VGG16网络，不返回原始模型的全连接层
baseModel = VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape = (224,224,3)))
# 初始化新的全连接层
headModel = FCN.FCHeadNet.build(baseModel,len(classNames),256)
#拼接模型
model = Model(inputs=baseModel.input,outputs = headModel)

# 遍历所有层，并冻结对应层的权重
for layer in baseModel.layers:
    layer.trainable = False

# 编译模型
print("[INFO] compiling model....")
opt = RMSprop(lr = 0.001)
model.compile(loss = 'categorical_crossentropy',optimizer = opt,metrics =['accuracy'])
# 由于我们只训练新增的全连接层，因此，我们进行少量迭代
print("[INFO] training head...")
model.fit_generator(aug.flow(trainX,trainY,batch_size = 32),
                             validation_data = (testX,testY),epochs=25,
                             steps_per_epoch = len(trainX) //  32,verbose = 1)

# 评估模型
print("[INFO] evaluating after initialization...")
predictions = model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis =1),
                            predictions.argmax(axis =1),target_names=classNames))

# 全连接拟合完新速据之后，接下来
# 对整个网络进行微调
for layer in baseModel.layers[15:]:
    layer.trainable = True

# 从新编译模型
print("[INFO] re-compiling model ...")
opt = SGD(lr=0.001)
# 使用很小的学习率进行微调
model.compile(loss = 'categorical_crossentropy',optimizer = opt,
              metrics=['accuracy'])
# 对整个模型进行微调
print("[INFO] fine-tuning model...")
model.fit_generator(aug.flow(trainX,trainY,batch_size=32),
                    validation_data = (testX,testY),epochs = 100,
                    steps_per_epoch = len(trainX) // 32,verbose = 1)
# 评估微调后的模型结果
print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis =1),
        predictions.argmax(axis =1),target_names=classNames))

# 将模型保存到磁盘
print("[INFO] serializing model...")
model.save(args['model'])


