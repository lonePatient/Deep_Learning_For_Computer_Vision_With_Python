#encoding:utf-8
# 记载所需模块
import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import resnet
from pyimagesearch.callbacks import epochcheckpoint as EPO
from pyimagesearch.callbacks import trainingmonitor as TM
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.models import load_model
import keras.backend as K
import numpy as np
import argparse

# 解析命令行参数
ap = argparse.ArgumentParser()
ap.add_argument('-c','--checkpoints',required=True,help='path to output checkpoint directory')
ap.add_argument('-m','--model',type=str,help='path to *specific* model checkpoint to load')
ap.add_argument('-s','--start_epoch',type=int,default =0,help='epoch to restart training as ')
args = vars(ap.parse_args())

# 加载train和test数据集
print('[INFO] loading CIFAR-10 data...')
((trainX,trainY),(testX,testY)) = cifar10.load_data()
# 转化为float
trainX = trainX.astype("float")
testX = testX.astype("float")
# 计算RGB通道均值
mean = np.mean(trainX,axis =0)
# 零均值化
trainX -= mean
testX -= mean

# 标签编码处理
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# 数据增强
aug = ImageDataGenerator(width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         horizontal_flip = True,
                         fill_mode='nearest')
# 若未指定checkpoints模型，则直接初始化模型
if args['model'] is None:
    print("[INFO] compiling model...")
    opt = SGD(lr=1e-1)
    model = resnet.ResNet.build(32,32,3,10,(9,9,9),(64,64,128,256),reg=0.0005)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics = ['accuracy'])
# 否则从磁盘中加载checkpoints模型
else:
    print("[INFO] loading {}...".format(args['model']))
    model = load_model(args['model'])
    # 更新学习率
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr,1e-5)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))
# 回调函数列表
callbacks = [
    # checkpoint
    EPO.EpochCheckpoint(args['checkpoints'],every = 5,startAt = args['start_epoch']),
    # 监控训练过程
    TM.TrainingMonitor("output/resnet56_cifar10.png",
                       jsonPath="output/resnet56_cifar10.json",
                       startAt = args['start_epoch'])
]
# 训练网络
print("[INFO] training network.....")
model.fit_generator(
    aug.flow(trainX,trainY,batch_size=128),
    validation_data = (testX,testY),
    steps_per_epoch = len(trainX) // 128,
    epochs = 10,
    callbacks = callbacks,
    verbose =1
)