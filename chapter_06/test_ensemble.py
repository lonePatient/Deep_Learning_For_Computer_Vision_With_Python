# -*- coding: utf-8 -*-
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import  load_model
from keras.datasets import cifar10
import numpy as np
import argparse
import glob
import os


ap = argparse.ArgumentParser()
ap.add_argument('-m','--models',required=True,help='path to models directory')
args = vars(ap.parse_args())

(testX,testY) = cifar10.load_data()[1]
testX = testX.astype('float') /255.0

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]
# 类别one-hot编码
lb = LabelBinarizer()
testY = lb.fit_transform(testY)

modelPaths = os.path.sep.join([args['models'],"*.model"])
modelPaths = list(glob.glob(modelPaths))
models = []

for (i,modelPath) in enumerate(modelPaths):
    print("[INFO] loading model {}/{}".format(i+1,len(modelPaths)))
    models.append(load_model(modelPath))

print("[INFO] evaluating ensemble...")
predictions = []
#遍历模型
for model in models:
    # 模型预测
    predictions.append(model.predict(testX,batch_size=64))

# 平均所有模型结果
predictions = np.average(predictions,axis = 0)
# 模型结果
print(classification_report(testY.argmax(axis =1),
                            predictions.argmax(axis=1),target_names=labelNames))