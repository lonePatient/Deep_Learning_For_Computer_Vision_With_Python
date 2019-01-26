#encoding:utf-8
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor as IAP
from pyimagesearch.preprocessing import AspectAwarePreprocessor as AAP
from pyimagesearch.preprocessing import ImageMove as IM
from pyimagesearch.datasets import SimpleDatasetLoader as SDL
from pyimagesearch.nn.conv import MiniVGGNet as MVN
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap  = argparse.ArgumentParser()
ap.add_argument('-d','--dataset',required=True,help='path to input dataset')
args = vars(ap.parse_args())


print('[INFO] moving image to label folder.....')
im  = IM.MoveImageToLabel(dataPath=args['dataset'])
im.makeFolder()
im.move()

print("[INFO] loading images...")
imagePaths = [ x for x in list(paths.list_images(args['dataset'])) if x.split(os.path.sep)[-2] !='jpg']
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths ]
classNames = [str(x) for x in np.unique(classNames)]


aap = AAP.AspectAwarePreprocesser(64,64)
iap = IAP.ImageToArrayPreprocess()

sdl = SDL.SimpleDatasetLoader(preprocessors = [aap,iap])
(data,labels) = sdl.load(imagePaths,verbose = 500)
data = data.astype('float') / 255.0

(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.25,random_state  =43)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model....")
opt = SGD(lr=0.05)
model = MVN.MiniVGGNet.build(width = 64,height = 64,depth =3,classes = len(classNames))
model.compile(loss = 'categorical_crossentropy',optimizer = opt,
              metrics = ['accuracy'])

print('[INFO] training network....')
H = model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=32,epochs=100,verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=classNames))
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
