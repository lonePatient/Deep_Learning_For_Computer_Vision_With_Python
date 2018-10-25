#encoding:utf-8
import cv2
import json
import numpy as np
from sklearn.model_selection import train_test_split
from config import imagenet_alexnet_config as config
from pyimagesearch.utils.imagenethelper import ImageNetHelper
from pyimagesearch.utils.imagenettfrecord import ImageNetTfrecord


print('[INFO] loading image paths...')
inh = ImageNetHelper(config)
(trainPaths,trainLabels) = inh.buildTrainingSet()
(valPaths,valLabels) = inh.buildValidationSet()

print('[INFO] constructing splits...')
split = train_test_split(trainPaths,trainLabels,
                         test_size = config.NUM_TEST_IMAGES,stratify=trainLabels,
                         random_state=42)
trainPaths,testPaths,trainLabels,testLabels = split

datasets = [
    ('train',trainPaths,trainLabels,config.TRAIN_TFRECORD),
    ('val',valPaths,valLabels,config.VAL_TFRECORD),
    ('test',testPaths,testLabels,config.TEST_TFRECORD)
]

(R,G,B) = [],[],[]

for (dType,paths,labels,outputPath) in datasets:
    print('[INFO] building {}...'.format(outputPath))
    inr = ImageNetTfrecord(outputPath)

    probar = inh._pbar(name='Building %s List: '%dType,maxval=len(paths))
    for (i,(path,label)) in enumerate(zip(paths,labels)):
        inr._save_one(label=label, filename=path, isTrain=True)
        if dType == 'train':
            image = cv2.imread(path)
            (b,g,r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        probar.update(i)
    probar.finish()
    inr.tfwriter.close()

print('[INFO] serializing means...')
D = {'R':np.mean(R),'G':np.mean(G),'B':np.mean(B)}
with open(config.DATASET_MEAN,'w') as f:
    f.write(json.dumps)
