#encoding:utf-8
from pyimagesearch.utils.ranked import rank5_accuracy
import argparse
import pickle
import h5py

# 解析命令行参数
ap = argparse.ArgumentParser()
ap.add_argument('-d','--db',required=True,help='path HDF5 databases')
ap.add_argument('-m','--model',required=True,help = 'path to pre-trained model')
args = vars(ap.parse_args())

# 加载模型
print("[INFO] loading pre-trained model...")
model = pickle.loads(open(args['model'],'rb').read())

db = h5py.File(args['db'],'r')
i = int(db['labels'].shape[0] * 0.75)
# 继续宁预测
print ("[INFO] predicting....")
preds = model.predict_proba(db['features'][i:])
(rank1,rank5) = rank5_accuracy(preds,db['labels'][i:])
# 结果打印
print("[INFO] rank-1:{:.2f}%".format(rank1 * 100))
print("[INFO] rank-5:{:.2f}%".format(rank5 * 100))
db.close()

