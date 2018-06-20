# -*- coding: utf-8 -*-
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-d","--db",required = True,help = 'path HDF5 database')
ap.add_argument("-m","--model",required = True,help = 'path to output model')
ap.add_argument('-j','--jobs',type=int,default=-1,help='# of jobs to run when tuning hyperparameters')
args = vars(ap.parse_args())

db = h5py.File(args['db'],'r')
i = int(db['labels'].shape[0] * 0.75)
# 模型调优
print("[INFO] tuning hyperparameters...")
params = {"C":[0.1,1.0,10.0,100.0,1000.0,10000.0]}
model = GridSearchCV(LogisticRegression(),params,cv=3,n_jobs=args['jobs'])
model.fit(db['features'][:i],db['labels'][:i])
print("[INFO] best hyperparameters :{} ".format(model.best_estimator_))
# 评估模型
print("[INFO] evaluating...")
preds = model.predict(db['features'][i:])
print(classification_report(db['labels'][i:],preds,target_names=db['label_names']))

# 保存模型到磁盘
print("[INFO] saving model...")
f = open(args['model'],'wb')
f.write(pickle.dumps(model.best_estimator_))
f.close()
db.close()
