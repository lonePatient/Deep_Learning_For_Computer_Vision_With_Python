# -*- coding: utf-8 -*-
# 加载所需模块
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import argparse
import pickle
import h5py

#命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-d","--db",required = True,
                help = 'path HDF5 datasetbase')
ap.add_argument('-m','--model',required=True,
                help='path to output model')
ap.add_argument('-j','--jobs',type=int,default=-1,
                help = '# of jobs to run when tuning hyperparameters')
                
args = vars(ap.parse_args())

# 读取hdf5数据集
db = h5py.File(args['db'],'r')
i = int(db['labels'].shape[0] * 0.75)# 分割点

print("[INFO] tuning hyperparameters...")
# 正则化系数参数范围
params = {"C":[0.0001,0.001,0.01,0.1,1.0]}
# 网格搜索，进行调参
model = GridSearchCV(LogisticRegression(),params,cv =3,
                     n_jobs = args['jobs'])
model.fit(db['feature'][:i],db['labels'][:i])
print('[INFO] best hyperparameters: {}'.format(model.best_params_))

#性能结果
print('[INFO] evaluating...')
preds = model.predict(db['feature'][i:])
print(classification_report(db['labels'][i:],preds,
                            target_names = db['label_names']))
#计算准确度
acc = accuracy_score(db['labels'][i:],preds)
print('[INFO] score: {}'.format(acc))

# 保存模型
print('[INFO] saving model...')
with open(args['model'],'wb') as fw:
    fw.write(pickle.dumps(model.best_estimator_))
    
db.close()


