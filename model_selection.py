import pandas as pd
import numpy as np
from statistics import median
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import sys
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import ElasticNet, RidgeClassifier, Lasso
# from sklearn.svm import SVC
from xgboost import XGBClassifier
from data_wrangling import data_wrangler
from sklearn.linear_model import LogisticRegression,ElasticNet,RidgeClassifier,SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

# Possible models
ridge = RidgeClassifier(alpha = .5)
en = ElasticNet(alpha = .5)
svc = SVC(C = 0.7)
xgb = XGBClassifier(reg_lambda = .2,reg_alpha = .4,n_estimators = 100,min_child_weight = 2, max_depth = 3,learning_rate = .1)
logistic = LogisticRegression(penalty = 'l1')
models = [ridge,en,xgb]

def objective(space):

    eval_set  = [(xtrain,ytrain), (xvalid.values,yvalid)]
    xgb = XGBClassifier(n_estimators = 10000,
                        reg_lambda = space['reg_lambda'],
                        reg_alpha = space['reg_alpha'],
                        max_depth = space['max_depth'],
                        min_child_weight = space['min_child_weight'],
                        subsample = space['subsample'])
    
    xgb.fit(xtrain,ytrain,eval_set = eval_set,eval_metric="auc", early_stopping_rounds=30)

    
    pred = xgb.predict_proba(xvalid.values)[:,1]
    auc = roc_auc_score(yvalid, pred)
    print("SCORE:", auc)

    return{'loss':1-auc, 'status': STATUS_OK }


space ={
        'max_depth': hp.choice('max_depth', np.arange(1, 30, dtype=int)),
        'min_child_weight': hp.quniform ('x_min_child', 1, 10, 1),
        'subsample': hp.uniform ('x_subsample', 0.8, 1),
        'reg_lambda': hp.uniform('reg_lambda',.05,1),
        'reg_alpha': hp.uniform('reg_alpha',.05,1)
    }

{'max_depth': 14, 'reg_alpha': 0.9024871933247666, 'reg_lambda': 0.14620126870865735,
'x_min_child': 6.0, 'x_subsample': 0.9264182026793351}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print(best)