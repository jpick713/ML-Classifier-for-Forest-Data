# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 21:21:12 2019

@author: jpick
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.compose import ColumnTransformer
from boruta import BorutaPy
import pdpipe as pdp
import tensorflow as tf
import kerastuner as kt




np.random.seed(123)
# download data and look for columns with null values
data = pd.read_csv('csv files for training/train.csv')
X=data.iloc[:,:55]
y=data.iloc[:,-1]

count_NA=[sum(X[col].isna()) for col in X.columns]
print(count_NA)

print(X.columns)
print(X.transpose()[11:55])

#encoding and scaling the variable with pdpipe
X_new=pdp.OneHotEncode().apply(X)
y_new=LabelEncoder().fit_transform(y)
print(y_new.dtype)
numeric_cols=X.columns[1:11]
X[numeric_cols].head()
X_new=pdp.Scale('StandardScaler').apply(X_new)
X_new=pdp.ColDrop('Id').apply(X_new)
print(X_new.shape)

# Using selectors to pluck out most important features
selector_thres=VarianceThreshold()

selector_feat=selector_thres.fit(X_new, 0.0)
print(selector_feat.get_support())

K_selector=SelectKBest(k=13)
K_selector_feat=K_selector.fit(X_new,y_new)
print(K_selector_feat.get_support())
X_k=K_selector.transform(X_new)


# using boruta to select features
X_bor=X_new.values
y_bor=y_new


rf= RandomForestClassifier()
gbc=GradientBoostingClassifier()

feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
feat_selector.fit(X_bor,y_bor)

print(feat_selector.support_)
X_filtered=feat_selector.transform(X_bor)
print(X_filtered.shape)

# breaking the filtered datasets into train and validation sets
X_train, X_val, y_train, y_val=train_test_split(X_filtered,y_new,test_size=0.1)
X_k_train, X_k_val, y_K_train, y_k_val=train_test_split(X_k,y_new,test_size=0.1)

m1=rf.fit(X_train,y_train)
m2=gbc.fit(X_train,y_train)

p1=m1.predict(X_val)
p2=m2.predict(X_val)

print(accuracy_score(y_val,p1))
print(accuracy_score(y_val,p2))

def build_model(hp):
  model_type = hp.Choice('model_type', ['random_forest', 'ridge', 'gbc'])
  if model_type == 'random_forest':
    model = RandomForestClassifier(
        n_estimators=hp.Int('n_estimators', 70, 120, step=10),
        max_depth=hp.Int('max_depth', 15, 25)
        )
  elif model_type== 'gbc':
      model = GradientBoostingClassifier(
          n_estimators=hp.Int('n_estimators', 90, 180, step=10),
          max_depth=hp.Int('max_depth', 1, 8),
          learning_rate=hp.Float('lr', 1e-3, 1, sampling='log')
          )
  else:
    model = RidgeClassifier(
        alpha=hp.Float('alpha', 1e-3, 1, sampling='log'))
  return model

tuner = kt.tuners.Sklearn(
    oracle=kt.oracles.BayesianOptimization(
        objective=kt.Objective('score', 'max'),
        max_trials=10),
    hypermodel=build_model,
    scoring= make_scorer(accuracy_score),
    cv= StratifiedKFold(5),
    directory='.',
    project_name='my_proj')



tuner.search(X_train, y_train)
tuner.search(X_k_train, y_K_train)

best_model = tuner.get_best_models(num_models=1)[0]

best_model_2 = tuner.get_best_models(num_models=1)[0]

print(best_model)
print(best_model_2)

pbest=best_model.predict(X_val)
pbest_2=best_model_2.predict(X_k_val)

print(accuracy_score(y_val,pbest))

m1=RandomForestClassifier(n_estimators=120, max_depth=19).fit(X_k_train,y_K_train)

p1=m1.predict(X_k_val)

print(accuracy_score(y_k_val,p1))

print(accuracy_score(y_k_val,pbest_2))