
# evaluate svd with logistic regression algorithm for classification
import numpy as np
import pandas as pd
import os
import time

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

# define dataset
#X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=7)
# read calibration file and remove all the initial zero rows
calibPath = os.path.dirname(os.path.abspath(__file__)) + "/calib/"
xp = list(pd.read_csv(calibPath + 'Calib.txt', sep=' ', header=None).values)
x = [i for i in xp if all(i)]
x = np.array(x)

# randomly shuffle input
np.random.shuffle(x)

# define train/test split
thr = 80
split = int(len(x) * thr / 100)
train_x = x[0:split, :]
test_x = x[split:, :]

print(train_x)
print(train_x.shape)
print(type(train_x))
# define the pipeline
sp = [('svd', TruncatedSVD(n_components=2)), ('m', LogisticRegression())]
model = Pipeline(steps=sp)
print(model.get_params)
# evaluate model

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
print(cv.get_n_splits)
n_sc = cross_val_score(model, train_x, scoring='accuracy', cv=8, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_sc), np(n_sc)))
print(model.get_params)