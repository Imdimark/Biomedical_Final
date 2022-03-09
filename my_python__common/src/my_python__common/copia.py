
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
from sklearn.manifold import TSNE
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
train_signal = x[0:split, :]
test_x = x[split:, :]
_pc=2
print('lunghezza')
print(len(train_signal[0]))
#pca = PCA(n_components=len(train_signal[0]))
a = TSNE(n_components=_pc, learning_rate='auto',  init='random').fit_transform(x)
#a è train score out
tsne = TSNE(n_components=_pc, learning_rate='auto',  init='random')
print(tsne.fit(train_signal))
print(a)
print(tsne)
coeff = tsne.embedding_

train_score = np.matmul((train_signal - np.mean(train_signal, 0)), coeff)
train_score[:, _pc:] = 0
train_score_out = train_score[:, 0:_pc]
train_signal_rec = np.matmul(train_score, coeff.T) + np.mean(train_signal, 0)


test_score = np.matmul((test_x - np.mean(train_signal, 0)), coeff)
test_score[:, _pc:] = 0
test_score_out = test_score[:, 0:_pc]
test_signal_rec = np.matmul(test_score, coeff.T) + np.mean(train_signal, 0)

         



