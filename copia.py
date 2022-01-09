
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
#prv=TSNE(n_components=2, *, perplexity=30.0, early_exaggeration=12.0, learning_rate='warn', n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')[source]
X_embedded = TSNE(n_components=2, learning_rate='auto',  init='random').fit_transform(x)
print(X_embedded)
# define the pipeline
