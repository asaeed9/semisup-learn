import numpy as np
import random
from frameworks.CPLELearning import CPLELearningModel
from sklearn.datasets.mldata import fetch_mldata
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import sklearn.svm
from methods.scikitWQDA import WQDA
from frameworks.SelfLearning import SelfLearningModel
import pandas as pd
from tables import *
import h5py
import math
from methods import scikitTSVM
from examples.plotutils import evaluate_and_plot


#cancer = fetch_mldata("Lung cancer (Ontario)")
kernel = "linear"
hearts = fetch_mldata("heart")

# print(hearts['target'])
# print(hearts['data'])

X = hearts['data']
ytrue = hearts['target']
ytrue[ytrue<0]=0
# label a few points
labeled_N = 20
nsamples = math.floor(labeled_N/2)
ys = np.array([-1]*len(ytrue)) # -1 denotes unlabeled point
random_labeled_points = list(np.random.choice(np.where(ytrue == 0)[0], int(nsamples)))+ \
                        list(np.random.choice(np.where(ytrue == 1)[0], int(nsamples)))

ys[random_labeled_points] = ytrue[random_labeled_points]

# # supervised score
#basemodel = WQDA() # weighted Quadratic Discriminant Analysis
basemodel = SGDClassifier(loss='log', penalty='l1', tol=1e-3, max_iter=1000) # scikit logistic regression
basemodel.fit(X[random_labeled_points, :], ys[random_labeled_points])
print ("supervised log.reg. score", basemodel.score(X, ytrue))
#
# # fast (but naive, unsafe) self learning framework
# ssmodel = SelfLearningModel(basemodel)
# ssmodel.fit(X, ys)
# print ("self-learning log.reg. score", ssmodel.score(X, ytrue))

lbl =  "S3VM (Gieseke et al. 2012):"
print (lbl)
model = scikitTSVM.SKTSVM(kernel=kernel)
model.fit(X, ys.astype(int))
evaluate_and_plot(model, X, ys, ytrue, lbl, 2)
