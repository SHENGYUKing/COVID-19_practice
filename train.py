# -*- coding: utf-8 -*-

# Copyright 2020. SHENGYUKing.
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The dataset comes from https://github.com/ieee8023/covid-chestxray-dataset

import model as mod
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn import neighbors
# from sklearn.metrics import r2_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


def plt_learning_curve(model, pltname, x, y, ylim=None, cv=None, n_jobs=1, train_size=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(pltname)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(model, x, y,
                                                            cv=cv, n_jobs=n_jobs, train_sizes=train_size)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean+test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


DATA_PATH = './data/images_new/'
full_database = mod.load_database(DATA_PATH, 'max')
x, y = full_database[:, 1:], full_database[:, 0]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

estimator = neighbors.KNeighborsClassifier()
estimator.fit(x_train, y_train)

y_predict = estimator.predict(x_test)
scorce = estimator.score(x_test, y_test, sample_weight=None)
# scorce = r2_score(y_test, y_predict)

print('y_predict = ')
print(y_predict)

print('y_test = ')
print(y_test)

print('Acc: {}%' .format(scorce * 100))

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=1)
title = "Learnming Curves\n(@276samples,test_size=0.2,normalize=MAX)"
plt_learning_curve(estimator, title, x, y, ylim=(0.0, 1.0), n_jobs=1)
plt.legend()
plt.show()
