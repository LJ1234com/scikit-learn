import numpy as np
import sklearn
import sklearn.datasets
import sklearn.svm

# estimator exposes a score method that can judge the quality of the fit (or the prediction) on new data.
# Bigger is better.

digits = sklearn.datasets.load_digits()
x = digits.data
y = digits.target
svm = sklearn.svm.SVC(kernel='linear', C=1)
svm.fit(x[:-100], y[:-100])
score = svm.score(x[-100:], y[-100:])
print(score)

# KFold cross-validation.
kfolds = 3
x_folds = np.array_split(x, kfolds)
y_folds = np.array_split(y, kfolds)


for k in range(kfolds):
    x_train = list(x_folds)
    x_test = x_train.pop(k)
    x_train = np.concatenate(x_train)  # merge the two left folds

    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)  # merge the two left folds

    svm.fit(x_train, y_train)
    score = svm.score(x_test, y_test)
    print(score)
