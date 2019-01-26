import numpy as np
import sklearn
import sklearn.datasets
import sklearn.svm
import sklearn.metrics
import matplotlib.pyplot as plt



digits = sklearn.datasets.load_digits()
n_samples = len(digits.images)
x = digits.images.reshape(n_samples, -1)
x_train = x[:n_samples // 2]
y_train = digits.target[:n_samples // 2]
x_test = x[n_samples // 2:]
y_test =  digits.target[n_samples // 2 :]


svm = sklearn.svm.SVC(kernel='linear', gamma=0.01, C=100)
svm.fit(x_train, y_train)

predict = svm.predict(x_test)
accu = np.sum(y_test == predict) / len(y_test)

print(sklearn.metrics.classification_report(y_test, predict))
print(sklearn.metrics.confusion_matrix(y_test, predict))
print(accu)
