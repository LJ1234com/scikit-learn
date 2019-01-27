import numpy as np
import sklearn
import sklearn.datasets
import sklearn.neighbors



iris = sklearn.datasets.load_iris()
data = iris.data
target = iris.target

index = np.random.permutation(len(data))
x_train = data[index[:-10]]
y_train = target[index[:-10]]
x_test = data[index[-10:]]
y_test = target[index[-10:]]

knn = sklearn.neighbors.KNeighborsClassifier()
knn.fit(x_train, y_train)
predict = knn.predict(x_test)

print(predict)
print(y_test)
