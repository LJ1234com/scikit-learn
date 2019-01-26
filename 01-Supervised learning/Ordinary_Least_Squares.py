import numpy as np
import sklearn
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

diabetes = sklearn.datasets.load_diabetes()

x = diabetes.data[:, 2].reshape(-1, 1)
x_train = x[:-20]
y_train = diabetes.target[:-20]

x_test = x[-20:]
y_test = diabetes.target[-20:]

model = sklearn.linear_model.LinearRegression()
model.fit(x_train, y_train)
print(model.coef_, model.intercept_)

########## plot train and fitted curve ########
plt.subplot(121)
plt.scatter(x_train, y_train, c='black')
plt.plot(x_train, model.predict(x_train), c='g', linewidth=5)
plt.title('Train')
plt.xticks(())
plt.yticks(())

########## plot test curve ########
predict = model.predict(x_test)
plt.subplot(122)
plt.scatter(x_test, y_test, c='black')
plt.plot(x_test, predict, c='g', linewidth=5)
plt.title('Test')
plt.xticks(())
plt.yticks(())

plt.show()
