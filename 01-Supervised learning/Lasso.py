import numpy as np
import sklearn
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

x = np.array([0, 5, 10, 15, 20, 25]).reshape(-1, 1)
y = [0, 7, 9, 15, 22, 24]

model = sklearn.linear_model.LinearRegression()
model.fit(x, y)

lasso1 = sklearn.linear_model.Lasso(alpha=5)
lasso1.fit(x, y)


plt.scatter(x, y)
plt.plot(x, model.predict(x), c='g', linewidth=3, label='linear_regression')
plt.plot(x, lasso1.predict(x), c='r', linewidth=3, label='lasso1_regression')
plt.legend(loc='best')
plt.show()
