import numpy as np
import sklearn
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

x = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

alphas = np.logspace(-10, -2, 200)
coefs = []

for alpha in alphas:
    ridge = sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(x, y)
    coefs.append(ridge.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])

plt.xlabel('alpha')
plt.ylabel('weight')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')

plt.show()

## Setting the regularization parameter: generalized Cross-Validation
import numpy as np
import sklearn
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

x = np.array([0, 5, 10, 15, 20, 25]).reshape(-1, 1)
y = [0, 7, 9, 15, 22, 24]

model = sklearn.linear_model.LinearRegression()
model.fit(x, y)

ridge1 = sklearn.linear_model.Ridge(alpha=50)
ridge1.fit(x, y)

ridge2 = sklearn.linear_model.RidgeCV(alphas=[100], cv=3)
ridge2.fit(x, y)
print(ridge2.alpha_)

plt.scatter(x, y)
plt.plot(x, model.predict(x), c='g', linewidth=3, label='linear_regression')
plt.plot(x, ridge1.predict(x), c='r', linewidth=3, label='ridge1_regression')
plt.plot(x, ridge2.predict(x), c='b', linewidth=3, label='ridge2_regression')
plt.legend(loc='best')
plt.show()
