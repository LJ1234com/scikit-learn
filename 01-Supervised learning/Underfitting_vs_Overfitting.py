import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

def true_fun(x):
    return np.cos(1.5 * np.pi * x)

n_samples = 30
degrees = [1, 4, 15]

x = np.sort(np.random.rand(n_samples))
y = true_fun(x) + 0.1 * np.random.randn(n_samples)

plt.figure()
for i in range(len(degrees)):
    plt.subplot(1, len(degrees), i + 1)
    plt.xticks(); plt.yticks()
    polynomial_features = sklearn.preprocessing.PolynomialFeatures(degree=degrees[i], include_bias=False)
    model = sklearn.linear_model.LinearRegression()
    pipeline = sklearn.pipeline.Pipeline([('polynomial_features', polynomial_features), ('linear_regression', model)])
    pipeline.fit(x.reshape(-1, 1), y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, x[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10)



    x_test = np.linspace(0, 1, 100)
    plt.scatter(x, y, edgecolor='b',s =20, label='Samples')
    plt.plot(x_test, pipeline.predict(x_test.reshape(-1, 1)), label='Fitted')
    plt.plot(x_test, true_fun(x_test), label='True function')
    plt.xlabel('x'); plt.ylabel('y')
    plt.xlim(0, 1); plt.ylim(-2, 2)
    plt.legend()
    plt.title('Degree {}\nMSE={:.2e}(+/-{:.2e})'.format(degrees[i],-scores.mean(), scores.std()))


plt.show()
