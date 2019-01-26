import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sklearn
from sklearn.linear_model import BayesianRidge, LinearRegression

np.random.seed(0)
n_samples, n_features = 100, 100
x = np.random.randn(n_samples, n_features)
lambda_ = 4.0
w = np.zeros(n_features)
relevant_features = np.random.randint(0, n_features, 10)

for i in relevant_features:
    w[i] = stats.norm.rvs(loc=0, scale=1.0 / np.sqrt(lambda_))

alpha_ = 50.0
noise = stats.norm.rvs(loc=0, scale=1.0 / np.sqrt(alpha_), size=n_samples)
y = np.dot(x, w) + noise

# #############################################################################
## Fit the Bayesian Ridge Regression
clf = BayesianRidge(compute_score=True)
clf.fit(x, y)

## Fit OLS for comparison
ols = LinearRegression()
ols.fit(x, y)
clf = sklearn.linear_model.BayesianRidge(compute_score=True)
clf.fit(x, y)

# #############################################################################
# Plot true weights, estimated weights, histogram of the weights, and
# predictions with standard deviations
plt.figure()
plt.title('Weight of the model')
plt.plot(w,         c='gold',       linestyle='-', linewidth=2, label='Ground truth')
plt.plot(clf.coef_, c='lightgreen', linestyle='-', linewidth=1,  label='Bayesian Ridge estimate')
plt.plot(ols.coef_, c='navy',       linestyle='--', linewidth=1, label='OLS estimate')
plt.legend(loc='best', prop=dict(size=12))
plt.xlabel('Features')
plt.ylabel('Value of the weights')

plt.figure(figsize=(6, 5))
plt.title("Histogram of the weights")
plt.hist(clf.coef_, bins=n_features, color='gold', log=True, edgecolor='black')
plt.scatter(clf.coef_[relevant_features], np.full(len(relevant_features), 5.), c='navy', label="Relevant features")
plt.ylabel("Features")
plt.xlabel("Values of the weights")
plt.legend(loc="upper left")

plt.figure(figsize=(6, 5))
plt.title("Marginal log-likelihood")
plt.plot(clf.scores_, color='navy', linewidth=2)
plt.ylabel("Score")
plt.xlabel("Iterations")
plt.show()

# Plotting some predictions for polynomial regression
def f(x, noise_amount):
    y = np.sqrt(x) * np.sin(x)
    noise = np.random.normal(0, 1, len(x))
    return y + noise_amount * noise

degree = 10
X = np.linspace(0, 10, 100)
y = f(X, noise_amount=0.1)
clf_poly = BayesianRidge()
clf_poly.fit(np.vander(X, degree), y)

X_plot = np.linspace(0, 11, 25)
y_plot = f(X_plot, noise_amount=0)
y_mean, y_std = clf_poly.predict(np.vander(X_plot, degree), return_std=True)

plt.figure(figsize=(6, 5))
plt.plot(X_plot, y_plot, color='gold', linewidth=2, label="Ground Truth")
plt.errorbar(X_plot, y_mean, y_std, color='navy', label="Polynomial Bayesian Ridge Regression", linewidth=2)
plt.ylabel("Output y")
plt.xlabel("Feature X")
plt.legend(loc="lower left")
plt.show()
