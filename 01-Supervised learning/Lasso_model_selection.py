import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC

diabetes = sklearn.datasets.load_diabetes()
x = diabetes.data
y = diabetes.target

rng = np.random.RandomState(42)
x = np.c_[x, rng.randn(x.shape[0], 14)]
x /= np.sqrt(np.sum(x ** 2, axis=0))  # normalize data as done by Lars to allow for comparison

# #############################################################################
# LassoLarsIC: least angle regression with BIC/AIC criterion
bic = sklearn.linear_model.LassoLarsIC(criterion='bic')
t1 = time.time()
bic.fit(x, y)
t_bic = time.time() - t1
alpha_bic = bic.alpha_

aic = sklearn.linear_model.LassoLarsIC(criterion='aic')
aic.fit(x, y)
alpha_aic = aic.alpha_

def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color, linewidth=3, label='%s criterion' % name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=3, label='alpha: %s estimate' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')

plt.figure()
plot_ic_criterion(aic, 'AIC', 'b')
plot_ic_criterion(bic, 'BIC', 'r')
plt.legend() # default is loc='best'
plt.title('Information-criterion for model selection (training time %.3fs)' % t_bic)
plt.show()

# #############################################################################
# LassoCV: coordinate descent
t1 = time.time()
model = sklearn.linear_model.LassoCV(cv=10)
model.fit(x, y)
t_cv = time.time() - t1
alphas_log = -np.log10(model.alphas_)

plt.figure()
plt.plot(alphas_log, model.mse_path_, ':')
plt.plot(alphas_log, model.mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha: CV estimate')

plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: coordinate descent ''(train time: %.2fs)' % t_cv)
plt.axis('tight')
plt.ylim(2300, 4000)
plt.show()

# #############################################################################
# LassoLarsCV: least angle regression
t1 = time.time()
model = LassoLarsCV(cv=10)
model.fit(x, y)
t_lasso_lars_cv = time.time() - t1
alphas_log = -np.log10(model.cv_alphas_)

plt.figure()
plt.plot(alphas_log, model.mse_path_, ':')
plt.plot(alphas_log, model.mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV')
plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: Lars (train time: %.2fs)' % t_lasso_lars_cv)
plt.axis('tight')
plt.ylim(2300, 4000)
plt.show()
