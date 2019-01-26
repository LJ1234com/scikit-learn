import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets.samples_generator import make_blobs
from sklearn import linear_model
from sklearn import svm

X, y = sklearn.datasets.samples_generator.make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
model = sklearn.linear_model.SGDClassifier(loss='hinge', alpha=0.01, max_iter=1000, fit_intercept=True)
model.fit(X, y)

xx = np.linspace(-1, 5, 10)
yy = np.linspace(-1, 5, 10)
X1, X2 = np.meshgrid(xx, yy)
z = np.empty(X1.shape)

for (i, j), val in np.ndenumerate(X1):
    x1 = val
    x2 = X2[i, j]
    p = model.decision_function([[x1, x2]])
    z[i, j ] = p[0]

levels = [-1.0, 0.0, 1.0]
linestyles = ['dashed', 'solid', 'dashed']
colors = 'k'
plt.contour(X1, X2, z, levels, colors=colors, linestyles=linestyles)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolor='black', s=20)

plt.axis('tight')
plt.show()



################################################
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolor='black', s=20)

model2 = svm.SVC(kernel='linear', C=1000)
model2.fit(X, y)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
XX, YY = np.meshgrid(xx, yy)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model2.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='g', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(model2.support_vectors_[:, 0], model2.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='r')
plt.show()
