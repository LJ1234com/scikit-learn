import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.datasets
import sklearn.svm
import matplotlib.pyplot as plt


digits = sklearn.datasets.load_digits()
x = digits.data
y = digits.target

svc = sklearn.svm.SVC(kernel='linear')
Cs = np.logspace(-10, 0, 10)

scores = []
score_std = []
for C in Cs:
    svc.C = C
    score = sklearn.model_selection.cross_val_score(svc, x, y, cv=5)
    scores.append(np.mean(score))
    score_std.append(np.std(score))


plt.semilogx(Cs, scores)
plt.semilogx(Cs, np.array(scores) + np.array(score_std), 'b--')
plt.semilogx(Cs, np.array(scores) - np.array(score_std), 'b--')
plt.xlabel('C')
plt.ylabel('CV Score')
plt.show()
