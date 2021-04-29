import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, precision_score, f1_score, confusion_matrix, roc_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def split(X, y, cond):
    indices_true = [i for i in range(len(y)) if y[i] > cond]
    indices_false = [i for i in range(len(y)) if y[i] < cond]
    return [X[indices_true, :], X[indices_false, :]]

df = pd.read_csv('week4-2.csv',comment='#') 
X1=df.iloc[:, 0]
X2=df.iloc[:, 1]
X=np.column_stack((X1, X2))
y=df.iloc[:, 2]
y = np.array(df.iloc[:, 2])
y = np.reshape(y, (-1, 1))

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Visualisation of Data', fontsize=14)
ax.set_xlabel('X1 (Normalised)', fontsize=12)
ax.set_ylabel('X2 (Normalised)', fontsize=12)
class1, class2 = split(X, y, 0)
ax.scatter(class1[:, 0], class1[:, 1], marker='o', c='C0', alpha=0.75)
ax.scatter(class2[:, 0], class2[:, 1], marker='o', c='C3', alpha=0.75)
ax.legend(['Class 1', 'Class 2'], loc='lower right')
plt.savefig('data_visualisation')
plt.show()

kf = KFold(n_splits = 5)
loss = []; loss_std = []
acc = []; acc_std = []
q_range = [1, 2, 3, 4, 5, 6]
for q in q_range:
    Xpoly_lr = PolynomialFeatures(q).fit_transform(X)
    logreg = LogisticRegression(C=1, penalty='l2', solver='lbfgs')
    curr_loss=[]; curr_acc=[]
    for train, test in kf.split(Xpoly_lr):
        logreg.fit(Xpoly_lr[train], y[train].ravel())
        y_pred = logreg.predict(Xpoly_lr[test])
        curr_loss.append(log_loss(y[test], y_pred))
        curr_acc.append(accuracy_score(y[test], y_pred))
    loss.append(np.array(curr_loss).mean())
    loss_std.append(np.array(curr_loss).std())
    acc.append(np.array(curr_acc).mean())
    acc_std.append(np.array(curr_acc).std())

fig = plt.figure(figsize=(14, 7))
fig.suptitle('Optimal Value of Polynomial Features Degree Parameter', fontsize=16)
ax = fig.add_subplot(1, 2, 1)
ax.set_title('Loss', fontsize=14)
ax.set_xlabel('q', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.errorbar(q_range, loss, yerr=loss_std, linewidth=2.0, capsize=5.0, elinewidth=2.0, markeredgewidth=2.0)
ax.scatter(q_range, loss, marker='o')
ax = fig.add_subplot(1, 2, 2)
ax.set_title('Accuracy', fontsize=14)
ax.set_xlabel('q', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.errorbar(q_range, acc, yerr=acc_std, linewidth=2.0, capsize=5.0, elinewidth=2.0, markeredgewidth=2.0)
ax.scatter(q_range, acc, marker='o')
plt.savefig('lr_q_cross_validation')
plt.show()

q_lr = q_range[np.argmax(np.diff(acc) < 0.0025)]
print('Optimal Value: %d\n' % (q_lr))
Xpoly_lr = PolynomialFeatures(q_lr).fit_transform(X)

kf = KFold(n_splits=5)
loss = []; loss_std = []
acc = []; acc_std = []
penalties = [0.1, 0.5, 1, 5, 10, 50, 100]
for penalty in penalties:
    logreg = LogisticRegression(C=penalty, penalty='l2', solver='lbfgs')
    curr_loss = []; curr_acc = []
    for train, test in kf.split(X):
        logreg.fit(Xpoly_lr[train], y[train].ravel())
        y_pred = logreg.predict(Xpoly_lr[test])
        curr_loss.append(log_loss(y[test], y_pred))
        curr_acc.append(accuracy_score(y[test], y_pred))
    loss.append(np.array(curr_loss).mean())
    loss_std.append(np.array(curr_loss).std())
    acc.append(np.array(curr_acc).mean())
    acc_std.append(np.array(curr_acc).std())

fig = plt.figure(figsize=(14, 7))
fig.suptitle('Optimal Value of Penalty Parameter', fontsize=16)
ax = fig.add_subplot(1, 2, 1)
ax.set_title('Loss', fontsize=14)
ax.set_xlabel('C', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.errorbar(penalties, loss, yerr=loss_std, linewidth=2.0, capsize=5.0, elinewidth=2.0, markeredgewidth=2.0)
ax.scatter(penalties, loss, marker='o')
ax = fig.add_subplot(1, 2, 2)
ax.set_title('Accuracy', fontsize=14)
ax.set_xlabel('C', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.errorbar(penalties, acc, yerr=acc_std, linewidth=2.0, capsize=5.0, elinewidth=2.0, markeredgewidth=2.0)
ax.scatter(penalties, acc, marker='o')
plt.savefig('lr_C_cross_validation')
plt.show()

penalty = penalties[np.argmax(np.diff(acc) < 0.0025)]
print('Optimal Value: %d\n' % (penalty))
logreg = LogisticRegression(C=penalty, penalty='l2', solver='lbfgs')

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Logistic Regresion Classifier', fontsize=14)
ax.set_xlabel('X1 (Normalised)', fontsize=12)
ax.set_ylabel('X2 (Normalised)', fontsize=12)
class1, class2 = split(X, y, 0)
ax.scatter(class1[:, 0], class1[:, 1], marker='o', c='C0', alpha=0.2)
ax.scatter(class2[:, 0], class2[:, 1], marker='o', c='C3', alpha=0.2)
class1, class2 = split(X[test], y_pred, 0)
ax.scatter(class1[:, 0], class1[:, 1], marker='+', c='C0', alpha=1.0)
ax.scatter(class2[:, 0], class2[:, 1], marker='+', c='C3', alpha=1.0)
ax.legend(['C1 True', 'C2 True', 'C1 Pred', 'C2 Pred'], loc='lower right')
plt.savefig('lr_classifier')
plt.show()



q_knn = 1
Xpoly_knn = PolynomialFeatures(q_knn).fit_transform(X)

kf = KFold(n_splits=5)
acc = []; acc_std = []
k_range = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    curr_acc = []
    for train, test in kf.split(X):
        knn.fit(Xpoly_knn[train], y[train].ravel())
        y_pred = knn.predict(Xpoly_knn[test])
        curr_acc.append(accuracy_score(y[test], y_pred))
    acc.append(np.array(curr_acc).mean())
    acc_std.append(np.array(curr_acc).std())

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Optimal Value of Neighbours', fontsize=14)
ax.set_xlabel('k', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.errorbar(k_range, acc, yerr=acc_std, linewidth=2.0, capsize=5.0, elinewidth=2.0, markeredgewidth=2.0)
ax.scatter(k_range, acc, marker='o')
plt.savefig('kNN_k_cross_validation')
plt.show()

k = k_range[np.argmax(np.diff(acc) < 0.0025)]
print('Optimal Value: %d\n' % (k))
knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')

kf = KFold(n_splits = 5)
acc = []; acc_std = []
q_range = [1, 2, 3, 4, 5, 6]
for q in q_range:
    Xpoly_knn = PolynomialFeatures(q).fit_transform(X)
    curr_acc=[]
    for train, test in kf.split(Xpoly_knn):
        knn.fit(Xpoly_knn[train], y[train].ravel())
        y_pred = knn.predict(Xpoly_knn[test])
        curr_acc.append(accuracy_score(y[test], y_pred))
    loss.append(np.array(curr_loss).mean())
    loss_std.append(np.array(curr_loss).std())
    acc.append(np.array(curr_acc).mean())
    acc_std.append(np.array(curr_acc).std())

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Optimal Value of Polynomial Features Degree Parameter', fontsize=14)
ax.set_xlabel('q', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.errorbar(q_range, acc, yerr=acc_std, linewidth=2.0, capsize=5.0, elinewidth=2.0, markeredgewidth=2.0)
ax.scatter(q_range, acc, marker='o')
plt.savefig('kNN_q_cross_validation')
plt.show()

q_knn = q_range[np.argmax(np.diff(acc) < 0.0025)]
print('Optimal Value: %d\n' % (q_knn))
Xpoly_knn = PolynomialFeatures(q_knn).fit_transform(X)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('kNN Classifier', fontsize=14)
ax.set_xlabel('X1 (Normalised)', fontsize=12)
ax.set_ylabel('X2 (Normalised)', fontsize=12)
class1, class2 = split(X, y, 0)
ax.scatter(class1[:, 0], class1[:, 1], marker='o', c='C0', alpha=0.2)
ax.scatter(class2[:, 0], class2[:, 1], marker='o', c='C3', alpha=0.2)
class1, class2 = split(X[test], y_pred, 0)
ax.scatter(class1[:, 0], class1[:, 1], marker='+', c='C0', alpha=1.0)
ax.scatter(class2[:, 0], class2[:, 1], marker='+', c='C3', alpha=1.0)
ax.legend(['C1 True', 'C2 True', 'C1 Pred', 'C2 Pred'], loc='lower right')
plt.savefig('kNN_classifier')
plt.show()



dummy = DummyClassifier(strategy='most_frequent')

rs = ShuffleSplit(n_splits=1, test_size=0.2)
train, test = next(rs.split(X, y))
print('Confusion Matrixes:')
print('%-20s %-10s %-10s %-10s %-10s' % ('Classifier', 'TP', 'FN', 'FP', 'TN'))
labels = ['Logistic Regression', 'kNN', 'Dummy']
models = [logreg, knn, dummy]
Xpolys = [Xpoly_lr, Xpoly_knn, PolynomialFeatures(1).fit_transform(X)]
for itr, model in enumerate(models):
    model.fit(Xpolys[itr][train], y[train].ravel())
    y_pred = model.predict(Xpolys[itr][test])
    tn, fp, fn, tp = confusion_matrix(y[test], y_pred).ravel()
    print('%-20s %-10d %-10d %-10d %-10d' % (labels[itr], tp, fn, fp, tn))
print()

print('Metrics:')
print('%-20s %-12s %-12s %-7s %-7s %-12s' % ('Classifier', 'Accuracy(%)', 'Precision(%)', 'TPR(%)', 'FPR(%)', 'F1 Score(%)'))
for itr, model in enumerate(models):
    y_pred = model.predict(Xpolys[itr][test])
    tn, fp, fn, tp = confusion_matrix(y[test], y_pred).ravel()
    acc = accuracy_score(y[test], y_pred)
    pre = precision_score(y[test], y_pred)
    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)
    f1_sc = f1_score(y[test], y_pred)
    print('%-20s %-12.2f %-12.2f %-7.2f %-7.2f %-13.2f' % (labels[itr], acc, pre, tpr, fpr, f1_sc))

    

rs = ShuffleSplit(n_splits=1, test_size=0.2)
train, test = next(rs.split(X, y))
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('ROC Curve', fontsize=14)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
logreg.fit(Xpoly_lr[train], y[train].ravel())
y_scores = logreg.decision_function(Xpoly_lr[test])
fpr, tpr, _ = roc_curve(y[test], y_scores)
ax.plot(fpr, tpr)
knn.fit(Xpoly_knn[train], y[train].ravel())
y_scores = knn.predict_proba(Xpoly_knn[test])[:, 1]
fpr, tpr, _ = roc_curve(y[test], y_scores)
ax.plot(fpr, tpr)
Xpoly = PolynomialFeatures(1).fit_transform(X)
dummy.fit(Xpoly[train], y[train].ravel())
y_scores = dummy.predict_proba(Xpoly[test])[:, 1]
fpr, tpr, _ = roc_curve(y[test], y_scores)
ax.plot(fpr, tpr, '--')
ax.legend(['Logistic Regression', 'kNN', 'Baseline'], loc='lower right')
plt.savefig('roc_curve')
plt.show()