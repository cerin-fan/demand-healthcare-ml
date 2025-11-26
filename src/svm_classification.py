import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#manage data
all_patients = pd.read_csv('NPHA-doctor-visits.csv')
# print(all_patients.head())
X = all_patients.drop('Number of Doctors Visited',axis=1)
y = all_patients['Number of Doctors Visited']
#standard
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# SVM
clf_cv = SVC(C=0.01, kernel='rbf', degree=3, gamma='auto', probability=True)

#K-FOLD
n_folds = 5

# GridSearchCV for hyperparametrics
# param_grid = {'C':[0.01, 0.1, 1, 10, 100, 1000]}
# clf_cv = SVC(kernel='rbf', degree=3, gamma='auto', probability=True)
# grid_search = GridSearchCV(clf_cv, param_grid, cv=5)
# grid_search.fit(Xs, y)
# print(f"Best C value: {grid_search.best_params_['C']}")

#predict
predictions = cross_val_predict(clf_cv, Xs, y, cv=n_folds)
for i,pre in enumerate(predictions):
    print(f"Predicted: {i}:{pre},true value: {y[i]}")

scores = cross_val_score(clf_cv, Xs, y, cv=n_folds,scoring='accuracy')
print(scores)
avg = (100*np.mean(scores),100*np.std(scores)/np.sqrt(scores.shape[0]))
print("Average score and standard deviation: (%.2f+-%.3f)%%" %avg)


