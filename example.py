#!/usr/bin/env python3
from RegressionClassifier import RegressionClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from utils import load_data

#
# Load and split data
#
X_train, y_train, X_test, y_test, colnames = load_data("thyroid.csv")

#
# Hyperparamter tuning
#
k = X_train.shape[1]
param_grid = {
    'base_estimator': [Ridge(alpha=1.0/k**2), Ridge(alpha=1.0/k),
                       Ridge(alpha=k**0.5), Ridge(alpha=k)]
}
grid = GridSearchCV(
    RegressionClassifier(Ridge()),
    param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

print("*** Fit Summary ***")
print("Best k-fold score: {}".format(grid.best_score_))
print("Test score: {}".format(grid.score(X_test, y_test)))
print("Best hyper-parameters: {}".format(grid.best_params_))
