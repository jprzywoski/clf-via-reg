#!/usr/bin/env python3
from functools import partial
import sklearn
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSCanonical
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Lars
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.datasets import make_classification
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from RegressionClassifier import RegressionClassifier
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#
# Control variables
# n - Table number
# ratings - ratings per dataset; used for calculating the final normalized
# rating on all data sets
# Ratings are calculated as the ratio of accuracy divided by maximum accuracy
# for the given dataset; this gives us a normalized scale
# Final rating is the average rating over all datasets
# K - k value for k-fold CV
#
n = 1
K = 10
ratings = {}

data_loaders = [
    (partial(load_wine, return_X_y=True), "Wine"),
    (partial(load_iris, return_X_y=True),  "Iris"),
    (partial(load_breast_cancer, return_X_y=True), "Cancer"),
    (partial(load_digits, return_X_y=True), "Digits"),
    (partial(make_classification, n_samples=1000, random_state=42),
        "Classification"),
    (partial(make_circles, n_samples=1000, random_state=42), "Circles"),
    (partial(make_moons, n_samples=1000, random_state=42), "Moons"),
    (partial(make_blobs, n_samples=1000, random_state=42), "Blobs"),
]

#
# Load and split data
#
for loader, dataset in data_loaders:
    X, y = loader()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #
    # List of estimators to evaluate
    #
    estimators = [
        #
        # Baseline
        #
        DummyClassifier(strategy="most_frequent"),

        #
        # Regression classifiers
        #
        LinearRegression(),
        Ridge(),
        SGDRegressor(),
        ARDRegression(),
        BayesianRidge(),
        KernelRidge(),
        ElasticNet(),
        Lars(),
        Lasso(),
        CCA(),
        HuberRegressor(),
        LassoLarsIC(),
        PLSCanonical(),
        PLSRegression(),
        TheilSenRegressor(),
        TweedieRegressor(),
        # RANSACRegressor(),

        #
        # Proper classifiers
        #
        LinearDiscriminantAnalysis(),
        LogisticRegression(random_state=42),
        KNeighborsClassifier(),
        MLPClassifier(random_state=42),
        SVC(random_state=42),
        AdaBoostClassifier(random_state=42),
        RandomForestClassifier(random_state=42),

    ]

    #
    # Main loop
    # Evalute each of estimators using k-fold cross validation on the training
    # set and also calculate final test scores
    #
    results = []
    #
    # Constants used for accesing results
    #
    ESTIMATOR = 0
    TRAIN_ACCU = 1
    TEST_ACCU = 2
    for e in estimators:
        #
        # Pick a classifier from the list
        #
        if not sklearn.base.is_classifier(e):
            clf = RegressionClassifier(e)
        else:
            clf = e
        clfstr = str(e).split("(")[ESTIMATOR]
        #
        # Populate rankings dict
        #
        if clfstr not in ratings:
            ratings[clfstr] = []

        #
        # Run k-fold cross validation on X_train
        #
        scores = cross_val_score(clf, X_train, y_train, cv=K, n_jobs=-1)
        train_score = scores.mean()

        #
        # Get the X_test score
        #
        clf.fit(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        #
        # Combine the results
        #
        results.append([clfstr, train_score, test_score])

    #
    # Format results into a table
    #
    print("")
    print("{: <30}{: >15}{: >15}{: >15}".format(
        "Method", "{}-fold CV".format(K), "Test", "Rating"))
    #
    # Calculate max. value of all test accuracies - used in ratings
    #
    accus = []
    for it in results:
        accus.append(it[TEST_ACCU])
    max_accu = max(accus)
    #
    # Print final results
    #
    for res in sorted(results, key=lambda e: e[TEST_ACCU], reverse=True):
        rating = 100 * (res[TEST_ACCU] / max_accu)
        ratings[res[ESTIMATOR]].append(rating)
        print("{: <30}{:15.2%}{:15.2%}{:15.2f}".format(
            res[ESTIMATOR], res[TRAIN_ACCU], res[TEST_ACCU], rating))
    print("")
    print("Table {}: {}".format(n, dataset))
    n += 1
    print("")

#
# Print final classifier ratings calculated over all tested datasets
#
CLF = 0
RATING = 1
final_rating = []
for clf, ratings in ratings.items():
    final_rating.append([clf, sum(ratings) / len(ratings)])
final_rating.sort(key=lambda e: e[RATING], reverse=True)

print("")
print("{: <30}{: >15}".format("Method", "Rating"))
for it in final_rating:
    print("{: <30}{:15.2f}".format(it[CLF], it[RATING]))
print("")
print("Table {}: Final classifier rating over all datasets".format(n))
print("")
