# Classification via Regression for sklearn

Jakub Przywóski
Sun 26 Dec 06:52:30 GMT 2021

This repository contains source code for a sklearn compatible wrapper for
regression estimators that can turn them into classifiers.

A benchmark suite is provided for comparing the results of this method
against a bunch of other classifiers. Several toy datasets are used for comparison.

Original implementation came from this tread on https://stats.stackexchange.com/:

https://stats.stackexchange.com/questions/22381/why-not-approach-classification-through-regression

# Instructions

It is as simple as this:

		>>> from sklearn.linear_model import BayesianRidge
		>>> from RegressionClassifier import RegressionClassifier
		>>> from sklearn.datasets import load_breast_cancer
		>>> X, y = load_breast_cancer(return_X_y=True)
		>>> clf = RegressionClassifier(BayesianRidge()).fit(X, y)
		>>> clf.score(X, y)
		0.961335676625659

# Running the benchmark suite

	$ ./testRegressionClassifier.py 2>/dev/null

	Method                             10-fold CV           Test         Rating
	LassoLarsIC                            98.33%        100.00%         100.00
	LinearDiscriminantAnalysis             97.50%        100.00%         100.00
	LogisticRegression                     95.83%        100.00%         100.00
	MLPClassifier                          95.83%        100.00%         100.00
	RandomForestClassifier                 96.67%        100.00%         100.00
	LinearRegression                       99.17%         98.31%          98.31
	Ridge                                  97.50%         98.31%          98.31
	ARDRegression                          99.17%         98.31%          98.31
	BayesianRidge                          98.33%         98.31%          98.31
	Lars                                   99.17%         98.31%          98.31
	CCA                                    98.33%         98.31%          98.31
	HuberRegressor                         99.17%         98.31%          98.31
	PLSRegression                          97.50%         98.31%          98.31
	TheilSenRegressor                      99.17%         98.31%          98.31
	KNeighborsClassifier                   95.00%         98.31%          98.31
	SVC                                    97.42%         98.31%          98.31
	SGDRegressor                           94.09%         96.61%          96.61
	KernelRidge                            94.17%         96.61%          96.61
	PLSCanonical                           94.92%         96.61%          96.61
	AdaBoostClassifier                     88.26%         91.53%          91.53
	TweedieRegressor                       77.50%         84.75%          84.75
	DummyClassifier                        39.47%         40.68%          40.68
	ElasticNet                             39.77%         40.68%          40.68
	Lasso                                  39.77%         40.68%          40.68

	Table 1: Wine


	Method                             10-fold CV           Test         Rating
	LinearDiscriminantAnalysis             98.00%         98.00%         100.00
	KNeighborsClassifier                   95.00%         98.00%         100.00
	SVC                                    96.00%         98.00%         100.00
	RandomForestClassifier                 95.00%         98.00%         100.00
	AdaBoostClassifier                     93.00%         92.00%          93.88
	LogisticRegression                     93.00%         90.00%          91.84
	MLPClassifier                          92.00%         90.00%          91.84
	Ridge                                  83.00%         84.00%          85.71
	TheilSenRegressor                      81.00%         84.00%          85.71
	BayesianRidge                          84.00%         82.00%          83.67
	TweedieRegressor                       81.00%         82.00%          83.67
	LinearRegression                       82.00%         80.00%          81.63
	ARDRegression                          84.00%         80.00%          81.63
	Lars                                   82.00%         80.00%          81.63
	CCA                                    83.00%         80.00%          81.63
	HuberRegressor                         78.00%         80.00%          81.63
	LassoLarsIC                            83.00%         80.00%          81.63
	PLSCanonical                           82.00%         80.00%          81.63
	PLSRegression                          80.00%         78.00%          79.59
	SGDRegressor                           67.00%         70.00%          71.43
	KernelRidge                            63.00%         70.00%          71.43
	DummyClassifier                        35.00%         30.00%          30.61
	ElasticNet                             27.00%         30.00%          30.61
	Lasso                                  27.00%         30.00%          30.61

	Table 2: Iris


	Method                             10-fold CV           Test         Rating
	MLPClassifier                          96.05%         97.87%         100.00
	SVC                                    96.84%         97.87%         100.00
	ARDRegression                          96.58%         96.81%          98.91
	KNeighborsClassifier                   96.85%         96.81%          98.91
	PLSRegression                          95.26%         95.74%          97.83
	LogisticRegression                     96.58%         95.74%          97.83
	RandomForestClassifier                 94.49%         95.74%          97.83
	Ridge                                  96.05%         95.21%          97.28
	BayesianRidge                          96.05%         95.21%          97.28
	LassoLarsIC                            95.00%         95.21%          97.28
	AdaBoostClassifier                     97.64%         95.21%          97.28
	LinearRegression                       95.79%         94.15%          96.20
	HuberRegressor                         95.00%         94.15%          96.20
	LinearDiscriminantAnalysis             95.53%         94.15%          96.20
	TheilSenRegressor                      95.53%         93.62%          95.65
	PLSCanonical                           93.69%         92.55%          94.57
	SGDRegressor                           92.11%         90.43%          92.39
	CCA                                    87.91%         90.43%          92.39
	KernelRidge                            88.70%         84.57%          86.41
	TweedieRegressor                       77.14%         80.85%          82.61
	DummyClassifier                        61.94%         64.36%          65.76
	ElasticNet                             61.93%         64.36%          65.76
	Lasso                                  61.93%         64.36%          65.76
	Lars                                   52.23%         50.00%          51.09

	Table 3: Cancer


	Method                             10-fold CV           Test         Rating
	KNeighborsClassifier                   97.92%         99.33%         100.00
	SVC                                    98.84%         98.48%          99.15
	RandomForestClassifier                 96.92%         97.47%          98.14
	MLPClassifier                          97.59%         96.97%          97.63
	LogisticRegression                     96.34%         96.63%          97.29
	LinearDiscriminantAnalysis             94.93%         94.95%          95.59
	LassoLarsIC                            93.02%         94.44%          95.08
	KernelRidge                            93.02%         94.28%          94.92
	Ridge                                  93.10%         93.94%          94.58
	CCA                                    92.27%         93.94%          94.58
	LinearRegression                       92.93%         93.77%          94.41
	BayesianRidge                          93.02%         93.77%          94.41
	ARDRegression                          92.85%         93.27%          93.90
	SGDRegressor                           92.27%         93.10%          93.73
	TheilSenRegressor                      92.18%         93.10%          93.73
	Lars                                   85.03%         90.91%          91.53
	PLSRegression                          90.61%         90.91%          91.53
	TweedieRegressor                       87.28%         87.37%          87.97
	PLSCanonical                           87.95%         86.03%          86.61
	HuberRegressor                         35.75%         34.51%          34.75
	AdaBoostClassifier                     34.25%         34.18%          34.41
	ElasticNet                              7.81%          9.43%           9.49
	Lasso                                   7.81%          9.43%           9.49
	DummyClassifier                        10.47%          9.26%           9.32

	Table 4: Digits


	Method                             10-fold CV           Test         Rating
	RandomForestClassifier                 89.85%         86.36%         100.00
	PLSRegression                          87.16%         85.76%          99.30
	MLPClassifier                          88.21%         85.76%          99.30
	Ridge                                  87.46%         85.45%          98.95
	LogisticRegression                     87.76%         85.45%          98.95
	AdaBoostClassifier                     86.27%         85.45%          98.95
	LinearRegression                       87.31%         85.15%          98.60
	ARDRegression                          88.06%         85.15%          98.60
	BayesianRidge                          87.31%         85.15%          98.60
	KernelRidge                            87.16%         85.15%          98.60
	Lars                                   87.31%         85.15%          98.60
	CCA                                    87.31%         85.15%          98.60
	TheilSenRegressor                      87.76%         85.15%          98.60
	LinearDiscriminantAnalysis             87.31%         85.15%          98.60
	HuberRegressor                         87.31%         84.85%          98.25
	SVC                                    86.87%         84.85%          98.25
	LassoLarsIC                            88.51%         84.55%          97.89
	SGDRegressor                           86.57%         82.73%          95.79
	PLSCanonical                           85.22%         81.21%          94.04
	TweedieRegressor                       82.09%         79.70%          92.28
	KNeighborsClassifier                   80.00%         77.88%          90.18
	DummyClassifier                        50.30%         49.39%          57.19
	ElasticNet                             46.27%         49.39%          57.19
	Lasso                                  46.27%         49.39%          57.19

	Table 5: Classification


	Method                             10-fold CV           Test         Rating
	KNeighborsClassifier                  100.00%        100.00%         100.00
	SVC                                   100.00%        100.00%         100.00
	AdaBoostClassifier                     99.25%         98.79%          98.79
	RandomForestClassifier                 98.81%         97.88%          97.88
	MLPClassifier                          51.49%         81.21%          81.21
	LogisticRegression                     50.45%         51.21%          51.21
	Ridge                                  50.00%         50.61%          50.61
	LinearRegression                       50.00%         50.00%          50.00
	Lars                                   50.00%         50.00%          50.00
	HuberRegressor                         50.00%         50.00%          50.00
	PLSRegression                          50.00%         50.00%          50.00
	LinearDiscriminantAnalysis             50.90%         50.00%          50.00
	PLSCanonical                           48.36%         48.48%          48.48
	CCA                                    48.51%         48.18%          48.18
	SGDRegressor                           50.60%         47.88%          47.88
	DummyClassifier                        51.34%         47.27%          47.27
	ARDRegression                          51.34%         47.27%          47.27
	BayesianRidge                          51.34%         47.27%          47.27
	ElasticNet                             51.34%         47.27%          47.27
	Lasso                                  51.34%         47.27%          47.27
	LassoLarsIC                            51.34%         47.27%          47.27
	TheilSenRegressor                      51.34%         47.27%          47.27
	TweedieRegressor                       51.34%         47.27%          47.27
	KernelRidge                            50.15%         46.67%          46.67

	Table 6: Circles


	Method                             10-fold CV           Test         Rating
	KNeighborsClassifier                  100.00%        100.00%         100.00
	SVC                                   100.00%        100.00%         100.00
	RandomForestClassifier                 99.70%         99.70%          99.70
	AdaBoostClassifier                     99.85%         98.18%          98.18
	MLPClassifier                          87.31%         90.61%          90.61
	LogisticRegression                     87.91%         89.70%          89.70
	LinearRegression                       87.46%         89.39%          89.39
	Ridge                                  87.46%         89.39%          89.39
	ARDRegression                          87.46%         89.39%          89.39
	BayesianRidge                          87.46%         89.39%          89.39
	Lars                                   87.46%         89.39%          89.39
	CCA                                    87.31%         89.39%          89.39
	LassoLarsIC                            87.46%         89.39%          89.39
	PLSRegression                          87.46%         89.39%          89.39
	LinearDiscriminantAnalysis             87.31%         89.39%          89.39
	TheilSenRegressor                      87.31%         89.09%          89.09
	HuberRegressor                         85.82%         87.88%          87.88
	PLSCanonical                           85.37%         87.27%          87.27
	TweedieRegressor                       85.97%         86.97%          86.97
	SGDRegressor                           84.63%         86.67%          86.67
	KernelRidge                            76.57%         76.06%          76.06
	DummyClassifier                        51.34%         47.27%          47.27
	ElasticNet                             51.34%         47.27%          47.27
	Lasso                                  51.34%         47.27%          47.27

	Table 7: Moons


	Method                             10-fold CV           Test         Rating
	LinearRegression                      100.00%        100.00%         100.00
	Ridge                                 100.00%        100.00%         100.00
	SGDRegressor                          100.00%        100.00%         100.00
	ARDRegression                         100.00%        100.00%         100.00
	BayesianRidge                         100.00%        100.00%         100.00
	Lars                                  100.00%        100.00%         100.00
	CCA                                   100.00%        100.00%         100.00
	HuberRegressor                        100.00%        100.00%         100.00
	LassoLarsIC                           100.00%        100.00%         100.00
	PLSCanonical                          100.00%        100.00%         100.00
	PLSRegression                         100.00%        100.00%         100.00
	TheilSenRegressor                     100.00%        100.00%         100.00
	TweedieRegressor                       98.36%        100.00%         100.00
	LinearDiscriminantAnalysis            100.00%        100.00%         100.00
	LogisticRegression                    100.00%        100.00%         100.00
	KNeighborsClassifier                  100.00%        100.00%         100.00
	MLPClassifier                         100.00%        100.00%         100.00
	SVC                                   100.00%        100.00%         100.00
	AdaBoostClassifier                    100.00%        100.00%         100.00
	RandomForestClassifier                100.00%        100.00%         100.00
	KernelRidge                            67.31%         65.45%          65.45
	DummyClassifier                        34.63%         30.91%          30.91
	ElasticNet                             32.54%         30.91%          30.91
	Lasso                                  32.54%         30.91%          30.91

	Table 8: Blobs


	Method                                 Rating
	SVC                                     99.46
	RandomForestClassifier                  99.19
	KNeighborsClassifier                    98.42
	MLPClassifier                           95.07
	LinearDiscriminantAnalysis              91.22
	LogisticRegression                      90.85
	Ridge                                   89.35
	AdaBoostClassifier                      89.13
	BayesianRidge                           88.62
	LassoLarsIC                             88.57
	LinearRegression                        88.57
	TheilSenRegressor                       88.55
	ARDRegression                           88.50
	PLSRegression                           88.24
	CCA                                     87.88
	PLSCanonical                            86.15
	SGDRegressor                            85.56
	TweedieRegressor                        83.19
	Lars                                    82.57
	HuberRegressor                          80.88
	KernelRidge                             79.52
	ElasticNet                              41.15
	Lasso                                   41.15
	DummyClassifier                         41.13

	Table 9: Final classifier rating over all datasets

# Remarks

The idea kind of works. For some datasets the performance can be competitive
against proper classifiers, but usually we get pretty average results.

The advantage is training speed - most regressor have no hyper parameters to tune,
so there is no need for grid search optimization.

I did not tune the hyper parameters for any of the classifiers - if I did
the final rankings would likely be different.
