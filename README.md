# Classification via Regression for sklearn

Jakub Przywóski
Sun 26 Dec 06:52:30 GMT 2021

This repository contains source code for a sklearn compatible wrapper for
regression estimators that can turn them into classifiers.

A benchmark suite is provided for comparing the results of this method
against a bunch of other classifiers. Several toy datasets are used for comparison.

Original implementation came from this tread on https://stats.stackexchange.com/:

https://stats.stackexchange.com/questions/22381/why-not-approach-classification-through-regression

Also similar idea is used by RidgeClassifier:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html

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
	LinearDiscriminantAnalysis             97.50%        100.00%         100.00
	LogisticRegression                     95.83%        100.00%         100.00
	MLPClassifier                          95.83%        100.00%         100.00
	RandomForestClassifier                 96.67%        100.00%         100.00
	ARDRegression                          95.83%         98.31%          98.31
	BayesianRidge                          96.67%         98.31%          98.31
	Lars                                   96.67%         98.31%          98.31
	CCA                                    96.67%         98.31%          98.31
	KNeighborsClassifier                   95.00%         98.31%          98.31
	SVC                                    97.42%         98.31%          98.31
	KernelRidge                            90.76%         93.22%          93.22
	AdaBoostClassifier                     88.26%         91.53%          91.53
	DummyClassifier                        39.47%         40.68%          40.68
	ElasticNet                             27.58%         25.42%          25.42
	Lasso                                  27.58%         25.42%          25.42
	
	Table 1: Wine
	
	
	Method                             10-fold CV           Test         Rating
	LinearDiscriminantAnalysis             98.00%         98.00%         100.00
	KNeighborsClassifier                   95.00%         98.00%         100.00
	SVC                                    96.00%         98.00%         100.00
	RandomForestClassifier                 95.00%         98.00%         100.00
	AdaBoostClassifier                     93.00%         92.00%          93.88
	LogisticRegression                     93.00%         90.00%          91.84
	MLPClassifier                          92.00%         90.00%          91.84
	BayesianRidge                          77.00%         82.00%          83.67
	Lars                                   76.00%         82.00%          83.67
	ARDRegression                          78.00%         78.00%          79.59
	CCA                                    74.00%         78.00%          79.59
	KernelRidge                            63.00%         70.00%          71.43
	ElasticNet                             34.00%         32.00%          32.65
	Lasso                                  34.00%         32.00%          32.65
	DummyClassifier                        35.00%         30.00%          30.61
	
	Table 2: Iris
	
	
	Method                             10-fold CV           Test         Rating
	MLPClassifier                          96.05%         97.87%         100.00
	SVC                                    96.84%         97.87%         100.00
	ARDRegression                          96.58%         96.81%          98.91
	KNeighborsClassifier                   96.85%         96.81%          98.91
	LogisticRegression                     96.58%         95.74%          97.83
	RandomForestClassifier                 94.49%         95.74%          97.83
	BayesianRidge                          96.05%         95.21%          97.28
	AdaBoostClassifier                     97.64%         95.21%          97.28
	LinearDiscriminantAnalysis             95.53%         94.15%          96.20
	CCA                                    87.91%         90.43%          92.39
	KernelRidge                            88.70%         84.57%          86.41
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
	CCA                                    88.12%         89.39%          90.00
	BayesianRidge                          84.38%         87.71%          88.31
	ARDRegression                          83.88%         87.54%          88.14
	KernelRidge                            84.21%         87.54%          88.14
	Lars                                   77.55%         86.53%          87.12
	AdaBoostClassifier                     34.25%         34.18%          34.41
	ElasticNet                              9.31%         11.45%          11.53
	Lasso                                   9.31%         11.45%          11.53
	DummyClassifier                        10.47%          9.26%           9.32
	
	Table 4: Digits
	
	
	Method                             10-fold CV           Test         Rating
	RandomForestClassifier                 89.85%         86.36%         100.00
	MLPClassifier                          88.21%         85.76%          99.30
	LogisticRegression                     87.76%         85.45%          98.95
	AdaBoostClassifier                     86.27%         85.45%          98.95
	ARDRegression                          88.06%         85.15%          98.60
	BayesianRidge                          87.31%         85.15%          98.60
	KernelRidge                            87.16%         85.15%          98.60
	Lars                                   87.31%         85.15%          98.60
	CCA                                    87.31%         85.15%          98.60
	LinearDiscriminantAnalysis             87.31%         85.15%          98.60
	SVC                                    86.87%         84.85%          98.25
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
	Lars                                   50.00%         50.00%          50.00
	LinearDiscriminantAnalysis             50.90%         50.00%          50.00
	CCA                                    48.51%         48.18%          48.18
	DummyClassifier                        51.34%         47.27%          47.27
	ARDRegression                          51.34%         47.27%          47.27
	BayesianRidge                          51.34%         47.27%          47.27
	ElasticNet                             51.34%         47.27%          47.27
	Lasso                                  51.34%         47.27%          47.27
	KernelRidge                            50.15%         46.67%          46.67
	
	Table 6: Circles
	
	
	Method                             10-fold CV           Test         Rating
	KNeighborsClassifier                  100.00%        100.00%         100.00
	SVC                                   100.00%        100.00%         100.00
	RandomForestClassifier                 99.70%         99.70%          99.70
	AdaBoostClassifier                     99.85%         98.18%          98.18
	MLPClassifier                          87.31%         90.61%          90.61
	LogisticRegression                     87.91%         89.70%          89.70
	ARDRegression                          87.46%         89.39%          89.39
	BayesianRidge                          87.46%         89.39%          89.39
	Lars                                   87.46%         89.39%          89.39
	CCA                                    87.31%         89.39%          89.39
	LinearDiscriminantAnalysis             87.31%         89.39%          89.39
	KernelRidge                            76.57%         76.06%          76.06
	DummyClassifier                        51.34%         47.27%          47.27
	ElasticNet                             51.34%         47.27%          47.27
	Lasso                                  51.34%         47.27%          47.27
	
	Table 7: Moons
	
	
	Method                             10-fold CV           Test         Rating
	ARDRegression                         100.00%        100.00%         100.00
	BayesianRidge                         100.00%        100.00%         100.00
	KernelRidge                           100.00%        100.00%         100.00
	Lars                                  100.00%        100.00%         100.00
	CCA                                   100.00%        100.00%         100.00
	LinearDiscriminantAnalysis            100.00%        100.00%         100.00
	LogisticRegression                    100.00%        100.00%         100.00
	KNeighborsClassifier                  100.00%        100.00%         100.00
	MLPClassifier                         100.00%        100.00%         100.00
	SVC                                   100.00%        100.00%         100.00
	AdaBoostClassifier                    100.00%        100.00%         100.00
	RandomForestClassifier                100.00%        100.00%         100.00
	ElasticNet                             32.69%         34.55%          34.55
	Lasso                                  32.69%         34.55%          34.55
	DummyClassifier                        34.63%         30.91%          30.91
	
	Table 8: Blobs
	
	
	Method                                 Rating
	SVC                                     99.46
	RandomForestClassifier                  99.19
	KNeighborsClassifier                    98.42
	MLPClassifier                           95.07
	LinearDiscriminantAnalysis              91.22
	LogisticRegression                      90.85
	AdaBoostClassifier                      89.13
	BayesianRidge                           87.85
	ARDRegression                           87.53
	CCA                                     87.06
	KernelRidge                             82.57
	Lars                                    82.27
	DummyClassifier                         41.13
	ElasticNet                              40.21
	Lasso                                   40.21
	
	Table 9: Final classifier rating over all datasets

# Remarks

The idea kind of works. For some datasets the performance can be competitive
against proper classifiers, but usually we get pretty average results.

The advantage is training speed - most regressor have no hyper parameters to tune,
so there is no need for grid search optimization.

I did not tune the hyper parameters for any of the classifiers - if I did
the final rankings would likely be different.
