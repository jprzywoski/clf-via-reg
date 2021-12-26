""" The 3-Clause BSD License

Copyright 2021 Jakub Przywóski

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES;  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import BayesianRidge
import numpy as np


class RegressionClassifier:
    """Classification via regression.

    This classifier converts the target values into binary range and then
    treats the problem as a regression task. Multi-class problems are handled
    by OneVsRestClassifier.

    Parameters
    ----------
    base_estimator : None, default=None
        This is the regression estimator used for classification. If not
        specified ``BayesianRidge`` will be used.

    Attributes
    ----------
    _base_estimator : base_estimator
        This is the regression estimator used for classification.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from RegressionClassifier import RegressionClassifier
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> clf = RegressionClassifier().fit(X, y)
    >>> clf.score(X, y)
    0.961335676625659
    """

    def __init__(self, base_estimator=None):
        if base_estimator is None:
            base_estimator = BayesianRidge()
        self.base_estimator = base_estimator
        self._base_estimator = OneVsRestClassifier(
            _RegressionClassifier(self.base_estimator))

    def fit(self, X, y):
        self._base_estimator.fit(X, y)
        return self

    def predict(self, X):
        return self._base_estimator.predict(X)

    def decision_function(self, X):
        return self.predict(X)

    def predict_proba(self, X):
        return self._base_estimator.predict_proba(X)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep=True):
        return {"base_estimator": self.base_estimator}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class _RegressionClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, reg):
        self.reg = reg

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return np.clip(np.rint(self.reg.predict(X)), 0, 1)

    def decision_function(self, X):
        return self.predict(X)

    def predict_proba(self, X):
        return np.clip(self.reg.predict(X), 0, 1)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep=True):
        return {"reg": self.reg}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
