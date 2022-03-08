#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(csvfile, test_size=0.33):
    """ Data is assumed to be in 'Xy' format: all X# features are numerical,
    'y' is the last column (can be a string) and there is no index column. """
    #
    # Load data
    #
    df = pd.read_csv(csvfile, index_col=None)
    X = df.values[:, :-1]
    y = df.values[:, -1]
    columns = df.columns

    #
    # Transform label strings
    #
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = y.astype(np.int)

    #
    # Train - test split
    #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42)

    #
    # Scale
    #
    scaler = MinMaxScaler((np.finfo(float).eps, 1.0))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #
    # Remove constant values
    #
    selector = VarianceThreshold()
    X_train = selector.fit_transform(X_train)
    X_test = selector.transform(X_test)

    #
    # Extract features
    #
    pca = PCA(n_components="mle")
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, y_train, X_test, y_test, columns
