import numpy as np

from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.linear_model import (
    LinearRegression,
    SGDRegressor,
    ElasticNet,
    Ridge,
    Lasso,
)
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import Normalizer, StandardScaler

from sklearn.base import TransformerMixin

from typing import Union


class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray()


def train_pipe(
    regressor: Union[
        LinearRegression,
        Lasso,
        Ridge,
        ElasticNet,
        SGDRegressor,
        SVR,
        HistGradientBoostingRegressor,
    ],
    X: np.ndarray,
    y: np.ndarray,
    tfidf: bool = True,
    normalization: bool = False,
    scaling: bool = False,
    requires_dense: bool = False,
) -> Pipeline:
    """training sklearn pipeline

    Args:
        pipe (Pipeline): pipeline
        X (np.ndarray): input
        y (np.ndarray): output
        tfidf (bool, optional): use tfidf transformation. Defaults to True.
        normalization (bool, optional): use normalization. Defaults to False.
        scaling (bool, optional): use scaling. Defaults to False.
        requires_dense (bool, optional): only to be set True for regressors that do not work with sparse matrices. Defaults to False.

    Returns:
        Pipeline: trained pipeline
    """

    pipe = Pipeline([("vect", CountVectorizer())])

    if tfidf:
        pipe.steps.extend([("tfidf", TfidfTransformer())])
    if normalization:
        pipe.steps.extend([("normalizer", Normalizer(norm="l2"))])
    if scaling:
        pipe.steps.extend([("scaler", StandardScaler(with_mean=False))])
    if requires_dense:
        pipe.steps.extend([("to_dense", DenseTransformer())])
    pipe.steps.extend([("reg", regressor)])

    pipe.fit(X, y)

    return pipe


def evaluate_regressor(
    regressor: Union[
        LinearRegression,
        Lasso,
        Ridge,
        ElasticNet,
        SGDRegressor,
        SVR,
        HistGradientBoostingRegressor,
    ],
    X: np.ndarray,
    y: np.ndarray,
    tfidf: bool = True,
    normalization: bool = False,
    scaling: bool = False,
    requires_dense: bool = False,
) -> float:
    """Evaluation function that uses crossvalidation for regressor pipes used for optuna hyperparameter optimization

    Args:
        regressor (Union[ LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor, SVR, HistGradientBoostingRegressor, ]): regressor
        X (np.ndarray): input
        y (np.ndarray): array of labels
        tfidf (bool, optional): use tfidf transformation. Defaults to True.
        normalization (bool, optional): use normalization. Defaults to False.
        scaling (bool, optional): use scaling. Defaults to False.
        requires_dense (bool, optional): only to be set True for regressors that do not work with sparse matrices. Defaults to False.

    Returns:
        float: median of crossvalidation
    """

    pipe = Pipeline([("vect", CountVectorizer())])

    if tfidf:
        pipe.steps.extend([("tfidf", TfidfTransformer())])
    if normalization:
        pipe.steps.extend([("normalizer", Normalizer(norm="l2"))])
    if scaling:
        pipe.steps.extend([("scaler", StandardScaler(with_mean=False))])
    if requires_dense:
        pipe.steps.extend([("to_dense", DenseTransformer())])

    pipe.steps.extend([("reg", regressor)])

    cv = ShuffleSplit(n_splits=3, test_size=0.2)
    scores = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error")

    score = round(np.median(scores), 3)

    return score
