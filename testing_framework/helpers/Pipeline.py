from functools import partial

import iisignature
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from testing_framework.helpers import ProcessGenerator, Functions


class Features(BaseEstimator, TransformerMixin):
    def __init__(self, depth=3): self.depth = int(depth)
    def fit(self, X, y=None): return self
    def transform(self, X):
        return iisignature.sig(X, self.depth)


def sample_and_get_X_y(
    X_generator:callable = partial(ProcessGenerator.fbm_d_batch),
    y_transforms:tuple = (
        partial(Functions.fbm_to_geometric_fbm),
        partial(Functions.call_on_last_index_d_dim),
        partial(Functions.add_randomness)
    )
):
    #sample from process
    fbm_paths, time = X_generator()

    #compute X and y
    X = combine_paths_and_time(fbm_paths, time)
    y = fbm_paths, time
    for func in y_transforms:
        y = [func(*y)]
    return X, *y

def get_pipe(
        depth:int = 4,
        alpha:float = 0.05
):
    return Pipeline([
        ("features",  Features(depth)),
        ("scale", StandardScaler()),
        ("estimator", Lasso(alpha)),
    ])



def combine_paths_and_time(
        paths:np.ndarray,
        time:np.ndarray
) -> np.ndarray:
    n_samples, n_steps, n_dims = paths.shape
    time = np.broadcast_to(time.reshape(1, n_steps, 1), (n_samples, n_steps, 1))
    return np.concatenate([paths, time], axis=-1)
