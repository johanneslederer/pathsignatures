import numpy as np

#---- Process to price path ----

def fbm_to_geometric_fbm(
        fbm_path:np.ndarray,
        time:np.ndarray,
        mu:float = 0.05,
        sigma:float = 0.2
) -> np.ndarray:
    n_samples, n_steps, n_dims = fbm_path.shape
    time = np.broadcast_to(time.reshape(1, n_steps, 1), (n_samples, n_steps, 1))
    return np.exp((mu - 0.5*sigma**2)*time + sigma*fbm_path)

def add_linear_drift_to_fbm(
        fbm_path:np.ndarray,
        time:np.ndarray,
        mu:float = 0.05,
) -> np.ndarray:
    n_samples, n_steps, n_dims = fbm_path.shape
    time = np.broadcast_to(time.reshape(1, n_steps, 1), (n_samples, n_steps, 1))
    return 1 + mu * time + fbm_path


#---- Price path to value ----

def call_on_last_index_d_dim(
        path: np.ndarray,
        strike: float = 1
) -> np.ndarray:
    last = path[:, -1, :]  # (n_samples, d)  last time point
    basket_max = last.max(axis=1)  # (n_samples,)    max over dimensions
    return np.maximum(basket_max - strike, 0)  # (n_samples,)

def asian_option(
        path: np.ndarray,
        strike: float
) -> np.ndarray:
    if path.shape[2] != 1:
        raise ValueError(f"Asian option are only allowed for a 1-dim path. Got {path.shape[2]}-dim path instead.")
    return np.maximum(np.mean(path, axis=1) - strike, 0)

def rainbow_option_1(
        path: np.ndarray
) -> np.ndarray:
    if path.shape[2] != 2:
        raise ValueError(f"Rainbow option are only allowed for a 2-dim path. Got {path.shape[2]}-dim path instead.")
    x_0_t = path[:, -1, 0]
    x_1_t = path[:, -1, 1]
    return np.maximum(x_0_t - x_1_t, 0)

def rainbow_option_2(
        path: np.ndarray,
        strike: float
) -> np.ndarray:
    if path.shape[2] != 2:
        raise ValueError(f"Rainbow option are only allowed for a 2-dim path. Got {path.shape[2]}-dim path instead.")
    x_0_t = path[:, -1, 0]
    x_1_t = path[:, -1, 1]
    return np.maximum(np.maximum(x_0_t, x_1_t) - strike, 0)


#---- Adding noise ----

def add_randomness(
        y:np.ndarray,
        std:int=0.05,
        seed:int=0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return y + rng.normal(loc=0, scale=std, size=y.shape)