import numpy as np
from typing import Tuple, Optional
import fbm


def brownian_motion(
        n_paths: int = 100,
        n_steps: int = 100,
        n_dims: int = 1,
        t: float = 1.0,
        t0: float = 0.0,
        s0: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
        Simulate n_paths n_dims dimensional Brownian motion over [t0, t] with n_steps starting from s0.
    """
    if t0 > t:
        raise ValueError(f"t0={t0} must be less than or equal to t={t}")
    if n_steps <= 0 or n_dims <= 0 or n_paths <= 0:
        raise ValueError(f"n_steps={n_steps} and n_paths={n_paths} and n_dims={n_dims} must be positive integers.")
    if s0 is None:
        s0 = np.zeros(n_dims)
    else:
        s0 = np.asarray(s0)
        if s0.shape != (n_dims,):
            raise ValueError(f"s0 must be of shape ({n_dims},), got {s0.shape}")
    if rng is None:
        rng = np.random.default_rng()

    time = np.linspace(t0, t, n_steps)
    h = (t-t0)/n_steps
    increments = rng.standard_normal((n_paths, n_steps-1, n_dims)) * np.sqrt(h)

    #append s0 to the increments of each path and then compute the cumulative sum to get the paths.
    s0_broadcast = np.broadcast_to(s0, (n_paths, 1, n_dims))
    space = np.cumsum(np.concatenate([s0_broadcast, increments], axis=1), axis=1)
    return space, time


def fbm_d_batch(
        hurst: float = 0.5,
        n_paths: int = 100,
        n_steps: int = 100,
        n_dims: int = 1,
        t: float = 1.0,
        method: str = "daviesharte",
        dependency_matrix:Optional[np.ndarray]=None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate multiple d dimensional samples of fractional Brownian motion."""
    if dependency_matrix is None:
        dependency_matrix = np.eye(n_dims)
    if dependency_matrix.shape[0] != n_dims:
        raise ValueError(f"Number of rows of the dependency matrix={dependency_matrix.shape[0]} must be equal to n_dims={n_dims}")

    fbm_instance = fbm.FBM(n=n_steps, hurst=hurst, length=t, method=method)

    paths =[]
    for _ in range(n_paths):
        uncorrelated_path = np.stack([fbm_instance.fbm() for _ in range(n_dims)]).T
        path = uncorrelated_path @ dependency_matrix
        paths.append(path)
    return np.stack(paths), fbm_instance.times()