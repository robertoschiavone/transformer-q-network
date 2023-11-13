import numpy as np


def upsample(x: np.ndarray, new_len: int) -> np.ndarray:
    return np.interp(np.linspace(0, 1, new_len), np.linspace(0, 1, len(x)), x)


def normalize_minus1_1(x: np.ndarray) -> np.ndarray:
    return 2 * ((x - np.min(x)) / (np.max(x) - np.min(x))) - 1


def normalize_0_tau(x: np.ndarray, tau: float | int) -> np.ndarray:
    return (x - np.min(x)) / (tau - np.min(x))


def rolling_average(x: np.ndarray, n: int = 10) -> np.ndarray:
    result = np.cumsum(x)
    result[n:] = result[n:] - result[:-n]
    return result[n - 1:] / n


def downsample(x: np.ndarray) -> np.ndarray:
    return upsample(x, 11)
