import numpy as np
from scipy.special import jv


def phi(
    z: np.float64, p: np.float64, q: np.float64, z0: np.float64, deltaz: np.float64
) -> np.float64:
    rplus: np.float64 = p / deltaz
    rminus: np.float64 = q / deltaz

    r: np.float64 = rminus / rplus
    d: np.float64 = np.cosh(2.0 * rplus * z0) + r * np.sinh(2.0 * rplus * z0)

    if z - z0 < 0.0:
        return (
            np.cosh(2.0 * rplus * (z0 - z)) + r * np.sinh(2.0 * rplus * (z0 - z))
        ) / d

    else:
        return np.exp(-2.0 * rminus * (z - z0)) / d


def dphidz(
    z: np.float64, p: np.float64, q: np.float64, z0: np.float64, deltaz: np.float64
) -> np.float64:
    rplus: np.float64 = p / deltaz
    rminus: np.float64 = q / deltaz

    r: np.float64 = rminus / rplus
    d: np.float64 = np.cosh(2.0 * rplus * z0) + r * np.sinh(2.0 * rplus * z0)

    if z - z0 < 0.0:
        return (
            -2.0
            * rplus
            * (np.sinh(2.0 * rplus * (z0 - z)) + r * np.cosh(2.0 * rplus * (z0 - z)))
            / d
        )

    else:
        return -2.0 * rminus * np.exp(-2.0 * rminus * (z - z0)) / d


def phi_low(
    z: np.float64, p: np.float64, q: np.float64, z0: np.float64, deltaz: np.float64
) -> np.float64:
    return jv(p, q * np.exp(-z / (2.0 * deltaz))) / jv(p, q)


def dphidz_low(
    z: np.float64, p: np.float64, q: np.float64, z0: np.float64, deltaz: np.float64
) -> np.float64:
    return (
        (
            q
            * np.exp(-z / (2.0 * deltaz))
            * jv(p + 1.0, q * np.exp(-z / (2.0 * deltaz)))
            - p * jv(p, q * np.exp(-z / (2.0 * deltaz)))
        )
        / (2.0 * deltaz)
        / jv(p, q)
    )
