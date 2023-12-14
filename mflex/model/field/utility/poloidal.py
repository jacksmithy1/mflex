#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import jv


def phi(
    z: np.float64, p: np.float64, q: np.float64, z0: np.float64, deltaz: np.float64
) -> np.float64:
    """
    Returns poloidal component of magnetic field vector according
    to asymptotic approximatio of Neukirch and Wiegelmann (2019).
    """

    rplus = p / deltaz
    rminus = q / deltaz

    r = rminus / rplus
    d = np.cosh(2.0 * rplus * z0) + r * np.sinh(2.0 * rplus * z0)

    if z - z0 < 0.0:
        return (
            np.cosh(2.0 * rplus * (z0 - z)) + r * np.sinh(2.0 * rplus * (z0 - z))
        ) / d

    else:
        return np.exp(-2.0 * rminus * (z - z0)) / d


def phi_vectorized(
    z: np.ndarray, p: np.ndarray, q: np.ndarray, z0: float, deltaz: float
) -> np.ndarray:
    """
    Vectorized version of the phi function that operates on NumPy arrays.
    """
    rplus = p / deltaz
    rminus = q / deltaz

    r = rminus / rplus
    d = np.cosh(2.0 * rplus * z0) + r * np.sinh(2.0 * rplus * z0)

    result = np.empty_like(z)
    mask = z < z0
    result[mask] = (
        np.cosh(2.0 * rplus * (z0 - z[mask]))
        + r[mask] * np.sinh(2.0 * rplus * (z0 - z[mask]))
    ) / d[mask]
    result[~mask] = np.exp(-2.0 * rminus[~mask] * (z[~mask] - z0)) / d[~mask]

    return result


def dphidz(
    z: np.float64, p: np.float64, q: np.float64, z0: np.float64, deltaz: np.float64
) -> np.float64:
    """
    Returns z derivatie of poloidal component of magnetic field vector according
    to asymptotic approximation of Neukirch and Wiegelmann (2019).
    """

    rplus = p / deltaz
    rminus = q / deltaz

    r = rminus / rplus
    d = np.cosh(2.0 * rplus * z0) + r * np.sinh(2.0 * rplus * z0)

    if z - z0 < 0.0:
        return (
            -2.0
            * rplus
            * (np.sinh(2.0 * rplus * (z0 - z)) + r * np.cosh(2.0 * rplus * (z0 - z)))
            / d
        )

    else:
        return -2.0 * rminus * np.exp(-2.0 * rminus * (z - z0)) / d


def dphidz_vectorized(
    z: np.ndarray, p: np.ndarray, q: np.ndarray, z0: float, deltaz: float
) -> np.ndarray:
    """
    Vectorized version of the dphidz function that operates on NumPy arrays.
    """
    rplus = p / deltaz
    rminus = q / deltaz

    r = rminus / rplus
    d = np.cosh(2.0 * rplus * z0) + r * np.sinh(2.0 * rplus * z0)

    result = np.empty_like(z)
    mask = z < z0
    result[mask] = (
        -2.0
        * rplus
        * (
            np.sinh(2.0 * rplus * (z0 - z[mask]))
            + r[mask] * np.cosh(2.0 * rplus * (z0 - z[mask]))
        )
    ) / d[mask]
    result[~mask] = (
        -2.0 * rminus[~mask] * np.exp(-2.0 * rminus[~mask] * (z[~mask] - z0))
    ) / d[~mask]

    return result


def phi_low(
    z: np.float64, p: np.float64, q: np.float64, kappa: np.float64
) -> np.float64:
    """
    Returns poloidal component of magnetic field vector using
    height profile by Low (1991, 1992).
    """

    return jv(p, q * np.exp(-z * kappa / 2.0)) / jv(p, q)


def dphidz_low(
    z: np.float64, p: np.float64, q: np.float64, kappa: np.float64
) -> np.float64:
    """
    Returns z derivative of poloidal component of magnetic field vector using
    height profile by Low (1991, 1992).
    """

    return (
        (
            q * np.exp(-z * kappa / 2.0) * jv(p + 1.0, q * np.exp(-z * kappa / 2.0))
            - p * jv(p, q * np.exp(-z * kappa / 2.0))
        )
        * kappa
        / 2.0
        / jv(p, q)
    )
