#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def dipole(x: np.float64, y: np.float64) -> np.float64:
    """
    Returns value of Dipole-VonMises distribution at given x and y
    inspired by Neukirch and Wiegelmann (2019).
    """

    xx: np.float64 = np.pi * (x - 1.0)
    yy: np.float64 = np.pi * (y - 1.0)
    mu_x: float = -1.2
    mu_y: float = mu_x
    kappa_x: float = 10.0
    kappa_y: float = kappa_x

    return np.exp(kappa_x * np.cos(xx - mu_x)) / (
        2.0 * np.pi * np.i0(kappa_x)
    ) * np.exp(kappa_y * np.cos(yy - mu_y)) / (2.0 * np.pi * np.i0(kappa_y)) - np.exp(
        kappa_x * np.cos(xx + mu_x)
    ) / (
        2.0 * np.pi * np.i0(kappa_x)
    ) * np.exp(
        kappa_y * np.cos(yy + mu_y)
    ) / (
        2.0 * np.pi * np.i0(kappa_y)
    )


def non_periodic(x: np.float64, y: np.float64) -> np.float64:
    """
    Returns value of Non-periodic-VonMises distribution at given x and y
    inspired by Neukirch and Wiegelmann (2019).
    """

    xx: np.float64 = np.pi * (x - 1.0)
    yy: np.float64 = np.pi * (y - 1.0)
    mu_x: float = 1.0
    mu_y: float = -mu_x
    kappa_x: float = 20.0
    kappa_y: float = kappa_x
    mu_x1: float = mu_x
    mu_y1: float = mu_y
    mu_x2: float = -mu_x
    mu_y2: float = -mu_y
    mu_x3: float = mu_x2
    mu_y3: float = mu_y1

    return (
        np.exp(kappa_x * np.cos(xx - mu_x1))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y1))
        / (2.0 * np.pi * np.i0(kappa_y))
        - np.exp(kappa_x * np.cos(xx - mu_x2))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y2))
        / (2.0 * np.pi * np.i0(kappa_y))
        - np.exp(kappa_x * np.cos(xx - mu_x3))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y3))
        / (2.0 * np.pi * np.i0(kappa_y))
    )


def dalmatian(x: np.float64, y: np.float64) -> np.float64:
    """
    Returns value of Multipole-VonMises distribution at given x and y
    inspired by Neukirch and Wiegelmann (2019).
    """

    xx: np.float64 = np.pi * (x - 1.0)
    yy: np.float64 = np.pi * (y - 1.0)
    mu_x: float = 1.0
    mu_y: float = -mu_x
    kappa_x: float = 20.0
    kappa_y: float = kappa_x
    mu_x1: float = mu_x
    mu_y1: float = mu_y
    mu_x2: float = -1.2
    mu_y2: float = -1.2
    mu_x3: float = -2.4
    mu_y3: float = 1.9
    mu_x4: float = 2.1
    mu_y4: float = -1.6
    mu_x5: float = -1.5
    mu_y5: float = 1.2
    mu_x6: float = 2.5
    mu_y6: float = 0.0
    mu_x7: float = 0.0
    mu_y7: float = -2.0
    mu_x8: float = -1.0
    mu_y8: float = -2.4

    return (
        +np.exp(kappa_x * np.cos(xx - mu_x1))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y1))
        / (2.0 * np.pi * np.i0(kappa_y))
        - np.exp(kappa_x * np.cos(xx - mu_x2))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y2))
        / (2.0 * np.pi * np.i0(kappa_y))
        + np.exp(kappa_x * np.cos(xx - mu_x3))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y3))
        / (2.0 * np.pi * np.i0(kappa_y))
        + np.exp(kappa_x * np.cos(xx - mu_x4))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y4))
        / (2.0 * np.pi * np.i0(kappa_y))
        - np.exp(kappa_x * np.cos(xx - mu_x5))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y5))
        / (2.0 * np.pi * np.i0(kappa_y))
        - np.exp(kappa_x * np.cos(xx - mu_x6))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y6))
        / (2.0 * np.pi * np.i0(kappa_y))
        - np.exp(kappa_x * np.cos(xx - mu_x7))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y7))
        / (2.0 * np.pi * np.i0(kappa_y))
        + np.exp(kappa_x * np.cos(xx - mu_x8))
        / (2.0 * np.pi * np.i0(kappa_x))
        * np.exp(kappa_y * np.cos(yy - mu_y8))
        / (2.0 * np.pi * np.i0(kappa_y))
    )
