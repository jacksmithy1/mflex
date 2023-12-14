#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from mflex.model.field.utility.height_profile import f, dfdz, f_low, dfdz_low


def btemp(z, z0, deltaz, T0, T1):
    return T0 + T1 * np.tanh((z - z0) / deltaz)


def bpressure(z, z0, deltaz, h, T0, T1):
    q1 = deltaz / (2.0 * h * (1.0 + T1 / T0))
    q2 = deltaz / (2.0 * h * (1.0 - T1 / T0))
    q3 = deltaz * (T1 / T0) / (h * (1.0 - (T1 / T0) ** 2))

    p1 = (
        2.0
        * np.exp(-2.0 * (z - z0) / deltaz)
        / (1.0 + np.exp(-2.0 * (z - z0) / deltaz))
        / (1.0 + np.tanh(z0 / deltaz))
    )
    p2 = (1.0 - np.tanh(z0 / deltaz)) / (1.0 + np.tanh((z - z0) / deltaz))
    p3 = (1.0 + T1 / T0 * np.tanh((z - z0) / deltaz)) / (
        1.0 - T1 / T0 * np.tanh(z0 / deltaz)
    )

    return (p1**q1) * (p2**q2) * (p3**q3)


def bdensity(z, z0, deltaz, h, T0, T1):
    temp0 = T0 - T1 * np.tanh(z0 / deltaz)
    dummypres = bpressure(z, z0, deltaz, h, T0, T1)
    dummytemp = btemp(z, z0, deltaz, T0, T1)
    return dummypres / dummytemp * temp0


def deltapres(
    z: np.float64,
    z0: np.float64,
    deltaz: np.float64,
    a: float,
    b: float,
    bz: np.float64,
) -> np.float64:
    """
    Returns variation of pressure with height z at given x and y.
    """

    return -f(z, z0, deltaz, a, b) * bz**2.0 / 2.0  # (8.0**np.pi)


def deltapres_low(
    z: np.float64,
    kappa: np.float64,
    a: float,
    bz: np.float64,
) -> np.float64:
    """
    Returns variation of pressure with height z at given x and y.
    """
    return -f_low(z, a, kappa) * bz**2.0 / 2.0  # (8.0**np.pi)


def pres(z, z0, deltaz, z0_b, deltaz_b, a, b, beta0, bz, h, T0, T1):
    return 0.5 * beta0 * bpressure(z, z0, deltaz, h, T0, T1) + deltapres(
        z, z0_b, deltaz_b, a, b, bz
    )


def deltaden(
    z: np.float64,
    z0: np.float64,
    deltaz: np.float64,
    a: float,
    b: float,
    bz: np.float64,
    bzdotgradbz: np.float64,
) -> np.float64:
    """
    Returns variation of density with height z at given x and y.
    """

    return (
        dfdz(z, z0, deltaz, a, b) * bz**2.0 / 2.0
        + f(z, z0, deltaz, a, b) * bzdotgradbz
    )  # / (g * 4.0 * np.pi)


def deltaden_low(
    z: np.float64,
    kappa: np.float64,
    a: float,
    bz: np.float64,
    bzdotgradbz: np.float64,
) -> np.float64:
    """
    Returns variation of density with height z at given x and y.
    """
    return dfdz_low(z, a, kappa) * bz**2.0 / 2.0 + f_low(z, a, kappa) * bzdotgradbz


def den(
    z,
    z0,
    deltaz,
    z0_b,
    deltaz_b,
    a,
    b,
    bz,
    bzdotgradbz,
    beta0,
    h,
    T0,
    T1,
    T_photosphere,
):
    return 0.5 * beta0 / h * T0 / T_photosphere * bdensity(
        z, z0, deltaz, h, T0, T1
    ) + deltaden(z, z0_b, deltaz_b, a, b, bz, bzdotgradbz)


def btemp_linear(z, temps, heights):
    t1, t2, t3, t4 = temps[0], temps[1], temps[2], temps[3]
    h1, h2, h3, h4 = heights[0], heights[1], heights[2], heights[3]

    m1 = (t2 - t1) / (h2 - h1)
    m2 = (t3 - t2) / (h3 - h2)
    m3 = (t4 - t3) / (h4 - h3)

    if z >= h1 and z <= h2:
        t = t1 + m1 * (z - h1)
    elif z >= h2 and z <= h3:
        t = t2 + m2 * (z - h2)
    elif z >= h3 and z <= h4:
        t = t3 + m3 * (z - h3)
    else:
        print("z= " + str(z) + " not in range")
        raise ValueError

    return t


def temp(z, z0, deltaz, a, b, bz, bzdotgradbz, beta0, h, T0, T1, T_photosphere):
    p = pres(z, z0, deltaz, a, b, beta0, bz, h, T0, T1)
    d = den(z, z0, deltaz, a, b, bz, bzdotgradbz, beta0, h, T0, T1, T_photosphere)
    return p / d
