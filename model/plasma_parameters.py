import numpy as np
from model.field.utility.height_profile import f, dfdz

# Need to read some papers

"""
def btemp_linear(z, temps, heights):
    t1, t2, t3 = temps[0], temps[1], temps[2]
    h1, h2, h3 = heights[0], heights[1], heights[2]

    m1 = (t2 - t1) / (h2 - h1)
    m2 = (t3 - t2) / (h3 - h2)

    if z >= h1 and z <= h2:
        t = t1 + m1 * (z - h1)
    elif z >= h2 and z <= h3:
        t = t2 + m2 * (z - h2)
    else:
        print("z= " + str(z) + " not in range")
        raise ValueError

    return t


def bpressure(z, z0, deltaz, h, T0, T1):
    q1 = deltaz / (2.0 * h * (1.0 + T1 / T0))
    q2 = deltaz / (2.0 * h * (1.0 - T1 / T0))
    q3 = deltaz * (T1 / T0) / (2.0 * h * (1.0 - (T1 / T0) ** 2))

    p1 = (
        2.0
        * np.exp(-2.0 * (z - z0) / deltaz)
        / (1.0 * np.exp(-2.0 * (z - z0) / deltaz))
        / (1.0 + np.tanh(z0 / deltaz))
    )
    p2 = (1.0 - np.tanh(z0 / deltaz)) / (1.0 + np.tanh((z - z0) / deltaz))
    p3 = (1.0 + T1 / T0 * np.tanh((z - z0) / deltaz)) / (
        1.0 - T1 / T0 * np.tanh(z0 / deltaz)
    )

    return p1**q1 * p2**q2 * p3**q3


def btemp(z, z0, deltaz, T0, T1):
    return T0 + T1 * np.tanh((z - z0) / deltaz)


def bdensity(z, z0, deltaz, h, T0, T1):
    temp0 = btemp(z, z0, deltaz, T0, T1)
    return bpressure(z, z0, deltaz, h, T1, T0) / btemp(z, z0, deltaz, T0, T1) * temp0


def pres(ix, iy, iz, z, z0, deltaz, a, b, beta0, bz, h, T0, T1):
    Bzsqr = bz[iy, ix, iz] ** 2.0
    return (
        0.5 * beta0 * bpressure(z, z0, deltaz, h, T0, T1)
        + Bzsqr * f(z, z0, deltaz, a, b) / 2.0
    )


def den(
    ix, iy, iz, z, z0, deltaz, a, b, bx, by, bz, dBz, beta0, h, T0, T1, T_photosphere
):
    Bx = bx[iy, ix, iz]
    By = by[iy, ix, iz]
    Bz = bz[iy, ix, iz]
    dBzdx = dBz[iy, ix, iz, 0]
    dBzdy = dBz[iy, ix, iz, 1]
    dBzdz = dBz[iy, ix, iz, 2]
    BdotgradBz = Bx * dBzdx + By * dBzdy + Bz * dBzdz
    return (
        0.5 * beta0 / h * T0 / T_photosphere * bdensity(z, z0, deltaz, T0, T1, h)
        + dfdz(z, z0, deltaz, a, b) * Bz**2.0 / 2.0
        + f(z, z0, deltaz, a, b) * BdotgradBz
    )


def temp(
    ix, iy, iz, z, z0, deltaz, a, b, bx, by, bz, dBz, beta0, h, T0, T1, T_photosphere
):
    p = pres(ix, iy, iz, z, z0, deltaz, a, b, beta0, bz, h, T0, T1)
    d = den(
        ix,
        iy,
        iz,
        z,
        z0,
        deltaz,
        a,
        b,
        bx,
        by,
        bz,
        dBz,
        beta0,
        h,
        T0,
        T1,
        T_photosphere,
    )
    return p / d
"""


def deltapres(
    z: np.float64,
    z0: np.float64,
    deltaz: np.float64,
    a: float,
    b: float,
    bz: np.float64,
) -> np.float64:
    return -f(z, z0, deltaz, a, b) * bz**2.0 / (8.0**np.pi)


def deltaden(
    z: np.float64,
    z0: np.float64,
    deltaz: np.float64,
    a: float,
    b: float,
    bz: np.float64,
    bzdotgradbz: np.float64,
    g: float,
) -> np.float64:
    return (
        dfdz(z, z0, deltaz, a, b) * bz**2.0 / 2.0
        + f(z, z0, deltaz, a, b) * bzdotgradbz
    ) / (g * 4.0 * np.pi)


# Something with derivative of Bz ugh
# B dot gradBz
# need gradBz
