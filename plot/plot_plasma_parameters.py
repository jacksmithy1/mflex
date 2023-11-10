#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from model.plasma_parameters import deltaden, deltapres
import numpy as np
import matplotlib.pyplot as plt


def plot_deltaparam(
    bfield: np.ndarray[np.float64, np.dtype[np.float64]],
    dbzpartial: np.ndarray[np.float64, np.dtype[np.float64]],
    nresol_x: int,
    nresol_y: int,
    nresol_z: int,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    zmin: np.float64,
    zmax: np.float64,
    z0: np.float64,
    deltaz: np.float64,
    a: float,
    b: float,
    alpha: float,
    g: float,
) -> None:
    """
    Returns plot of variations in plasma parameters pressure and density
    at [y,x] where photospheric magnetic field strength is maximal.
    """

    b_back: np.ndarray[np.float64, np.dtype[np.float64]] = np.zeros(
        (nresol_y, nresol_x)
    )
    b_back: np.ndarray[np.float64, np.dtype[np.float64]] = bfield[
        nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x, 0, 2
    ]
    maxcoord = np.unravel_index(np.argmax(b_back, axis=None), b_back.shape)

    iy: int = int(maxcoord[0])
    ix: int = int(maxcoord[1])

    x_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        np.arange(nresol_x) * (xmax - xmin) / (nresol_x - 1) + xmin
    )
    y_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        np.arange(nresol_y) * (ymax - ymin) / (nresol_y - 1) + ymin
    )
    z_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin
    )

    # Background Pressure and Density variations

    delta_p: np.ndarray[np.float64, np.dtype[np.float64]] = np.zeros(nresol_z)
    delta_d: np.ndarray[np.float64, np.dtype[np.float64]] = np.zeros(nresol_z)

    for iz in range(0, nresol_z):
        z: np.float64 = z_arr[iz]
        bz: np.float64 = bfield[iy, ix, iz, 2]
        bzdotgradbz: np.float64 = (
            bfield[iy, ix, iz, 1] * dbzpartial[iy, ix, iz, 1]
            + bfield[iy, ix, iz, 0] * dbzpartial[iy, ix, iz, 0]
            + bfield[iy, ix, iz, 2] * dbzpartial[iy, ix, iz, 2]
        )
        delta_p[iz] = deltapres(z, z0, deltaz, a, b, bz)
        delta_d[iz] = deltaden(z, z0, deltaz, a, b, bz, bzdotgradbz, g)

    plt.plot(z_arr, delta_p, label="Background pressure", linewidth=0.5)
    plt.plot(z_arr, delta_d, label="Background density", linewidth=0.5)
    plt.legend()

    # current_time = datetime.now()
    # dt_string = current_time.strftime("%d-%m-%Y_%H-%M-%S")

    # plotname = (
    #    "/Users/lilli/Desktop/ISSI_code/tests/solo_L2_phi-hrt-blos_20220307T000609_V01_plasmaparam_"
    #    + str(a)
    #    + "_"
    #    + str(b)
    #    + "_"
    #    + str(alpha)
    #    + "_"
    #    + str(nf_max)
    #    + "_"
    #    + dt_string
    #    + ".png"
    # )
    # plt.savefig(plotname, dpi=300)

    plt.show()
