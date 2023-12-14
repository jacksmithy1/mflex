#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def fft_coeff_seehafer(
    data_bz: np.ndarray[np.float64, np.dtype[np.float64]],
    k2_arr: np.ndarray[np.float64, np.dtype[np.float64]],
    nresol_x: int,
    nresol_y: int,
    nf_max: int,
) -> np.ndarray[np.float64, np.dtype[np.float64]]:
    """
    Given the Seehafer-mirrored photospheric magnetic field data_bz,
    returns coefficients anm for series expansion of 3D magnetic field.
    """

    if data_bz.shape[0] != nresol_y or data_bz.shape[1] != nresol_x:
        print("Shape of magnetogram does not match nresol_y x nresol_x]")
        raise ValueError

    anm = 0.0 * k2_arr
    signal = np.fft.fftshift(np.fft.fft2(data_bz) / nresol_x / nresol_y)

    """
    signal = abs(np.fft.fftshift(np.fft.fft2(data_bz) / nresol_x / nresol_y).real)

    if nresol_x % 2 == 0:
        centre_x = int(nresol_x / 2)
    else:
        centre_x = int((nresol_x + 1) / 2)
    if nresol_y % 2 == 0:
        centre_y = int(nresol_y / 2)
    else:
        centre_y = int((nresol_y + 1) / 2)

    for ix in range(1, nf_max):
        for iy in range(1, nf_max):
            anm[iy, ix] = (
                -signal[centre_y + iy, centre_x + ix]
                + signal[centre_y + iy, centre_x - ix]
                + signal[centre_y - iy, centre_x + ix]
                - signal[centre_y - iy, centre_x - ix]
            ) / k2_arr[iy, ix]
    """

    for ix in range(0, nresol_x, 2):
        for iy in range(1, nresol_y, 2):
            temp = signal[iy, ix]
            signal[iy, ix] = -temp

    for ix in range(1, nresol_x, 2):
        for iy in range(0, nresol_y, 2):
            temp = signal[iy, ix]
            signal[iy, ix] = -temp

    if nresol_x % 2 == 0:
        centre_x = int(nresol_x / 2)
    else:
        centre_x = int((nresol_x + 1) / 2)
    if nresol_y % 2 == 0:
        centre_y = int(nresol_y / 2)
    else:
        centre_y = int((nresol_y + 1) / 2)

    for ix in range(1, nf_max):
        for iy in range(1, nf_max):
            anm[iy, ix] = (
                -signal[centre_y + iy, centre_x + ix]
                + signal[centre_y + iy, centre_x - ix]
                + signal[centre_y - iy, centre_x + ix]
                - signal[centre_y - iy, centre_x - ix]
            ).real / k2_arr[iy, ix]

    return anm
