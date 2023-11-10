import numpy as np


def fft_coeff_seehafer(
    data_bz: np.ndarray[np.float64, np.dtype[np.float64]],
    k2_arr: np.ndarray[np.float64, np.dtype[np.float64]],
    nresol_x: int,
    nresol_y: int,
    nf_max: int,
) -> np.ndarray[np.float64, np.dtype[np.float64]]:
    # b_back is mirrored magnetogram
    # shape(k2_arr) == shape(b_back) = [nresol_y, nresol_x]

    if data_bz.shape[0] != nresol_y or data_bz.shape[1] != nresol_x:
        print("Shape of magnetogram does not match nresol_y x nresol_x]")
        raise ValueError

    anm: np.ndarray[np.float64, np.dtype[np.float64]] = 0.0 * k2_arr

    signal: np.ndarray[np.float64, np.dtype[np.float64]] = np.fft.fftshift(
        np.fft.fft2(data_bz) / nresol_x / nresol_y
    )

    for ix in range(0, nresol_x, 2):
        for iy in range(1, nresol_y, 2):
            temp: np.float64 = signal[iy, ix]
            signal[iy, ix] = -temp

    for ix in range(1, nresol_x, 2):
        for iy in range(0, nresol_y, 2):
            temp: np.float64 = signal[iy, ix]
            signal[iy, ix] = -temp

    centre_x: int = int(nresol_x / 2)
    centre_y: int = int(nresol_y / 2)

    for ix in range(1, nf_max):
        for iy in range(1, nf_max):
            anm[iy, ix] = (
                -signal[centre_y + iy, centre_x + ix]
                + signal[centre_y + iy, centre_x - ix]
                + signal[centre_y - iy, centre_x + ix]
                - signal[centre_y - iy, centre_x - ix]
            ).real / k2_arr[iy, ix]

    return anm
