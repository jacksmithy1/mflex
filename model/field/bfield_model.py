import numpy as np
from model.field.utility.seehafer import mirror_magnetogram
from model.field.utility.fft import fft_coeff_seehafer
from model.field.utility.poloidal import phi, phi_low, dphidz, dphidz_low


def magnetic_field(
    data_bz: np.ndarray[np.float64, np.dtype[np.float64]],
    z0: np.float64,
    deltaz: np.float64,
    a: float,
    b: float,
    alpha: float,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    zmin: np.float64,
    zmax: np.float64,
    nresol_x: int,
    nresol_y: int,
    nresol_z: int,
    pixelsize_x: np.float64,
    pixelsize_y: np.float64,
    nf_max: int,
    ffunc: str = "Neukirch",
) -> np.ndarray[np.float64, np.dtype[np.float64]]:
    length_scale: np.float64 = np.float64(2.0)  # Normalising length scale for Seehafer
    length_scale_x: np.float64 = 2.0 * nresol_x * pixelsize_x
    # Length scale in x direction for Seehafer
    length_scale_y: np.float64 = 2.0 * nresol_y * pixelsize_y
    # Length scale in y direction for Seehafer
    length_scale_x_norm: np.float64 = length_scale_x / length_scale
    # Normalised length scale in x direction for Seehafer
    length_scale_y_norm: np.float64 = length_scale_y / length_scale
    # Normalised length scale in y direction for Seehafer

    if xmin != 0.0 or ymin != 0.0 or zmin != 0.0:
        raise ValueError("Magnotgram not centred at origin")
    if not (xmax > 0.0 or ymax > 0.0 or zmax > 0.0):
        raise ValueError("Magnetrogram in wrong quadrant of Seehafer mirroring")

    x_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        np.arange(2.0 * nresol_x) * 2.0 * xmax / (2.0 * nresol_x - 1) - xmax
    )
    y_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        np.arange(2.0 * nresol_y) * 2.0 * ymax / (2.0 * nresol_y - 1) - ymax
    )
    z_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin
    )

    ratiodzls: np.float64 = deltaz / length_scale  # Normalised deltaz

    # kx, ky arrays, coefficients for x and y in Fourier series

    kx_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        np.arange(nf_max) * np.pi / length_scale_x_norm
    )  # [0:nf_max]
    ky_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        np.arange(nf_max) * np.pi / length_scale_y_norm
    )  # [0:nf_max]

    one_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        0.0 * np.arange(nf_max) + 1.0
    )

    ky_grid: np.ndarray[np.float64, np.dtype[np.float64]] = np.outer(
        ky_arr, one_arr
    )  # [0:nf_max, 0:nf_max]
    kx_grid: np.ndarray[np.float64, np.dtype[np.float64]] = np.outer(
        one_arr, kx_arr
    )  # [0:nf_max, 0:nf_max]

    # kx^2 + ky^2

    k2_arr: np.ndarray[np.float64, np.dtype[np.float64]] = np.outer(
        ky_arr**2, one_arr
    ) + np.outer(one_arr, kx_arr**2)
    k2_arr[0, 0] = (np.pi / length_scale) ** 2

    p_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        0.5 * ratiodzls * np.sqrt(k2_arr * (1.0 - a - a * b) - alpha**2)
    )
    q_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        0.5 * ratiodzls * np.sqrt(k2_arr * (1.0 - a + a * b) - alpha**2)
    )

    data_bz_seehafer: np.ndarray[np.float64, np.dtype[np.float64]] = mirror_magnetogram(
        data_bz, xmin, xmax, ymin, ymax, nresol_x, nresol_y
    )
    anm: np.ndarray[np.float64, np.dtype[np.float64]] = fft_coeff_seehafer(
        data_bz_seehafer, k2_arr, 2 * nresol_x, 2 * nresol_y, nf_max
    )

    phi_arr: np.ndarray[np.float64, np.dtype[np.float64]] = np.zeros(
        (nf_max, nf_max, nresol_z)
    )
    # [0:nf_max,0:nf_max, 0:nresol_z]
    dphidz_arr: np.ndarray[np.float64, np.dtype[np.float64]] = np.zeros(
        (nf_max, nf_max, nresol_z)
    )  # [0:nf_max,0:nf_max, 0:nresol_z]

    for iy in range(0, int(nf_max)):
        for ix in range(0, int(nf_max)):
            q: np.float64 = q_arr[iy, ix]
            p: np.float64 = p_arr[iy, ix]
            for iz in range(0, int(nresol_z)):
                z: np.float64 = z_arr[iz]
                if ffunc == "Low":
                    phi_arr[iy, ix, iz] = phi_low(z, p, q, z0, deltaz)
                    dphidz_arr[iy, ix, iz] = dphidz_low(z, p, q, z0, deltaz)
                if ffunc == "Neukirch":
                    phi_arr[iy, ix, iz] = phi(z, p, q, z0, deltaz)
                    dphidz_arr[iy, ix, iz] = dphidz(z, p, q, z0, deltaz)

    b_arr: np.ndarray[np.float64, np.dtype[np.float64]] = np.zeros(
        (2 * nresol_y, 2 * nresol_x, nresol_z, 3)
    )

    sin_x: np.ndarray[np.float64, np.dtype[np.float64]] = np.sin(
        np.outer(kx_arr, x_arr)
    )
    sin_y: np.ndarray[np.float64, np.dtype[np.float64]] = np.sin(
        np.outer(ky_arr, y_arr)
    )
    cos_x: np.ndarray[np.float64, np.dtype[np.float64]] = np.cos(
        np.outer(kx_arr, x_arr)
    )
    cos_y: np.ndarray[np.float64, np.dtype[np.float64]] = np.cos(
        np.outer(ky_arr, y_arr)
    )

    for iz in range(0, nresol_z):
        coeffs: np.ndarray[np.float64, np.dtype[np.float64]] = np.multiply(
            np.multiply(k2_arr, phi_arr[:, :, iz]), anm
        )
        # Componentwise multiplication, [0:nf_max, 0:nf_max]
        b_arr[:, :, iz, 2] = np.matmul(sin_y.T, np.matmul(coeffs, sin_x))
        # [0:2*nresol_y, 0:nf_max]*([0:nf_max, 0:nf_max]*[0:nf_max, 0:2*nresol_x]) = [0:2*nresol_y, 0:2*nresol_x]

        coeffs1: np.ndarray[np.float64, np.dtype[np.float64]] = np.multiply(
            np.multiply(anm, dphidz_arr[:, :, iz]), ky_grid
        )
        # Componentwise multiplication, [0:nf_max, 0:nf_max]
        coeffs2: np.ndarray[np.float64, np.dtype[np.float64]] = alpha * np.multiply(
            np.multiply(anm, phi_arr[:, :, iz]), kx_grid
        )
        # Componentwise multiplication, [0:nf_max, 0:nf_max]
        b_arr[:, :, iz, 0] = np.matmul(cos_y.T, np.matmul(coeffs1, sin_x)) - np.matmul(
            sin_y.T, np.matmul(coeffs2, cos_x)
        )
        # [0:2*nresol_y, 0:nf_max]*([0:nf_max, 0:nf_max]*[0:nf_max, 0:2*nresol_x]) = [0:2*nresol_y, 0:2*nresol_x]

        # Fieldline3D program was written for order B = [Bx,By,Bz] with indexing [ix,iy,iz] but here we have indexing [iy,ix,iz]
        # so in order to be consistent we have to switch to order B = [Bx,By,Bz] such that fieldline3D program treats X and Y as Y and X consistently

        coeffs3: np.ndarray[np.float64, np.dtype[np.float64]] = np.multiply(
            np.multiply(anm, dphidz_arr[:, :, iz]), kx_grid
        )
        # Componentwise multiplication, [0:nf_max, 0:nf_max]
        coeffs4: np.ndarray[np.float64, np.dtype[np.float64]] = alpha * np.multiply(
            np.multiply(anm, phi_arr[:, :, iz]), ky_grid
        )
        # Componentwise multiplication, [0:nf_max, 0:nf_max]
        b_arr[:, :, iz, 1] = np.matmul(sin_y.T, np.matmul(coeffs3, cos_x)) + np.matmul(
            cos_y.T, np.matmul(coeffs4, sin_x)
        )
        # [0:2*nresol_y, 0:nf_max]*([0:nf_max, 0:nf_max]*[0:nf_max, 0:2*nresol_x]) = [0:2*nresol_y, 0:2*nresol_x]
    """

    coeffs = (k2_arr[:, :, np.newaxis] * phi_arr) * anm[:, :, np.newaxis]
    # m_mat = np.einsum("nmk,mj->jnk", coeffs, sin_x)
    # b_arr[:, :, :, 2] = np.einsum("jnk,ni->ijk", m_mat, sin_y)
    b_arr[:, :, :, 2] = np.einsum(
        "jnk,ni->ijk", np.einsum("nmk,mj->jnk", coeffs, sin_x), sin_y
    )

    coeffs1 = (anm[:, :, np.newaxis] * dphidz_arr) * ky_grid[:, :, np.newaxis]
    coeffs2 = alpha * (anm[:, :, np.newaxis] * phi_arr) * kx_grid[:, :, np.newaxis]
    # m_mat1 = np.einsum("nmk,mj->jnk", coeffs1, sin_x)
    # m_mat2 = np.einsum("nmk,mj->jnk", coeffs2, cos_x)
    # b_arr[:, :, :, 0] = np.einsum("jnk,ni->ijk", m_mat1, cos_y) - np.einsum(
    #    "jnk,ni->ijk", m_mat2, sin_y
    # )
    b_arr[:, :, :, 0] = np.einsum(
        "jnk,ni->ijk", np.einsum("nmk,mj->jnk", coeffs1, sin_x), cos_y
    ) - np.einsum("jnk,ni->ijk", np.einsum("nmk,mj->jnk", coeffs2, cos_x), sin_y)

    coeffs3 = (anm[:, :, np.newaxis] * dphidz_arr) * kx_grid[:, :, np.newaxis]
    coeffs4 = alpha * (anm[:, :, np.newaxis] * phi_arr) * ky_grid[:, :, np.newaxis]
    # m_mat3 = np.einsum("nmk,mj->jnk", coeffs3, cos_x)
    # m_mat4 = np.einsum("nmk,mj->jnk", coeffs4, sin_x)
    # b_arr[:, :, :, 1] = np.einsum("jnk,ni->ijk", m_mat3, sin_y) + np.einsum(
    #    "jnk,ni->ijk", m_mat4, cos_y
    # )
    b_arr[:, :, :, 1] = np.einsum(
        "jnk,ni->ijk", np.einsum("nmk,mj->jnk", coeffs3, cos_x), sin_y
    ) + np.einsum("jnk,ni->ijk", np.einsum("nmk,mj->jnk", coeffs4, sin_x), cos_y)
    """
    return b_arr


def bz_partial_derivatives(
    data_bz: np.ndarray[np.float64, np.dtype[np.float64]],
    z0: np.float64,
    deltaz: np.float64,
    a: float,
    b: float,
    alpha: float,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    zmin: np.float64,
    zmax: np.float64,
    nresol_x: int,
    nresol_y: int,
    nresol_z: int,
    pixelsize_x: np.float64,
    pixelsize_y: np.float64,
    nf_max: int,
    ffunc: str = "Neukirch",
) -> np.ndarray[np.float64, np.dtype[np.float64]]:
    length_scale: float = 2.0  # Normalising length scale for Seehafer
    length_scale_x: float = 2.0 * nresol_x * float(pixelsize_x)
    # Length scale in x direction for Seehafer
    length_scale_y: float = 2.0 * nresol_y * float(pixelsize_y)
    # Length scale in y direction for Seehafer
    length_scale_x_norm: float = length_scale_x / length_scale
    # Normalised length scale in x direction for Seehafer
    length_scale_y_norm: float = length_scale_y / length_scale
    # Normalised length scale in y direction for Seehafer

    if xmin != 0.0 or ymin != 0.0 or zmin != 0.0:
        raise ValueError("Magnotgram not centred at origin")
    if not (xmax > 0.0 or ymax > 0.0 or zmax > 0.0):
        raise ValueError("Magnetrogram in wrong quadrant of Seehafer mirroring")

    x_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        np.arange(2.0 * nresol_x) * 2.0 * xmax / (2.0 * nresol_x - 1) - xmax
    )
    y_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        np.arange(2.0 * nresol_y) * 2.0 * ymax / (2.0 * nresol_y - 1) - ymax
    )
    z_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin
    )

    ratiodzls: np.float64 = deltaz / length_scale  # Normalised deltaz

    # kx, ky arrays, coefficients for x and y in Fourier series

    kx_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        np.arange(nf_max) * np.pi / length_scale_x_norm
    )  # [0:nf_max]
    ky_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        np.arange(nf_max) * np.pi / length_scale_y_norm
    )  # [0:nf_max]

    one_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        0.0 * np.arange(nf_max) + 1.0
    )

    ky_grid: np.ndarray[np.float64, np.dtype[np.float64]] = np.outer(
        ky_arr, one_arr
    )  # [0:nf_max, 0:nf_max]
    kx_grid: np.ndarray[np.float64, np.dtype[np.float64]] = np.outer(
        one_arr, kx_arr
    )  # [0:nf_max, 0:nf_max]

    # kx^2 + ky^2

    k2_arr: np.ndarray[np.float64, np.dtype[np.float64]] = np.outer(
        ky_arr**2, one_arr
    ) + np.outer(one_arr, kx_arr**2)
    k2_arr[0, 0] = (np.pi / length_scale) ** 2

    p_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        0.5 * ratiodzls * np.sqrt(k2_arr * (1.0 - a - a * b) - alpha**2)
    )
    q_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
        0.5 * ratiodzls * np.sqrt(k2_arr * (1.0 - a + a * b) - alpha**2)
    )

    data_bz_seehafer: np.ndarray[np.float64, np.dtype[np.float64]] = mirror_magnetogram(
        data_bz, xmin, xmax, ymin, ymax, nresol_x, nresol_y
    )
    anm: np.ndarray[np.float64, np.dtype[np.float64]] = fft_coeff_seehafer(
        data_bz_seehafer, k2_arr, 2 * nresol_x, 2 * nresol_y, nf_max
    )

    phi_arr: np.ndarray[np.float64, np.dtype[np.float64]] = np.zeros(
        (nf_max, nf_max, nresol_z)
    )  # [0:nf_max,0:nf_max, 0:nresol_z]
    dphidz_arr: np.ndarray[np.float64, np.dtype[np.float64]] = np.zeros(
        (nf_max, nf_max, nresol_z)
    )  # [0:nf_max,0:nf_max, 0:nresol_z]

    for iy in range(0, nf_max):
        for ix in range(0, nf_max):
            q: np.float64 = q_arr[iy, ix]
            p: np.float64 = p_arr[iy, ix]
            for iz in range(0, nresol_z):
                z: np.float64 = z_arr[iz]
                if ffunc == "Low":
                    phi_arr[iy, ix, iz] = phi_low(z, p, q, z0, deltaz)
                    dphidz_arr[iy, ix, iz] = dphidz_low(z, p, q, z0, deltaz)
                if ffunc == "Neukirch":
                    phi_arr[iy, ix, iz] = phi(z, p, q, z0, deltaz)
                    dphidz_arr[iy, ix, iz] = dphidz(z, p, q, z0, deltaz)

    bz_derivs: np.ndarray[np.float64, np.dtype[np.float64]] = np.zeros(
        (2 * nresol_y, 2 * nresol_x, nresol_z, 3)
    )

    sin_x: np.ndarray[np.float64, np.dtype[np.float64]] = np.sin(
        np.outer(kx_arr, x_arr)
    )
    sin_y: np.ndarray[np.float64, np.dtype[np.float64]] = np.sin(
        np.outer(ky_arr, y_arr)
    )
    cos_x: np.ndarray[np.float64, np.dtype[np.float64]] = np.cos(
        np.outer(kx_arr, x_arr)
    )
    cos_y: np.ndarray[np.float64, np.dtype[np.float64]] = np.cos(
        np.outer(ky_arr, y_arr)
    )

    for iz in range(0, nresol_z):
        coeffs: np.ndarray[np.float64, np.dtype[np.float64]] = np.multiply(
            np.multiply(k2_arr, dphidz_arr[:, :, iz]), anm
        )
        bz_derivs[:, :, iz, 2] = np.matmul(sin_y.T, np.matmul(coeffs, sin_x))

        coeffs2: np.ndarray[np.float64, np.dtype[np.float64]] = np.multiply(
            np.multiply(np.multiply(k2_arr, phi_arr[:, :, iz]), anm), kx_grid
        )
        bz_derivs[:, :, iz, 0] = np.matmul(sin_y.T, np.matmul(coeffs2, cos_x))

        coeffs3: np.ndarray[np.float64, np.dtype[np.float64]] = np.multiply(
            np.multiply(np.multiply(k2_arr, phi_arr[:, :, iz]), anm),
            ky_grid,
        )
        bz_derivs[:, :, iz, 1] = np.matmul(cos_y.T, np.matmul(coeffs3, sin_x))
    """
    coeffs = (k2_arr[:, :, np.newaxis] * dphidz_arr) * anm[:, :, np.newaxis]
    bz_derivs[:, :, :, 2] = np.einsum(
        "jnk,ni->ijk", np.einsum("nmk,mj->jnk", coeffs, sin_x), sin_y
    )

    coeffs1 = ((k2_arr[:, :, np.newaxis] * phi_arr) * anm[:, :, np.newaxis]) * kx_grid[
        :, :, np.newaxis
    ]
    bz_derivs[:, :, :, 2] = np.einsum(
        "jnk,ni->ijk", np.einsum("nmk,mj->jnk", coeffs1, cos_x), sin_y
    )

    coeffs2 = ((k2_arr[:, :, np.newaxis] * phi_arr) * anm[:, :, np.newaxis]) * ky_grid[
        :, :, np.newaxis
    ]
    bz_derivs[:, :, :, 1] = np.einsum(
        "jnk,ni->ijk", np.einsum("nmk,mj->jnk", coeffs2, cos_x), cos_y
    )
    """
    return bz_derivs
