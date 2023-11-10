import numpy as np


def Vec_corr_metric(
    B: np.ndarray[np.float64, np.dtype[np.float64]],
    b: np.ndarray[np.float64, np.dtype[np.float64]],
):
    """
    B : B_ref
    b : B_rec
    """
    num: np.float64 = np.sum(np.multiply(B, b))
    div: np.float64 = np.sqrt(np.sum(np.multiply(B, B)) * np.sum(np.multiply(b, b)))

    return num / div


def Cau_Schw_metric(
    B: np.ndarray[np.float64, np.dtype[np.float64]],
    b: np.ndarray[np.float64, np.dtype[np.float64]],
):
    """
    B : B_ref
    b : B_rec
    """
    N: np.float64 = np.size(B)
    num: np.ndarray[np.float64, np.dtype[np.float64]] = np.multiply(B, b)
    div: np.ndarray[np.float64, np.dtype[np.float64]] = np.reciprocal(
        np.multiply(abs(B), abs(b))
    )
    temp: np.float64 = np.sum(np.multiply(num, div))

    return temp / N


def Norm_vec_err_metric(
    B: np.ndarray[np.float64, np.dtype[np.float64]],
    b: np.ndarray[np.float64, np.dtype[np.float64]],
):
    """
    B : B_ref
    b : B_rec
    """
    num: np.float64 = np.sum(abs(np.subtract(B, b)))
    div: np.float64 = np.sum(np.abs(B))

    return num / div


def Mean_vec_err_metric(
    B: np.ndarray[np.float64, np.dtype[np.float64]],
    b: np.ndarray[np.float64, np.dtype[np.float64]],
):
    """
    B : B_ref
    b : B_rec
    """
    N: np.float64 = np.size(B)
    num: np.ndarray[np.float64, np.dtype[np.float64]] = abs(np.subtract(B, b))
    div: np.ndarray[np.float64, np.dtype[np.float64]] = abs(np.reciprocal(B))
    temp: np.float64 = np.sum(np.multiply(num, div))

    return temp / N


def Mag_ener_metric(
    B: np.ndarray[np.float64, np.dtype[np.float64]],
    b: np.ndarray[np.float64, np.dtype[np.float64]],
):
    """
    B : B_ref
    b : B_rec
    """
    Bx: np.ndarray[np.float64, np.dtype[np.float64]] = B[:, :, :, 1][0, 0]
    By: np.ndarray[np.float64, np.dtype[np.float64]] = B[:, :, :, 0][0, 0]
    Bz: np.ndarray[np.float64, np.dtype[np.float64]] = B[:, :, :, 2][0, 0]
    bx: np.ndarray[np.float64, np.dtype[np.float64]] = b[:, :, :, 1][0, 0]
    by: np.ndarray[np.float64, np.dtype[np.float64]] = b[:, :, :, 0][0, 0]
    bz: np.ndarray[np.float64, np.dtype[np.float64]] = b[:, :, :, 2][0, 0]

    num: np.float64 = np.sqrt(np.dot(bx, bx) + np.dot(by, by) + np.dot(bz, bz))
    div: np.float64 = np.sqrt(np.dot(Bx, Bx) + np.dot(By, By) + np.dot(Bz, Bz))

    return num / div
