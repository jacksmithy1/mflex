#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def Vec_corr_metric(
    B: np.ndarray[np.float64, np.dtype[np.float64]],
    b: np.ndarray[np.float64, np.dtype[np.float64]],
) -> np.float64:
    """
    Returns Vector Correlation metric of B : B_ref and b : B_rec.
    """

    return np.sum(np.multiply(B, b)) / (
        np.sqrt(np.sum(np.multiply(B, B)) * np.sum(np.multiply(b, b)))
    )


def Cau_Schw_metric(
    B: np.ndarray[np.float64, np.dtype[np.float64]],
    b: np.ndarray[np.float64, np.dtype[np.float64]],
) -> np.float64:
    """
    Returns Cauchy Schwarz metric of B : B_ref and b : B_rec.
    """

    N = np.size(B)
    num = np.multiply(B, b)
    div = np.reciprocal(np.multiply(abs(B), abs(b)))
    return np.sum(np.multiply(num, div)) / N


def Norm_vec_err_metric(
    B: np.ndarray[np.float64, np.dtype[np.float64]],
    b: np.ndarray[np.float64, np.dtype[np.float64]],
) -> np.float64:
    """
    Returns Normalised Vector Error metric of B : B_ref and b : B_rec.
    """

    return np.sum(abs(np.subtract(B, b))) / np.sum(np.abs(B))


def Mean_vec_err_metric(
    B: np.ndarray[np.float64, np.dtype[np.float64]],
    b: np.ndarray[np.float64, np.dtype[np.float64]],
) -> np.float64:
    """
    Returns Mean Vector Error metric of B : B_ref and b : B_rec.
    """

    N = np.size(B)
    num = abs(np.subtract(B, b))
    div = abs(np.reciprocal(B))

    return np.sum(np.multiply(num, div)) / N


def Mag_ener_metric(
    B: np.ndarray[np.float64, np.dtype[np.float64]],
    b: np.ndarray[np.float64, np.dtype[np.float64]],
) -> np.float64:
    """
    Returns Magnetic Energy metric of B : B_ref and b : B_rec.
    """

    Bx = B[:, :, :, 1][0, 0]
    By = B[:, :, :, 0][0, 0]
    Bz = B[:, :, :, 2][0, 0]
    bx = b[:, :, :, 1][0, 0]
    by = b[:, :, :, 0][0, 0]
    bz = b[:, :, :, 2][0, 0]

    num = np.sqrt(np.dot(bx, bx) + np.dot(by, by) + np.dot(bz, bz))
    div = np.sqrt(np.dot(Bx, Bx) + np.dot(By, By) + np.dot(Bz, Bz))

    return num / div
