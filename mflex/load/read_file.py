#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple, Dict
import scipy
import numpy as np
import math
from astropy.io.fits import open, getdata
from mflex.plot.plot_magnetogram import plot_magnetogram_boundary
from mflex.classes.clsmod import Data3D, DataBz

# TO DO
# Need to split def prep_ISSI_data from get_magnetogram or sth like that
# Need to decide on general format we want data to be given, probably as
# Bx = np.array(ny, nx) and By = np.array(ny, nx) at z = 0 (Photosphere)
# Additionally need pixelsize in km


def read_issi_rmhd(path: str) -> Data3D:
    """
    Returns dataclass Data3D extracted from RMHD_boundary_data.sav
    provided by ISSI Team.
    """

    data = scipy.io.readsav(path, python_dict=True, verbose=True)

    data_bz = data["b2dz"]
    # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_bx = data["b2dx"]
    # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_by = data["b2dy"]
    # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns

    print(data["info_unit"])
    print(data["info_pixel"])
    print(data["info_array"])

    bz_xlen = data_bz.shape[1]
    bz_ylen = data_bz.shape[0]

    # if bx_xlen != by_xlen or bx_xlen != bz_xlen or by_xlen != bz_xlen:
    #    print("x lengths of data do not match")
    #    raise ValueError
    # if bx_ylen != by_ylen or bx_ylen != bz_ylen or by_ylen != bz_ylen:
    #    print("y lengths of data do not match")
    #    raise ValueError
    # else:
    #    nresol_x = bx_xlen  # Data resolution in x direction
    #    nresol_y = bx_ylen  # Data resolution in y direction
    nresol_x = bz_xlen
    nresol_y = bz_ylen
    L = 5.0

    xmin = 0.0  # Minimum value of x in data length scale, not in Mm
    ymin = 0.0  # Minimum value of y in data length scale, not in Mm
    zmin = 0.0  # Minimum value of z in data length scale, not in Mm
    xmax = None  # Minimum value of x in data length scale, not in Mm
    ymax = None  # Minimum value of y in data length scale, not in Mm
    zmax = None

    pixelsize_x_km = 192.0
    pixelsize_y_km = 192.0
    pixelsize_z_km = 64.0
    xmax_km = nresol_x * pixelsize_x_km
    ymax_km = nresol_y * pixelsize_y_km

    zmax_km = 10000.0
    z0_km = 2000.0

    if nresol_x < nresol_y:
        xmax = L  # Maximum value of x in data length scale, not in Mm
        ymax = nresol_y / nresol_x * L
        # Maximum value of y in data length scale, not in Mm
        zmax = zmax_km / xmax_km * L
        z0 = z0_km / xmax_km * L
        pixelsize_z = pixelsize_z_km / xmax_km * L
    if nresol_y < nresol_x:
        ymax = L
        xmax = nresol_x / nresol_y * L
        zmax = zmax_km / ymax_km * L
        z0 = z0_km / ymax_km * L
        pixelsize_z = pixelsize_z_km / ymax_km * L
    if nresol_y == nresol_x:
        xmax = L
        ymax = L
        zmax = zmax_km / ymax_km * L
        z0 = z0_km / ymax_km * L
        pixelsize_z = pixelsize_z_km / ymax_km * L

    pixelsize_x = np.abs(xmax - xmin) / nresol_x  # Data pixel size in x direction
    pixelsize_y = np.abs(ymax - ymin) / nresol_y  # Data pixel size in y direction

    if pixelsize_x != pixelsize_y:
        raise ValueError(("directional pixel sizes of data do not match"))

    nresol_z = int(np.floor(zmax / pixelsize_z))
    nf_max = min(nresol_x, nresol_y)

    return Data3D(
        data_bx,
        data_by,
        data_bz,
        nresol_x,
        nresol_y,
        nresol_z,
        pixelsize_x,
        pixelsize_y,
        pixelsize_z,
        nf_max,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        z0,
    )


def read_issi_rmhd_zbased(path: str) -> Data3D:
    """
    Returns dataclass Data3D extracted from RMHD_boundary_data.sav
    provided by ISSI Team.
    """

    data = scipy.io.readsav(path, python_dict=True, verbose=True)

    data_bz = data["b2dz"]
    # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_bx = data["b2dx"]
    # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_by = data["b2dy"]
    # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns

    print(data["info_unit"])
    print(data["info_pixel"])
    print(data["info_array"])

    bz_xlen = data_bz.shape[1]
    bz_ylen = data_bz.shape[0]

    # if bx_xlen != by_xlen or bx_xlen != bz_xlen or by_xlen != bz_xlen:
    #    print("x lengths of data do not match")
    #    raise ValueError
    # if bx_ylen != by_ylen or bx_ylen != bz_ylen or by_ylen != bz_ylen:
    #    print("y lengths of data do not match")
    #    raise ValueError
    # else:
    #    nresol_x = bx_xlen  # Data resolution in x direction
    #    nresol_y = bx_ylen  # Data resolution in y direction
    nresol_x = bz_xlen
    nresol_y = bz_ylen
    L = 1.0

    xmin = 0.0  # Minimum value of x in data length scale, not in Mm
    ymin = 0.0  # Minimum value of y in data length scale, not in Mm
    zmin = 0.0  # Minimum value of z in data length scale, not in Mm
    xmax = None  # Minimum value of x in data length scale, not in Mm
    ymax = None  # Minimum value of y in data length scale, not in Mm
    zmax = None

    pixelsize_x_km = 192.0
    pixelsize_y_km = 192.0
    pixelsize_z_km = 64.0
    xmax_km = nresol_x * pixelsize_x_km
    ymax_km = nresol_y * pixelsize_y_km

    zmax_km = 10000.0
    z0_km = 2000.0

    zmax = L
    nresol_z = int(np.floor(zmax_km / pixelsize_z_km))

    xmax = xmax_km / zmax_km
    ymax = ymax_km / zmax_km

    pixelsize_z = np.abs(zmax - zmin) / nresol_z

    z0 = z0_km / zmax_km

    pixelsize_x = pixelsize_x_km / zmax_km  # Data pixel size in x direction
    pixelsize_y = pixelsize_y_km / zmax_km  # Data pixel size in y direction

    if pixelsize_x != pixelsize_y:
        raise ValueError(("directional pixel sizes of data do not match"))

    nf_max = min(nresol_x, nresol_y)

    print("xmax, ymax, zmax", xmax, ymax, zmax)
    print("xmax_km, ymax_km, zmax_km", xmax_km, ymax_km, zmax_km)
    print("nresol_x, nresol_y, nresol_z", nresol_x, nresol_y, nresol_z)
    print("pixelsize_x, pixelsize_x_km", pixelsize_x, pixelsize_x_km)
    print("pixelsize_y, pixelsize_y_km", pixelsize_y, pixelsize_y_km)
    print("pixelsize_z, pixelsize_z_km", pixelsize_z, pixelsize_z_km)

    return Data3D(
        data_bx,
        data_by,
        data_bz,
        nresol_x,
        nresol_y,
        nresol_z,
        pixelsize_x,
        pixelsize_y,
        pixelsize_z,
        nf_max,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        z0,
    )


def read_issi_analytical(path: str) -> Data3D:
    """
    Returns dataclass Data3D extracted from Analytical_boundary_data.sav
    provided by ISSI Team.
    """

    data = scipy.io.readsav(path, python_dict=True, verbose=True)
    data_bz = data["b2dz5"]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_bx = data["b2dx5"]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_by = data["b2dy5"]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    # scale_height = data['h3d']

    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns

    print(data["info_unit"])
    print(data["info_pixel"])
    # print(data['info_boundary'])
    # print(data["info_array"]

    # bx_xlen: np.int16 = data_bx.shape[1]
    # bx_ylen: np.int16 = data_bx.shape[0]
    # by_xlen: np.int16 = data_by.shape[1]
    # by_ylen: np.int16 = data_bx.shape[0]
    bz_xlen = data_bz.shape[1]
    bz_ylen = data_bz.shape[0]

    # if bx_xlen != by_xlen or bx_xlen != bz_xlen or by_xlen != bz_xlen:
    #    print("x lengths of data do not match")
    #    raise ValueError
    # if bx_ylen != by_ylen or bx_ylen != bz_ylen or by_ylen != bz_ylen:
    #    print("y lengths of data do not match")
    #    raise ValueError
    # else:
    #    nresol_x = bx_xlen  # Data resolution in x direction
    #    nresol_y = bx_ylen  # Data resolution in y direction
    nresol_x = bz_xlen
    nresol_y = bz_ylen
    nresol_z = 50
    L = 1000.0

    pixelsize_z_km = 40.0
    pixelsize_x_km = 40.0
    pixelsize_y_km = 40.0
    xmax_km = nresol_x * pixelsize_x_km
    ymax_km = nresol_y * pixelsize_y_km

    xmin = 0.0  # Minimum value of x in data length scale, not in Mm
    ymin = 0.0  # Minimum value of y in data length scale, not in Mm
    zmin = 0.0  # Minimum value of z in data length scale, not in Mm
    xmax = None
    ymax = None
    zmax = None

    zmax_km = 2000.0
    z0_km = 2000.0

    xmax = xmax_km / L
    ymax = ymax_km / L
    zmax = zmax_km / L
    z0 = z0_km / L

    pixelsize_x = np.abs(xmax - xmin) / nresol_x  # Data pixel size in x direction
    pixelsize_y = np.abs(ymax - ymin) / nresol_y  # Data pixel size in y direction
    pixelsize_z = np.abs(zmax - zmin) / nresol_z

    nf_max = min(nresol_x, nresol_y)

    print("xmax, ymax, zmax", xmax, ymax, zmax)
    print("xmax_km, ymax_km, zmax_km", xmax_km, ymax_km, zmax_km)
    print("nresol_x, nresol_y, nresol_z", nresol_x, nresol_y, nresol_z)
    print("pixelsize_x, pixelsize_x_km", pixelsize_x, pixelsize_x_km)
    print("pixelsize_y, pixelsize_y_km", pixelsize_y, pixelsize_y_km)
    print("pixelsize_z, pixelsize_z_km", pixelsize_z, pixelsize_z_km)

    return Data3D(
        data_bx,
        data_by,
        data_bz,
        nresol_x,
        nresol_y,
        nresol_z,
        pixelsize_x,
        pixelsize_y,
        pixelsize_z,
        nf_max,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        z0,
    )


def read_issi_analytical_alt(path: str) -> Data3D:
    """
    Returns dataclass Data3D extracted from Analytical_boundary_data.sav
    provided by ISSI Team.
    """
    data = scipy.io.readsav(path, python_dict=True, verbose=True)

    data_bz = data["b2dz5"]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_bx = data["b2dx5"]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_by = data["b2dy5"]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    # scale_height = data['h3d']

    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns

    print(data["info_unit"])
    print(data["info_pixel"])
    # print(data['info_boundary'])
    # print(data["info_array"]

    nresol_x = data_bz.shape[1]
    nresol_y = data_bz.shape[0]

    pixelsize_z_km = 40.0
    pixelsize_x_km = 40.0
    pixelsize_y_km = 40.0

    xmin = 0.0  # Minimum value of x in data length scale, not in Mm
    ymin = 0.0  # Minimum value of y in data length scale, not in Mm
    zmin = 0.0  # Minimum value of z in data length scale, not in Mm

    xmax_Mm = nresol_x * pixelsize_x_km / 1000.0
    ymax_Mm = nresol_y * pixelsize_y_km / 1000.0
    zmax_Mm = 2.0

    nresol_z = int(np.floor(zmax_Mm * 1000.0 / pixelsize_z_km))

    L = 1.6

    xmax = xmax_Mm / L
    ymax = ymax_Mm / L
    zmax = zmax_Mm / L

    z0 = zmax
    pixelsize_x = pixelsize_x_km / 1000.0 / L
    pixelsize_y = pixelsize_y_km / 1000.0 / L
    pixelsize_z = pixelsize_z_km / 1000.0 / L

    nf_max = min(nresol_x, nresol_y)

    print("xmax, ymax, zmax", xmax, ymax, zmax)
    print("xmax_Mm, ymax_Mm, zmax_Mm", xmax_Mm, ymax_Mm, zmax_Mm)
    print("nresol_x, nresol_y, nresol_z", nresol_x, nresol_y, nresol_z)
    print("pixelsize_x, pixelsize_x_km", pixelsize_x, pixelsize_x_km)
    print("pixelsize_y, pixelsize_y_km", pixelsize_y, pixelsize_y_km)
    print("pixelsize_z, pixelsize_z_km", pixelsize_z, pixelsize_z_km)

    return Data3D(
        data_bx,
        data_by,
        data_bz,
        nresol_x,
        nresol_y,
        nresol_z,
        pixelsize_x,
        pixelsize_y,
        pixelsize_z,
        nf_max,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        z0,
    )


def read_issi_analytical_zbased(path: str) -> Data3D:
    """
    Returns dataclass Data3D extracted from Analytical_boundary_data.sav
    provided by ISSI Team.
    """

    data = scipy.io.readsav(path, python_dict=True, verbose=True)
    data_bz = data["b2dz5"]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_bx = data["b2dx5"]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    data_by = data["b2dy5"]  # [0:nresol_y,0:nresol_x]
    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns
    # scale_height = data['h3d']

    # Y-axis size first as this corresponds to number of rows, then X-Axis size corresponding t number of columns

    print(data["info_unit"])
    print(data["info_pixel"])
    # print(data['info_boundary'])
    # print(data["info_array"]

    # bx_xlen: np.int16 = data_bx.shape[1]
    # bx_ylen: np.int16 = data_bx.shape[0]
    # by_xlen: np.int16 = data_by.shape[1]
    # by_ylen: np.int16 = data_bx.shape[0]
    bz_xlen = data_bz.shape[1]
    bz_ylen = data_bz.shape[0]

    # if bx_xlen != by_xlen or bx_xlen != bz_xlen or by_xlen != bz_xlen:
    #    print("x lengths of data do not match")
    #    raise ValueError
    # if bx_ylen != by_ylen or bx_ylen != bz_ylen or by_ylen != bz_ylen:
    #    print("y lengths of data do not match")
    #    raise ValueError
    # else:
    #    nresol_x = bx_xlen  # Data resolution in x direction
    #    nresol_y = bx_ylen  # Data resolution in y direction
    nresol_x = bz_xlen
    nresol_y = bz_ylen
    L = 1.0

    pixelsize_z_km = 40.0
    pixelsize_x_km = 40.0
    pixelsize_y_km = 40.0

    xmin = 0.0  # Minimum value of x in data length scale, not in Mm
    ymin = 0.0  # Minimum value of y in data length scale, not in Mm
    zmin = 0.0  # Minimum value of z in data length scale, not in Mm
    xmax = None
    ymax = None
    zmax = None

    xmax_km = nresol_x * pixelsize_x_km
    ymax_km = nresol_y * pixelsize_y_km

    zmax_km = 2000.0
    z0_km = 2000.0

    zmax = L
    nresol_z = int(np.floor(zmax_km / pixelsize_z_km))

    xmax = xmax_km / zmax_km
    ymax = ymax_km / zmax_km

    pixelsize_z = np.abs(zmax - zmin) / nresol_z

    z0 = z0_km / zmax_km

    pixelsize_x = pixelsize_x_km / zmax_km  # Data pixel size in x direction
    pixelsize_y = pixelsize_y_km / zmax_km  # Data pixel size in y direction

    nf_max = min(nresol_x, nresol_y)

    print("xmax, ymax, zmax", xmax, ymax, zmax)
    print("xmax_km, ymax_km, zmax_km", xmax_km, ymax_km, zmax_km)
    print("nresol_x, nresol_y, nresol_z", nresol_x, nresol_y, nresol_z)
    print("pixelsize_x, pixelsize_x_km", pixelsize_x, pixelsize_x_km)
    print("pixelsize_y, pixelsize_y_km", pixelsize_y, pixelsize_y_km)
    print("pixelsize_z, pixelsize_z_km", pixelsize_z, pixelsize_z_km)

    return Data3D(
        data_bx,
        data_by,
        data_bz,
        nresol_x,
        nresol_y,
        nresol_z,
        pixelsize_x,
        pixelsize_y,
        pixelsize_z,
        nf_max,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        z0,
    )


def read_fits_soar(path: str, header: bool = False) -> DataBz:
    """
    Returns dataclass DataBz extracted from _blos.fits file
    previously downloaded from Solar Orbiter Archive.
    """

    with open(path) as data:
        # data.info()
        image = getdata(path, ext=False)
        x_len = image.shape[0]
        y_len = image.shape[1]
        """plot_magnetogram_boundary(image, x_len, y_len)
        x_start = int(input("First pixel x axis: "))
        x_last = int(input("Last pixel x axis: "))
        y_start = int(input("First pixel y axis: "))
        y_last = int(input("Last pixel y axis: "))"""
        x_start = 400
        x_last = 1200
        y_start = 500
        y_last = 1000
        cut_image = image[y_start:y_last, x_start:x_last]
        # plot_magnetogram_boundary(cut_image, x_last - x_start, y_last - y_start)
        # if header == True:
        #    with open(
        #        "/Users/lilli/Desktop/SOAR/obs/solo_L2_phi-hrt-blos_20220307T000609_V01_HEADER.txt",
        #        "w",
        #    ) as f:
        #        for d in data:
        #            f.write(repr(d.header))
        #    print("File header has been printed to Desktop/SOAR/obs")
        hdr = data[0].header  # the primary HDU header
        dist = hdr["DSUN_OBS"]
        pixelsize_x_unit = hdr["CUNIT1"]
        pixelsize_y_unit = hdr["CUNIT2"]
        pixelsize_x_arcsec = hdr["CDELT1"]
        pixelsize_y_arcsec = hdr["CDELT2"]

        if not pixelsize_x_unit == pixelsize_y_unit:
            print("Pixelsize units not matchy-matchy")
            raise ValueError
        if not pixelsize_x_arcsec == pixelsize_y_arcsec:
            print("Data pixelsizes in x and y direction not matchy-matchy")
            raise ValueError
        else:
            pixelsize_radians = pixelsize_x_arcsec / 206265.0

    dist_km = dist / 1000.0
    pixelsize_km = np.floor(pixelsize_radians * dist_km)

    nresol_x = cut_image.shape[1]
    nresol_y = cut_image.shape[0]
    length_scale = 1.0  # L

    xmax_km = nresol_x * pixelsize_km
    ymax_km = nresol_y * pixelsize_km
    pixelsize_z_km = 90.0

    zmax_km = 10000.0
    z0_km = 2000.0

    xmin = 0.0  # Minimum value of x in data length scale, not in Mm
    ymin = 0.0  # Minimum value of y in data length scale, not in Mm
    zmin = 0.0  # Minimum value of z in data length scale, not in Mm

    if nresol_x < nresol_y:
        xmax = length_scale  # Maximum value of x in data length scale, not in Mm
        ymax = nresol_y / nresol_x  # Maximum value of y in data length scale, not in Mm
        zmax = zmax_km / xmax_km
        z0 = z0_km / xmax_km
        pixelsize_z = pixelsize_z_km / xmax_km
    if nresol_y < nresol_x:
        ymax = length_scale
        xmax = nresol_x / nresol_y
        zmax = zmax_km / ymax_km
        z0 = z0_km / ymax_km
        pixelsize_z = pixelsize_z_km / ymax_km
    if nresol_y == nresol_x:
        xmax = length_scale
        ymax = length_scale
        zmax = zmax_km / ymax_km
        z0 = z0_km / ymax_km
        pixelsize_z = pixelsize_z_km / ymax_km

    pixelsize_x = (
        abs(xmax - xmin) / nresol_x
    )  # Data pixel size in x direction in relation to xmin and xmax
    pixelsize_y = (
        abs(ymax - ymin) / nresol_y
    )  # Data pixel size in y direction in relation to ymin and ymax

    if pixelsize_x != pixelsize_y:
        print("Directional pixel sizes of data do not match")
        raise ValueError

    nresol_z = int(np.floor(zmax / pixelsize_z))

    nf_max = min(nresol_x, nresol_y)
    # Calulate parameters in Mm for checking purposes

    # xmax_Mm = nresol_x * pixelsize_km / 1000.0
    # ymax_Mm = nresol_y * pixelsize_km / 1000.0
    # zmax_Mm = nresol_z * pixelsize_km / 1000.0

    # print("Mm", xmax_Mm, ymax_Mm, zmax_Mm)

    # if xmax == length_scale:
    # ratio_Mm_xy = ymax_Mm / xmax_Mm
    # ratio_Mm_xz = zmax_Mm / xmax_Mm
    # if ymax == length_scale:
    # ratio_Mm_xy = xmax_Mm / ymax_Mm
    # ratio_Mm_xz = zmax_Mm / ymax_Mm

    # print("ratio Mm", ratio_Mm_xy, ratio_Mm_xz)
    # pixelsize_z = pixelsize_x * pixelsize_z_km / pixelsize_km
    # if pixelsize_z != pixelsize_x:
    #    print("nresol_z and zmax do not match")
    #    raise ValueError

    databz = DataBz(
        cut_image,
        nresol_x,
        nresol_y,
        nresol_z,
        pixelsize_x,
        pixelsize_y,
        pixelsize_z,
        nf_max,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        z0,
    )

    return databz


def read_fits_soar_zbased(path: str, header: bool = False) -> DataBz:
    """
    Returns dataclass DataBz extracted from _blos.fits file
    previously downloaded from Solar Orbiter Archive.
    """

    with open(path) as data:
        # data.info()
        image = getdata(path, ext=False)
        x_len = image.shape[0]
        y_len = image.shape[1]
        """plot_magnetogram_boundary(image, x_len, y_len)
        x_start = int(input("First pixel x axis: "))
        x_last = int(input("Last pixel x axis: "))
        y_start = int(input("First pixel y axis: "))
        y_last = int(input("Last pixel y axis: "))"""
        x_start = 400
        x_last = 1200
        y_start = 500
        y_last = 1000
        cut_image = image[y_start:y_last, x_start:x_last]
        # plot_magnetogram_boundary(cut_image, x_last - x_start, y_last - y_start)
        # if header == True:
        #    with open(
        #        "/Users/lilli/Desktop/SOAR/obs/solo_L2_phi-hrt-blos_20220307T000609_V01_HEADER.txt",
        #        "w",
        #    ) as f:
        #        for d in data:
        #            f.write(repr(d.header))
        #    print("File header has been printed to Desktop/SOAR/obs")
        hdr = data[0].header  # the primary HDU header
        dist = hdr["DSUN_OBS"]
        pixelsize_x_unit = hdr["CUNIT1"]
        pixelsize_y_unit = hdr["CUNIT2"]
        pixelsize_x_arcsec = hdr["CDELT1"]
        pixelsize_y_arcsec = hdr["CDELT2"]

        if not pixelsize_x_unit == pixelsize_y_unit:
            print("Pixelsize units not matchy-matchy")
            raise ValueError
        if not pixelsize_x_arcsec == pixelsize_y_arcsec:
            print("Data pixelsizes in x and y direction not matchy-matchy")
            raise ValueError
        else:
            pixelsize_radians = pixelsize_x_arcsec / 206265.0

    dist_km = dist / 1000.0
    pixelsize_km = np.floor(pixelsize_radians * dist_km)

    nresol_x = cut_image.shape[1]
    nresol_y = cut_image.shape[0]
    L = 1.0

    xmax_km = nresol_x * pixelsize_km
    ymax_km = nresol_y * pixelsize_km
    pixelsize_z_km = 90.0

    zmax_km = 10000.0
    z0_km = 2000.0

    xmin = 0.0  # Minimum value of x in data length scale, not in Mm
    ymin = 0.0  # Minimum value of y in data length scale, not in Mm
    zmin = 0.0  # Minimum value of z in data length scale, not in Mm

    xmax_km = nresol_x * pixelsize_km
    ymax_km = nresol_y * pixelsize_km

    zmax_km = 10000.0
    z0_km = 2000.0

    zmax = L
    nresol_z = int(np.floor(zmax_km / pixelsize_z_km))

    xmax = xmax_km / zmax_km
    ymax = ymax_km / zmax_km

    pixelsize_z = np.abs(zmax - zmin) / nresol_z

    z0 = z0_km / zmax_km

    pixelsize_x = pixelsize_km / zmax_km  # Data pixel size in x direction
    pixelsize_y = pixelsize_km / zmax_km  # Data pixel size in y direction

    if pixelsize_x != pixelsize_y:
        raise ValueError(("directional pixel sizes of data do not match"))

    nf_max = min(nresol_x, nresol_y)

    print("xmax, ymax, zmax", xmax, ymax, zmax)
    print("xmax_km, ymax_km, zmax_km", xmax_km, ymax_km, zmax_km)
    print("nresol_x, nresol_y, nresol_z", nresol_x, nresol_y, nresol_z)
    print("pixelsize_x, pixelsize_x_km", pixelsize_x, pixelsize_km)
    print("pixelsize_y, pixelsize_y_km", pixelsize_y, pixelsize_km)
    print("pixelsize_z, pixelsize_z_km", pixelsize_z, pixelsize_z_km)

    databz = DataBz(
        cut_image,
        nresol_x,
        nresol_y,
        nresol_z,
        pixelsize_x,
        pixelsize_y,
        pixelsize_z,
        nf_max,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        z0,
    )

    return databz
