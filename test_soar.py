#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mflex.load.read_file import read_fits_soar, read_issi_analytical, read_issi_rmhd
from mflex.model.field.bfield_model import magnetic_field, bz_partial_derivatives
import numpy as np
import matplotlib.pyplot as plt
from mflex.model.plasma_parameters import deltapres, deltaden
from mflex.plot.plot_magnetogram import plot_fieldlines_grid
from mflex.plot.plot_plasma_parameters import plot_deltaparam
from datetime import datetime
import cProfile
import pstats

# code_to_profile = """

path_blos: str = (
    "/Users/lilli/Desktop/SOAR/obs/solo_L2_phi-hrt-blos_20220307T000609_V01.fits"
)
data = read_fits_soar(path_blos)
# data = read_issi_rmhd("data/RMHD_boundary_data.sav")

data_bz: np.ndarray[np.float64, np.dtype[np.float64]] = data.data_z
# data_by: np.ndarray[np.float64, np.dtype[np.float64]] = data.data_y
# data_bx: np.ndarray[np.float64, np.dtype[np.float64]] = data.data_x
nresol_x: int = data.nresol_x
nresol_y: int = data.nresol_y
nresol_z: int = data.nresol_z
pixelsize_x: np.float64 = data.pixelsize_x
pixelsize_y: np.float64 = data.pixelsize_y
pixelsize_z: np.float64 = data.pixelsize_z
nf_max: int = data.nf_max
xmin: np.float64 = data.xmin
xmax: np.float64 = data.xmax
ymin: np.float64 = data.ymin
ymax: np.float64 = data.ymax
zmin: np.float64 = data.zmin
zmax: np.float64 = data.zmax
z0: np.float64 = data.z0

a: float = 0.24
alpha: float = 0.5
b: float = 1.0

deltaz: np.float64 = np.float64(z0 / 10.0)
# Background atmosphere

g: float = 272.2  # solar gravitational acceleration m/s^-2 gravitational acceleration

h1: float = 0.0001  # Initial step length for fieldline3D
eps: float = 1.0e-8
# Tolerance to which we require point on field line known for fieldline3D
hmin: float = 0.0  # Minimum step length for fieldline3D
hmax: float = 1.0  # Maximum step length for fieldline3D

bfield: np.ndarray[np.float64, np.dtype[np.float64]] = magnetic_field(
    data_bz,
    z0,
    deltaz,
    a,
    b,
    alpha,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
    nresol_x,
    nresol_y,
    nresol_z,
    pixelsize_x,
    pixelsize_y,
    nf_max,
)

# current_time = datetime.now()
# dt_string = current_time.strftime("%d-%m-%Y_%H-%M-%S")

# path = (
#    "/Users/lilli/Desktop/ISSI_code/tests/solo_L2_phi-hrt-blos_20220307T000609_V01_"
#    + str(a)
#    + "_"
#    + str(b)
#    + "_"
#    + str(alpha)
#    + "_"
#    + str(nf_max)
#    + "_"
#    + dt_string
#    + ".npy"
# )

# with open(path, "wb") as file:
#    np.save(file, bfield)

plot_fieldlines_grid(
    bfield,
    h1,
    hmin,
    hmax,
    eps,
    nresol_x,
    nresol_y,
    nresol_z,
    -xmax,
    xmax,
    -ymax,
    ymax,
    zmin,
    zmax,
    a,
    b,
    alpha,
    nf_max,
)

dpartial_bfield: np.ndarray[np.float64, np.dtype[np.float64]] = bz_partial_derivatives(
    data_bz,
    z0,
    deltaz,
    a,
    b,
    alpha,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
    nresol_x,
    nresol_y,
    nresol_z,
    pixelsize_x,
    pixelsize_y,
    nf_max,
)

plot_deltaparam(
    bfield,
    dpartial_bfield,
    nresol_x,
    nresol_y,
    nresol_z,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
    z0,
    deltaz,
    a,
    b,
    alpha,
    g,
)

# profile_file = "profile_results.txt"
# profiler = cProfile.Profile()
# profiler.runctx(code_to_profile, globals(), locals())
# profiler.dump_stats(profile_file)

# with open(profile_file, "w") as f:
#    stats = pstats.Stats(profiler, stream=f)
#    stats.sort_stats("cumulative")
#    stats.print_stats()
# print(f"Profiling results saved to {profile_file}")
