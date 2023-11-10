#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from simulate.simulate_boundary import dipole, non_periodic, dalmatian
import numpy as np
from model.field.bfield_model import magnetic_field, bz_partial_derivatives
from plot.plot_magnetogram import plot_fieldlines_grid, plot_magnetogram_boundary
from plot.plot_plasma_parameters import plot_deltaparam

nresol_x: int = 150
nresol_y: int = 150
nresol_z: int = 150
xmin: np.float64 = 0.0
xmax: np.float64 = 1.0
ymin: np.float64 = 0.0
ymax: np.float64 = 1.0
zmin: np.float64 = 0.0
zmax: np.float64 = 1.5
z0: np.float64 = 0.2
pixelsize_x: np.float64 = (xmax - xmin) / nresol_x
pixelsize_y: np.float64 = (ymax - ymin) / nresol_y
pixelsize_z: np.float64 = (zmax - zmin) / nresol_z
nf_max = 100

g: float = 272.2  # solar gravitational acceleration m/s^-2 gravitational acceleration

data_bz: np.ndarray[np.float64, np.dtype[np.float64]] = np.zeros((nresol_y, nresol_x))
x_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
    np.arange(nresol_x) * (nresol_x) / (nresol_x - 1)
)
y_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
    np.arange(nresol_y) * (nresol_y) / (nresol_y - 1)
)

for ix in range(0, nresol_x):
    for iy in range(0, nresol_y):
        x = x_arr[ix]
        y = y_arr[iy]
        data_bz[iy, ix] = dalmatian(x, y)

# plot_magnetogram_boundary(data_bz, nresol_y, nresol_x)

a: float = 0.24
alpha: float = 0.5
b: float = 1.0

h1: float = 0.0001  # Initial step length for fieldline3D
eps: float = 1.0e-8
# Tolerance to which we require point on field line known for fieldline3D
hmin: float = 0.0  # Minimum step length for fieldline3D
hmax: float = 1.0  # Maximum step length for fieldline3D

deltaz: np.float64 = np.float64(z0 / 10.0)

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

b_back_test = np.zeros((2 * nresol_y, 2 * nresol_x))
b_back_test = bfield[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x, 0, 2]
plot_magnetogram_boundary(b_back_test, nresol_x, nresol_y)

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
