#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from load.read_file import read_issi_analytical
import numpy as np
from plot.plot_magnetogram import (
    plot_fieldlines_grid,
    plot_magnetogram_boundary,
    plot_magnetogram_boundary_3D,
)
from model.field.bfield_model import magnetic_field, bz_partial_derivatives
import scipy
from plot.plot_plasma_parameters import plot_deltaparam

# TO DO

# Plasma Parameters
# optimise choice of a and alpha, switch case for potential, LFF and MHS
# Add Bessel or Neukirch parameter to calling function, also to file names

"""
B0      : amplitude of magnetic field strength
alpha   : linear force free parameter (in units of 1/L)
a       : amplitude parameter of f(z)
b       : parameter determining asymptotic value for z --> infinity
z0      : z-value of "transition" in current density
deltaz  : width of the transition in current density
T0      : temperature at z=z0
T1      : determines coronal temperature (T0+T1)
H       : pressure scale height (measured in units of basic length) based on T0
p0      : plasma pressure at z=0
rho0    : plasma mass density at z=0
nf_max  : number of Fourier modes in x and y
kx      : wave numbers in the x-direction (2*pi*n/L_x)
ky      : wave numbers in the y-direction (2*pi*m/L_y)
k2      : array of kx^2 + ky^2 values
p       : determined by k, a, b and alpha - sqrt(k^2*(1-a-a*b) - alpha^2)/2
q       : determined by k, a, b and alpha - sqrt(k^2*(1-a+a*b) - alpha^2)/2
anm     : array of Fourier coefficients of the sin(kx*x)*sin(ky*y) terms
bnm     : array of Fourier coefficients of the sin(kx*x)*cos(ky*y) terms
cnm     : array of Fourier coefficients of the cos(kx*x)*sin(ky*y) terms
dnm     : array of Fourier coefficients of the cos(kx*x)*cos(ky*y) terms
"""
data = read_issi_analytical("data/Analytic_boundary_data.sav")

# BFieldvec_Seehafer = np.load('field_data_potential.npy')

data_bx: np.ndarray[np.float64, np.dtype[np.float64]] = data.data_x
data_by: np.ndarray[np.float64, np.dtype[np.float64]] = data.data_y
data_bz: np.ndarray[np.float64, np.dtype[np.float64]] = data.data_z
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

h1: float = 0.0001  # Initial step length for fieldline3D
eps: float = 1.0e-8
# Tolerance to which we require point on field line known for fieldline3D
hmin: float = 0.0  # Minimum step length for fieldline3D
hmax: float = 1.0  # Maximum step length for fieldline3D
g: float = 272.2  # solar gravitational acceleration m/s^-2 gravitational acceleration

deltaz: np.float64 = np.float64(
    z0 / 10.0
)  # z0 at 2Mm so widht of transition region = 200km

# plot_magnetogram_boundary(data_bz, nresol_x, nresol_y)

bfield = magnetic_field(
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

# current_time = datetime.datetime.now()

# dt_string = current_time.strftime("%d-%m-%Y_%H-%M-%S")

# path = (
#    "/Users/lilli/Desktop/ISSI_data/B_ISSI_RMHD_tanh_"
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
#    np.save(file, B_Seehafer)

# b_back_test = np.zeros((2 * nresol_y, 2 * nresol_x))
# b_back_test = B_Seehafer[:, :, 0, 2]
# plot_magnetogram_boundary_3D(
#    b_back_test, nresol_x, nresol_y, -xmax, xmax, -ymax, ymax, zmin, zmax
# )

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
