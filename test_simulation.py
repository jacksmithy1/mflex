#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mflex.simulate.simulate_boundary import dipole, non_periodic, dalmatian
import numpy as np
from mflex.model.field.bfield_model import magnetic_field, bz_partial_derivatives
from mflex.plot.plot_magnetogram import plot_fieldlines_grid, plot_magnetogram_boundary
from mflex.plot.plot_plasma_parameters import plot_deltaparam
import matplotlib.pyplot as plt
from mflex.model.plasma_parameters import (
    bpressure,
    bdensity,
    btemp,
    deltapres,
    deltaden,
    pres,
    den,
)
from datetime import datetime


nresol_x: int = 120
nresol_y: int = 120
nresol_z: int = 200
xmin: np.float64 = 0.0
xmax: np.float64 = 2.0
ymin: np.float64 = 0.0
ymax: np.float64 = 2.0
zmin: np.float64 = 0.0
zmax: np.float64 = 1.5
z0: np.float64 = 0.2
pixelsize_x: np.float64 = (xmax - xmin) / nresol_x
pixelsize_y: np.float64 = (ymax - ymin) / nresol_y
pixelsize_z: np.float64 = (zmax - zmin) / nresol_z
nf_max = 80

data_bz: np.ndarray[np.float64, np.dtype[np.float64]] = np.zeros((nresol_y, nresol_x))
x_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
    np.arange(nresol_x) * (xmax - xmin) / (nresol_x - 1) + xmin
)
y_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
    np.arange(nresol_y) * (ymax - ymin) / (nresol_y - 1) + ymin
)
z_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
    np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin
)

for ix in range(0, nresol_x):
    for iy in range(0, nresol_y):
        x = x_arr[ix]
        y = y_arr[iy]
        data_bz[iy, ix] = dipole(x, y)

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

b_back = np.zeros((2 * nresol_y, 2 * nresol_x))
b_back = bfield[:, :, 0, 2]
plot_magnetogram_boundary(b_back, 2 * nresol_x, 2 * nresol_y)

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

t_photosphere = 5600.0
t_corona = 2.0 * 10.0**6
t0 = (t_photosphere + t_corona * np.tanh(z0 / deltaz)) / (1.0 + np.tanh(z0 / deltaz))
t1 = (t_corona - t_photosphere) / (1.0 + np.tanh(z0 / deltaz))
g_solar = 274.0
mbar = 1.0
h = 1.3807 * t0 / (mbar * 1.6726 * g_solar) * 0.001
rho0 = 3.0**-4
b0 = 500.0
p0 = 1.3807 * t_photosphere * rho0 / (mbar * 1.6726) * 1.0
pB0 = 3.9789 - 3 * b0**2
beta0 = p0 / pB0

backpres = 0.0 * z_arr
backtemp = 0.0 * z_arr
backden = 0.0 * z_arr

maxcoord = np.unravel_index(np.argmax(b_back, axis=None), b_back.shape)
iy: int = int(maxcoord[0])
ix: int = int(maxcoord[1])
# print(ix, iy)
# print(x_arr[ix], y_arr[iy])
dpres = 0.0 * z_arr
dden = 0.0 * z_arr
fpres = 0.0 * z_arr
fden = 0.0 * z_arr

for iz in range(nresol_z):
    z = z_arr[iz]
    bz = bfield[iy, ix, iz, 2]
    bzdotgradbz = (
        dpartial_bfield[iy, ix, iz, 1] * bfield[iy, ix, iz, 1]
        + dpartial_bfield[iy, ix, iz, 0] * bfield[iy, ix, iz, 0]
        + dpartial_bfield[iy, ix, iz, 2] * bfield[iy, ix, iz, 2]
    )
    backpres[iz] = bpressure(z, z0, deltaz, h, t0, t1)
    backden[iz] = bdensity(z, z0, deltaz, h, t0, t1)
    backtemp[iz] = btemp(z, z0, deltaz, t0, t1)
    dpres[iz] = deltapres(z, z0, deltaz, a, b, bz)
    dden[iz] = deltaden(z, z0, deltaz, a, b, bz, bzdotgradbz, g_solar)
    fpres[iz] = pres(z, z0, deltaz, a, b, beta0, bz, h, t0, t1)
    fden[iz] = den(
        z, z0, deltaz, a, b, bz, bzdotgradbz, beta0, h, t0, t1, t_photosphere
    )

current_time = datetime.now()
dt_string = current_time.strftime("%d-%m-%Y_%H-%M-%S")

plt.plot(z_arr, backpres, label="Background pressure", linewidth=0.5, color="orange")
plt.axvline(
    x=z0, color="black", label="z0 = " + str(z0), linestyle="dashed", linewidth=0.5
)
plt.legend()
plotname = (
    "/Users/lilli/Desktop/mflex/tests/vonmises_backpres_"
    + str(a)
    + "_"
    + str(b)
    + "_"
    + str(alpha)
    + "_"
    + str(nf_max)
    + "_"
    + dt_string
    + ".png"
)
plt.savefig(plotname, dpi=300)
plt.show()
plt.plot(z_arr, backden, label="Background density", linewidth=0.5, color="magenta")
plt.legend()
plotname = (
    "/Users/lilli/Desktop/mflex/tests/vonmises_backden_"
    + str(a)
    + "_"
    + str(b)
    + "_"
    + str(alpha)
    + "_"
    + str(nf_max)
    + "_"
    + dt_string
    + ".png"
)
plt.savefig(plotname, dpi=300)
plt.show()
plt.plot(
    z_arr, backtemp, label="Background temperature", linewidth=0.5, color="deepskyblue"
)
plt.legend()
plotname = (
    "/Users/lilli/Desktop/mflex/tests/vonmises_backtemp_"
    + str(a)
    + "_"
    + str(b)
    + "_"
    + str(alpha)
    + "_"
    + str(nf_max)
    + "_"
    + dt_string
    + ".png"
)
plt.savefig(plotname, dpi=300)
plt.show()
plt.plot(
    z_arr,
    dpres,
    label="Delta pressure",
    linewidth=0.5,
    color="orange",
)
plt.plot(
    z_arr,
    dden,
    label="Delta density",
    linewidth=0.5,
    color="magenta",
)
plt.axvline(
    x=z0, color="black", label="z0 = " + str(z0), linestyle="dashed", linewidth=0.5
)
plt.legend()
plotname = (
    "/Users/lilli/Desktop/mflex/tests/vonmises_deltap_deltarho_"
    + str(a)
    + "_"
    + str(b)
    + "_"
    + str(alpha)
    + "_"
    + str(nf_max)
    + "_"
    + dt_string
    + ".png"
)
plt.savefig(plotname, dpi=300)
plt.show()
exit()

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
