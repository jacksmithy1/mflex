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
b_back = np.zeros((2 * nresol_y, 2 * nresol_x))
b_back = bfield[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x, 0, 2]
# plot_magnetogram_boundary(b_back, nresol_x, nresol_y)

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
x_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
    np.arange(nresol_x) * (xmax - xmin) / (nresol_x - 1) + xmin
)
y_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
    np.arange(nresol_y) * (ymax - ymin) / (nresol_y - 1) + ymin
)
z_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
    np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin
)

backpres = 0.0 * z_arr
backtemp = 0.0 * z_arr
backden = 0.0 * z_arr

maxcoord = np.unravel_index(np.argmax(b_back, axis=None), b_back.shape)
iy: int = int(maxcoord[0])
ix: int = int(maxcoord[1])
print(ix, iy)
print(x_arr[ix], y_arr[iy])
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
    "/Users/lilli/Desktop/mflex/tests/soar_backpres_"
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
plt.plot(z_arr, backden, label="Background density", linewidth=0.5, color="magenta")
plt.legend()
plotname = (
    "/Users/lilli/Desktop/mflex/tests/soar_backden_"
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
    "/Users/lilli/Desktop/mflex/tests/soar_backtemp_"
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
    "/Users/lilli/Desktop/mflex/tests/soar_deltap_deltarho_"
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

# profile_file = "profile_results.txt"
# profiler = cProfile.Profile()
# profiler.runctx(code_to_profile, globals(), locals())
# profiler.dump_stats(profile_file)

# with open(profile_file, "w") as f:
#    stats = pstats.Stats(profiler, stream=f)
#    stats.sort_stats("cumulative")
#    stats.print_stats()
# print(f"Profiling results saved to {profile_file}")
