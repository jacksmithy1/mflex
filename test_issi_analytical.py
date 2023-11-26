#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mflex.load.read_file import read_issi_analytical
import numpy as np
from mflex.plot.plot_magnetogram import (
    plot_fieldlines_grid,
    plot_magnetogram_boundary,
    plot_magnetogram_boundary_3D,
)
from mflex.model.field.bfield_model import magnetic_field, bz_partial_derivatives
import scipy
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
    btemp_linear,
    temp,
)
from datetime import datetime
from mflex.model.field.utility.height_profile import f

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

a: float = 0.149
alpha: float = 1.0
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

b_back = np.zeros((2 * nresol_y, 2 * nresol_x))
b_back = bfield[:, :, 0, 2]
# plot_magnetogram_boundary(b_back, nresol_x, nresol_y)
"""
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
)"""

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

t_photosphere = 6000.0
t_corona = 1.0 * 10.0**6
t_z0 = 10000.0
t0 = (t_photosphere + t_corona * np.tanh(z0 / deltaz)) / (1.0 + np.tanh(z0 / deltaz))
t1 = (t_corona - t_photosphere) / (1.0 + np.tanh(z0 / deltaz))
t0 = t_z0
t1 = t0 - t_photosphere
g_solar = 272.2
mbar = 1.0  # Mean molecular weight
h = (
    1.3807 * t0 / (mbar * 1.6726 * g_solar) * 0.001
)  # presure scale height = kB * t0 / (mbar*g) in units of 10^4
rho0 = 3.0**-4  # plasma density at z = 0
b0 = 500.0  # 500 Gauss background magnetic field strength
b0 = 1.0
p0 = (
    1.3807 * t_photosphere * rho0 / (mbar * 1.6726) * 1.0 * 10**4
)  # Ideal gas law fulfilled on photosphere
pB0 = 3.9789 * 10**-3 * b0**2  # Magnetic pressure on photosphere
beta0 = p0 / pB0
b = 1.0

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

# for iz in range(nresol_z):
#    z = z_arr[iz]
#    backpres[iz] = bpressure(z, z0, deltaz, h, t0, t1)
#    backden[iz] = bdensity(z, z0, deltaz, h, t0, t1)
#    backtemp[iz] = btemp(z, z0, deltaz, t0, t1)

# plt.plot(z_arr, backtemp, label="Background temperature", linewidth=0.5)
# plt.plot(z_arr, backden, label="Background density", linewidth=0.5)
# plt.plot(z_arr, backpres, label="Background pressure", linewidth=0.5)
# plt.yscale("log")
# plt.legend()
# plt.show()

# exit()
b_back_small = b_back[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x]
maxcoord = np.unravel_index(
    np.argmax(b_back_small, axis=None),
    b_back_small.shape,
)
iy: int = int(maxcoord[0])
ix: int = int(maxcoord[1])
print(ix, iy)
print(x_arr[ix], y_arr[iy])
dpres = 0.0 * z_arr
dden = 0.0 * z_arr
fpres = 0.0 * z_arr
fden = 0.0 * z_arr
ftemp = 0.0 * z_arr
ffunc = 0.0 * z_arr

for iz in range(nresol_z):
    z = z_arr[iz]
    bz = bfield[iy, ix, iz, 2]
    # print(iz, bz)
    bzdotgradbz = (
        dpartial_bfield[iy, ix, iz, 1] * bfield[iy, ix, iz, 1]
        + dpartial_bfield[iy, ix, iz, 0] * bfield[iy, ix, iz, 0]
        + dpartial_bfield[iy, ix, iz, 2] * bfield[iy, ix, iz, 2]
    )
    backpres[iz] = bpressure(z, z0, deltaz, h, t0, t1)
    backden[iz] = bdensity(z, z0, deltaz, h, t0, t1)
    backtemp[iz] = btemp(z, z0, deltaz, t0, t1)
    dpres[iz] = deltapres(z, z0, deltaz, a, b, bz)
    dden[iz] = deltaden(z, z0, deltaz, a, b, bz, bzdotgradbz)
    fpres[iz] = pres(z, z0, deltaz, a, b, beta0, bz, h, t0, t1)
    fden[iz] = den(
        z, z0, deltaz, a, b, bz, bzdotgradbz, beta0, h, t0, t1, t_photosphere
    )
    ftemp[iz] = temp(
        z, z0, deltaz, a, b, bz, bzdotgradbz, beta0, h, t0, t1, t_photosphere
    )
    # ffunc[iz] = f(z, z0, deltaz, a, b) * bz**2.0 / 2.0


# plt.plot(z_arr, backpres, label="Background Pres", linewidth=0.5, color="green")
# plt.plot(z_arr, backden, label="Background Den", linewidth=0.5, color="red")
# plt.plot(z_arr, backtemp, label="Background Temp", linewidth=0.5, color="orange")
# plt.plot(z_arr, ftemp, label="Temp", linewidth=0.5, color="orange")
# plt.plot(z_arr, dpres, label="Delta Pres", linewidth=0.5, color="lightblue")
# plt.plot(z_arr, ffunc, label="Delta Pres", linewidth=0.5, color="lightblue")
# plt.plot(z_arr, dden, label="Delta Den", linewidth=0.5, color="pink")
# plt.plot(z_arr, fpres, label="Pressure", linewidth=0.5, color="navy")
plt.plot(z_arr, fden, label="Density", linewidth=0.5, color="purple")
plt.axvline(
    x=z0, color="black", label="z0 = " + str(z0), linestyle="dashed", linewidth=0.5
)
plt.legend()
plt.xlim([0, 2 * z0])
plt.show()
exit()
current_time = datetime.now()
dt_string = current_time.strftime("%d-%m-%Y_%H-%M-%S")


"""plt.plot(z_arr, backpres, label="Background pressure", linewidth=0.5, color="orange")
plt.axvline(
    x=z0, color="black", label="z0 = " + str(z0), linestyle="dashed", linewidth=0.5
)
plt.legend()
plotname = (
    "/Users/lilli/Desktop/mflex/tests/issi_analytical_backpres_"
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
exit()"""
"""plt.plot(z_arr, backden, label="Background density", linewidth=0.5, color="magenta")
plt.legend()
plotname = (
    "/Users/lilli/Desktop/mflex/tests/issi_analytical_backden_"
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
    "/Users/lilli/Desktop/mflex/tests/issi_analytical_backtemp_"
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
plt.show()"""


"""plt.plot(
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
)"""
plt.plot(z_arr, backpres, label="Background Pres", linewidth=0.5, color="lightblue")
# plt.plot(z_arr, backden, label="Background den", linewidth=0.5, color="purple")
plt.axvline(
    x=z0, color="black", label="z0 = " + str(z0), linestyle="dashed", linewidth=0.5
)
plt.xlim([0.0, 2 * z0])
plt.legend()
"""plotname = (
    "/Users/lilli/Desktop/mflex/tests/issi_analytical_deltap_deltarho_"
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
plt.savefig(plotname, dpi=300)"""
plt.show()
exit()
"""
"""

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
