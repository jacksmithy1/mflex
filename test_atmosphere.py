from mflex.simulate.simulate_boundary import dipole, non_periodic, dalmatian
import numpy as np
from mflex.model.field.bfield_model import (
    magnetic_field,
    bz_partial_derivatives,
    magnetic_field_low,
    bz_partial_derivatives_low,
)
from mflex.plot.plot_magnetogram import (
    plot_fieldlines_grid,
    plot_magnetogram_boundary,
    plot_fieldlines_polar,
)
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
    temp,
    deltaden_low,
    deltapres_low,
)
from datetime import datetime
from mflex.model.field.utility.height_profile import f_low, f
from mflex.model.field.utility.poloidal import phi, dphidz


nresol_x: int = 100
nresol_y: int = 100
nresol_z: int = 200
xmin: np.float64 = 0.0
xmax: np.float64 = 2.0  # in units of 10^4 km, therefore corresponds to 20Mm
ymin: np.float64 = 0.0
ymax: np.float64 = 2.0
zmin: np.float64 = 0.0
zmax: np.float64 = 2.0
z0: np.float64 = 0.2
pixelsize_x: np.float64 = (xmax - xmin) / nresol_x
pixelsize_y: np.float64 = (ymax - ymin) / nresol_y
pixelsize_z: np.float64 = (zmax - zmin) / nresol_z
nf_max = 100
deltaz: np.float64 = np.float64(z0 / 10.0)

t_photosphere = 5600.0  # Temperature at z = 0 (on photosphere) in Kelvin
t_corona = 2.0 * 10.0**6  # Temperature at z = 2.0 (at 20 Mm) in Kelvin
t0 = (t_photosphere + t_corona * np.tanh(z0 / deltaz)) / (
    1.0 + np.tanh(z0 / deltaz)
)  # Temperature at z = z0 in Kelvin
t1 = (t_corona - t_photosphere) / (
    1.0 + np.tanh(z0 / deltaz)
)  # t_corona - t0 in Kelvin
g_solar = 274.0  # gravitational acceleration in m/s^2
kB = 1.380649 * 10**-23  # Boltzmann constant in Joule/ Kelvin = kg m^2/(Ks^2)
mbar = 1.67262 * 10**-27  # mean molecular weight (proton mass)
h = (
    kB * t0 / (mbar * g_solar) * 10**-6 * 10**-1
)  # pressure scale height in 10^4 km (10**-6 to convert to Mm, and 10**-1 to convert to 10Mm = 10^4 km)
rho0 = 3.0**-4  # plasma density at z = 0 in kg/(m^3)
b0 = 500.0  # Gauss background magnetic field strength in 10^-4 kg/(s^2A) = 10^-4 T
p0 = t_photosphere * kB * rho0 / mbar  # plasma pressure in kg/(s^2 m)
mu0 = 1.25663706 * 10**-6  # permeability of free space in mkg/(s^2A^2)
pB0 = (b0 * 10**-4) ** 2 / (2 * mu0)  # magnetic pressure b0**2 / 2mu0 in kg/(s^2m)
beta0 = p0 / pB0  # Plasma Beta, ration plasma to magnetic pressure
print("plasma pressure", p0)
print("magnetic pressure", pB0)
print("beta0", beta0)

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
backden = 0.0 * z_arr
backtemp = 0.0 * z_arr

for iz in range(nresol_z):
    z = z_arr[iz]
    backpres[iz] = bpressure(z, z0, deltaz, h, t0, t1)
    backden[iz] = bdensity(z, z0, deltaz, h, t0, t1)
    backtemp[iz] = btemp(z, z0, deltaz, t0, t1)

plt.plot(
    z_arr,
    backpres,
    label="Background Pressure",
    linewidth=0.5,
    color="lightblue",
)
plt.xlim([0, 2 * z0])
plt.legend()
plt.show()

plt.plot(
    z_arr,
    backden,
    label="Background Density",
    linewidth=0.5,
    color="blue",
)
plt.xlim([0, 2 * z0])
plt.legend()
plt.show()

plt.plot(
    z_arr,
    backtemp,
    label="Background Temperature",
    linewidth=0.5,
    color="pink",
)
plt.legend()
plt.xlim([0, 2 * z0])
plt.show()

data_bz: np.ndarray[np.float64, np.dtype[np.float64]] = np.zeros((nresol_y, nresol_x))
for ix in range(0, nresol_x):
    for iy in range(0, nresol_y):
        x = x_arr[ix]
        y = y_arr[iy]
        data_bz[iy, ix] = dipole(x, y)

a = 0.149
alpha = 1.0
b = 1.0

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
b_back_small = b_back[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x]
maxcoord = np.unravel_index(
    np.argmax(b_back_small, axis=None),
    b_back_small.shape,
)
iy_max: int = int(maxcoord[0])
ix_max: int = int(maxcoord[1])
maxb0 = b_back_small[iy_max, ix_max]
print(maxb0)
exit()

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

dpres = 0.0 * z_arr
dden = 0.0 * z_arr
fpres = 0.0 * z_arr
fden = 0.0 * z_arr
ffunc = 0.0 * z_arr
ftemp = 0.0 * z_arr

for iz in range(nresol_z):
    z = z_arr[iz]
    bz = bfield[iy_max, ix_max, iz, 2]
    bzdotgradbz = (
        dpartial_bfield[iy_max, ix_max, iz, 1] * bfield[iy_max, ix_max, iz, 1]
        + dpartial_bfield[iy_max, ix_max, iz, 0] * bfield[iy_max, ix_max, iz, 0]
        + dpartial_bfield[iy_max, ix_max, iz, 2] * bfield[iy_max, ix_max, iz, 2]
    )
    dpres[iz] = deltapres(z, z0, deltaz, a, b, bz)
    dden[iz] = deltaden(z, z0, deltaz, a, b, bz, bzdotgradbz)
    fpres[iz] = pres(z, z0, deltaz, a, b, beta0, bz, h, t0, t1)
    fden[iz] = den(
        z, z0, deltaz, a, b, bz, bzdotgradbz, beta0, h, t0, t1, t_photosphere
    )
    """
    ftemp[iz] = temp(
        z, z0, deltaz, a, b, bz, bzdotgradbz, beta0, h, t0, t1, t_photosphere
    )"""

# plt.plot(z_arr, dpres, label="Delta Pressure", linewidth=0.5, color="lightblue")
plt.plot(z_arr, dden, label="Delta Density", linewidth=0.5, color="magenta")
plt.axvline(
    x=z0, color="black", label="z0 = " + str(z0), linestyle="dashed", linewidth=0.5
)
plt.xlim([0, 2 * z0])
plt.legend()
plt.show()

plt.plot(z_arr, dpres, label="Delta Pressure", linewidth=0.5, color="lightblue")
# plt.plot(z_arr, dden, label="Delta Density", linewidth=0.5, color="magenta")
plt.axvline(
    x=z0, color="black", label="z0 = " + str(z0), linestyle="dashed", linewidth=0.5
)
plt.xlim([0, 2 * z0])
plt.legend()
plt.show()


plt.plot(z_arr, fpres, label="Pressure", linewidth=0.5, color="lightblue")
# plt.plot(z_arr, fden, label="Density", linewidth=0.5, color="magenta")
plt.axvline(
    x=z0, color="black", label="z0 = " + str(z0), linestyle="dashed", linewidth=0.5
)
plt.xlim([0, 2 * z0])
plt.legend()
plt.show()

# plt.plot(z_arr, fpres, label="Pressure", linewidth=0.5, color="lightblue")
plt.plot(z_arr, fden, label="Density", linewidth=0.5, color="magenta")
plt.axvline(
    x=z0, color="black", label="z0 = " + str(z0), linestyle="dashed", linewidth=0.5
)
plt.xlim([0, 2 * z0])
plt.legend()
plt.show()
