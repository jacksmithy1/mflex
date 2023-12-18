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
from mflex.model.field.utility.height_profile import f_low, f, dfdz
from mflex.model.field.utility.poloidal import phi, dphidz
from PIL import Image

nresol_x = 100
nresol_y = 100
nresol_z = 400
xmin = 0.0
xmax = 2.0  # in units of 10^4 km, therefore corresponds to 20Mm
ymin = 0.0
ymax = 2.0
zmin = 0.0
zmax = 2.0
z0 = 0.2
pixelsize_x = (xmax - xmin) / nresol_x
pixelsize_y = (ymax - ymin) / nresol_y
pixelsize_z = (zmax - zmin) / nresol_z
nf_max = 100
deltaz = z0 / 10.0

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
rho0 = 3.0 * 10**-4  # plasma density at z = 0 in kg/(m^3)
b0 = 500.0  # Gauss background magnetic field strength in 10^-4 kg/(s^2A) = 10^-4 T
p0 = t_photosphere * kB * rho0 / mbar  # plasma pressure in kg/(s^2 m)
mu0 = 1.25663706 * 10**-6  # permeability of free space in mkg/(s^2A^2)
pB0 = (b0 * 10**-4) ** 2 / (2 * mu0)  # magnetic pressure b0**2 / 2mu0 in kg/(s^2m)
beta0 = p0 / pB0  # Plasma Beta, ration plasma to magnetic pressure
h_photo = h / t0 * t_photosphere

a = 0.12
alpha = 0.25
b = 1.0

z0_b = 0.1
deltaz_b = 0.02

print("Atmospheric z0", z0)
print("Atmospheric Delta z", deltaz)
print("Magnetic field z0", z0_b)
print("Magnetic field Delta z", deltaz_b)

print("Temperature Photosphere", t_photosphere)
print("Temperature Corona", t_corona)
print("T0", t0)
print("T1", t1)
print("g", g_solar)
print("mbar", mbar)
print("Pressure Scale height z0", h)
print("Pressure Scale height photosphere", h_photo)
print("Plasma Density Photosphere", rho0)
print("Magnetic field strength Photosphere", b0)
print("Plasma Pressure Photosphere", p0)
print("Magnetic Pressure Photosphere", pB0)
print("Plasma Beta", beta0)
print("Density scaling", 0.5 * beta0 / h * t0 / t_photosphere)

x_arr = np.arange(nresol_x) * (xmax - xmin) / (nresol_x - 1) + xmin
y_arr = np.arange(nresol_y) * (ymax - ymin) / (nresol_y - 1) + ymin
z_arr = np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin
x_plot = np.outer(y_arr, np.ones(nresol_x))
y_plot = np.outer(x_arr, np.ones(nresol_y)).T

backpres = 0.0 * z_arr
backden = 0.0 * z_arr
backtemp = 0.0 * z_arr

for iz in range(nresol_z):
    z = z_arr[iz]
    backpres[iz] = bpressure(z, z0, deltaz, h, t0, t1)
    backden[iz] = bdensity(z, z0, deltaz, h, t0, t1)
    backtemp[iz] = btemp(z, z0, deltaz, t0, t1)

"""
plt.plot(
    z_arr,
    backtemp,
    label="Background Temperature",
    linewidth=0.5,
    color="orange",
)
plt.yscale("log")
plt.legend()
plotname = (
    "/Users/lilli/Desktop/mflex/p_d_tests_VonMises/backtemp_"
    + str(a)
    + "_"
    + str(b)
    + "_"
    + str(alpha)
    + "_"
    + str(z0_b)
    + "_"
    + str(deltaz_b)
    + "_"
    + str(nf_max)
    + ".png"
)
plt.savefig(plotname, dpi=300)
# plt.xlim([0, 2 * z0])
plt.show()

plt.plot(
    z_arr,
    backpres,
    label="Background Pressure",
    linewidth=0.5,
    color="royalblue",
)
plt.yscale("log")
plt.legend()
plotname = (
    "/Users/lilli/Desktop/mflex/p_d_tests_VonMises/backpres_"
    + str(a)
    + "_"
    + str(b)
    + "_"
    + str(alpha)
    + "_"
    + str(z0_b)
    + "_"
    + str(deltaz_b)
    + "_"
    + str(nf_max)
    + ".png"
)
plt.savefig(plotname, dpi=300)
# plt.xlim([0, 2 * z0])
plt.show()

plt.plot(
    z_arr,
    backden,
    label="Background Density",
    linewidth=0.5,
    color="magenta",
)
# plt.xlim([0, 2 * z0])
plt.legend()
plt.yscale("log")
plotname = (
    "/Users/lilli/Desktop/mflex/p_d_tests_VonMises/backden_"
    + str(a)
    + "_"
    + str(b)
    + "_"
    + str(alpha)
    + "_"
    + str(z0_b)
    + "_"
    + str(deltaz_b)
    + "_"
    + str(nf_max)
    + ".png"
)
plt.savefig(plotname, dpi=300)
plt.show()
"""
data_bz = np.zeros((nresol_y, nresol_x))
for ix in range(0, nresol_x):
    for iy in range(0, nresol_y):
        x = x_arr[ix]
        y = y_arr[iy]
        data_bz[iy, ix] = dipole(x, y)

"""
fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(y_plot, x_plot, data_bz, 1000, cmap="bone")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_box_aspect(1.0)
plt.show()
exit()
"""

f_func = f(z_arr, z0_b, deltaz_b, a, b)
df_func = dfdz(z_arr, z0_b, deltaz_b, a, b)

plt.plot(
    z_arr,
    f_func,
    linewidth=0.5,
    color="black",
    label="f",
    linestyle="dotted",
)
plt.axvline(x=z0, color="black", linestyle="solid", linewidth=0.25)
plt.legend()
plt.xlabel("z")
plt.ylabel("f(z)")
plotname = (
    "/Users/lilli/Desktop/mflex/p_d_tests_VonMises/f_"
    + str(a)
    + "_"
    + str(b)
    + "_"
    + str(alpha)
    + "_"
    + str(z0_b)
    + "_"
    + str(deltaz_b)
    + "_"
    + str(nf_max)
    + ".png"
)
plt.savefig(plotname, dpi=300)
plt.show()

plt.plot(
    z_arr,
    df_func,
    linewidth=0.5,
    color="black",
    label="dfdz",
    linestyle="dashed",
)
plt.axvline(x=z0, color="black", linestyle="solid", linewidth=0.25)
plt.legend()
plt.xlabel("z")
plt.ylabel("df(z)/dz")
plotname = (
    "/Users/lilli/Desktop/mflex/p_d_tests_VonMises/dfdz_"
    + str(a)
    + "_"
    + str(b)
    + "_"
    + str(alpha)
    + "_"
    + str(z0_b)
    + "_"
    + str(deltaz_b)
    + "_"
    + str(nf_max)
    + ".png"
)
plt.savefig(plotname, dpi=300)
plt.show()

bfield = magnetic_field(
    data_bz,
    z0_b,
    deltaz_b,
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
iy_max = int(maxcoord[0])
ix_max = int(maxcoord[1])
maxb0 = b_back_small[iy_max, ix_max]
print("Maximum Bz on photosphere", maxb0)

maxcoord_gl = np.unravel_index(
    np.argmax(bfield[:, :, :, 2], axis=None),
    bfield[:, :, :, 2].shape,
)
iy_max_gl = int(maxcoord_gl[0])
ix_max_gl = int(maxcoord_gl[1])
iz_max_gl = int(maxcoord_gl[2])
maxb0_gl = bfield[iy_max_gl, ix_max_gl, iz_max_gl, 2]
print("Maximum Bz", maxb0_gl)

bz_sqr = np.zeros((nresol_y, nresol_x, nresol_z))
bz_sqr[:, :, :] = np.multiply(
    bfield[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x, :, 2],
    bfield[nresol_y : 2 * nresol_y, nresol_x : 2 * nresol_x, :, 2],
)

dpartial_bfield = bz_partial_derivatives(
    data_bz,
    z0_b,
    deltaz_b,
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

dpres = np.zeros((nresol_y, nresol_x, nresol_z))
dden = 0.0 * dpres
fpres = 0.0 * dpres
fden = 0.0 * dpres
ffunc = 0.0 * dpres

for ix in range(nresol_x):
    for iy in range(nresol_y):
        for iz in range(nresol_z):
            x = x_arr[ix]
            y = y_arr[iy]
            z = z_arr[iz]
            bz = bfield[iy, ix, iz, 2]
            bzdotgradbz = (
                dpartial_bfield[iy, ix, iz, 1] * bfield[iy, ix, iz, 1]
                + dpartial_bfield[iy, ix, iz, 0] * bfield[iy, ix, iz, 0]
                + dpartial_bfield[iy, ix, iz, 2] * bfield[iy, ix, iz, 2]
            )
            dpres[iy, ix, iz] = deltapres(z, z0_b, deltaz_b, a, b, bz)
            dden[iy, ix, iz] = deltaden(z, z0_b, deltaz_b, a, b, bz, bzdotgradbz)
            fpres[iy, ix, iz] = pres(
                z, z0, deltaz, z0_b, deltaz_b, a, b, beta0, bz, h, t0, t1
            )
            fden[iy, ix, iz] = den(
                z,
                z0,
                deltaz,
                z0_b,
                deltaz_b,
                a,
                b,
                bz,
                bzdotgradbz,
                beta0,
                h,
                t0,
                t1,
                t_photosphere,
            )

"""
for iz in range(0, nresol_z):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(y_plot, x_plot, fpres[:, :, iz], cmap="bone")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlim([0.0, 7.0])
    ax.view_init(30, -115, 0)
    filename = "3d/3d_pressure_test_" + str(iz) + ".png"
    plt.savefig(filename, dpi=300)

png_count = nresol_z
files = []
for iz in range(png_count):
    file_names = "3d/3d_pressure_test_" + str(iz) + ".png"
    files.append(file_names)

frames = []
for i in files:
    new_frame = Image.open(i)
    frames.append(new_frame)

frames[0].save(
    "3d_vis.gif",
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=100,
    loop=0,
)
"""

for ix in range(0, nresol_x, 8):
    for iy in range(1, nresol_y, 8):
        plt.plot(
            z_arr,
            fpres[ix, iy, :],
            linewidth=0.1,
            color="royalblue",
            linestyle="dotted",
        )
        plt.plot(
            z_arr,
            dpres[ix, iy, :],
            linewidth=0.1,
            color="lightblue",
            linestyle="dashed",
        )
        # plt.plot(z_arr, 0.5 * beta0 * backpres[ix, iy, :], linewidth=0.2, color="black", linestyle="solid")
plt.xlim([0, z0])
plt.legend()
plotname = (
    "/Users/lilli/Desktop/mflex/p_d_tests_VonMises/fpres_dpres_"
    + str(a)
    + "_"
    + str(b)
    + "_"
    + str(alpha)
    + "_"
    + str(z0_b)
    + "_"
    + str(deltaz_b)
    + "_"
    + str(nf_max)
    + ".png"
)
plt.savefig(plotname, dpi=300)
plt.show()

for ix in range(0, nresol_x, 8):
    for iy in range(1, nresol_y, 8):
        plt.plot(
            z_arr,
            fden[ix, iy, :],
            linewidth=0.1,
            color="magenta",
            linestyle="dotted",
        )
        plt.plot(
            z_arr,
            dden[ix, iy, :],
            linewidth=0.1,
            color="red",
            linestyle="dashed",
        )
plt.xlim([0, z0])
plt.legend()
plotname = (
    "/Users/lilli/Desktop/mflex/p_d_tests_VonMises/fden_dden_"
    + str(a)
    + "_"
    + str(b)
    + "_"
    + str(alpha)
    + "_"
    + str(z0_b)
    + "_"
    + str(deltaz_b)
    + "_"
    + str(nf_max)
    + ".png"
)
plt.savefig(plotname, dpi=300)
plt.show()

mincoord = np.unravel_index(
    np.argmin(fpres, axis=None),
    fpres.shape,
)
print(mincoord)
iy_min = int(mincoord[0])
ix_min = int(mincoord[1])
plt.plot(
    z_arr,
    fpres[iy_min, ix_min, :],
    linewidth=0.5,
    color="black",
    linestyle="dotted",
    label="Full pressure",
)
plt.plot(
    z_arr,
    -dpres[iy_min, ix_min, :],
    linewidth=0.5,
    color="black",
    linestyle="dashed",
    label="-Delta pressure",
)
plt.plot(
    z_arr,
    0.5 * beta0 * backpres,
    linewidth=0.5,
    color="black",
    linestyle="solid",
    label="Background pressure",
)
plt.xlim([0, 1.0])
# plt.yscale("log")
plt.legend()
plotname = (
    "/Users/lilli/Desktop/mflex/p_d_tests_VonMises/pres_at_min_"
    + str(a)
    + "_"
    + str(b)
    + "_"
    + str(alpha)
    + "_"
    + str(z0_b)
    + "_"
    + str(deltaz_b)
    + "_"
    + str(nf_max)
    + ".png"
)
plt.savefig(plotname, dpi=300)
plt.show()

mincoord = np.unravel_index(
    np.argmin(fden, axis=None),
    fden.shape,
)
print(mincoord)
iy_min = int(mincoord[0])
ix_min = int(mincoord[1])

plt.plot(
    z_arr,
    fden[iy_min, ix_min, :],
    linewidth=0.5,
    color="black",
    linestyle="dotted",
    label="Full density",
)
plt.plot(
    z_arr,
    -dden[iy_min, ix_min, :],
    linewidth=0.5,
    color="black",
    linestyle="dashed",
    label="-Delta density",
)
plt.plot(
    z_arr,
    0.5 * beta0 / h * t0 / t_photosphere * backden,
    linewidth=0.5,
    color="black",
    linestyle="solid",
    label="Background density",
)
plt.xlim([0, 1.0])
# plt.yscale("log")
plt.legend()
plotname = (
    "/Users/lilli/Desktop/mflex/p_d_tests_VonMises/den_at_min_"
    + str(a)
    + "_"
    + str(b)
    + "_"
    + str(alpha)
    + "_"
    + str(z0_b)
    + "_"
    + str(deltaz_b)
    + "_"
    + str(nf_max)
    + ".png"
)
plt.savefig(plotname, dpi=300)
plt.show()

print("Minimum pressure", fpres.min())
print(
    "Minimum pressure index", np.unravel_index(np.argmin(fpres, axis=None), fpres.shape)
)
iz_pres = np.unravel_index(np.argmin(fpres, axis=None), fpres.shape)[2]
print("Background pressure at Minimum", backpres[iz_pres])
print("Minimum density", fden.min())
print("Minimum density index", np.unravel_index(np.argmin(fden, axis=None), fden.shape))
iz_den = np.unravel_index(np.argmin(fden, axis=None), fden.shape)[2]
print("Background density at Minimum", backden[iz_den])
exit()
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(y_plot, x_plot, fpres[:, :, iz_pres], cmap="bone")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.view_init(30, -115, 0)
plt.title("Full pressure at pressure minimum")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(y_plot, x_plot, fden[:, :, iz_den], cmap="bone")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.view_init(30, -115, 0)
plt.title("Full density at density minimum")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(y_plot, x_plot, fpres[:, :, 0], cmap="bone")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.view_init(30, -115, 0)
plt.title("Full pressure at photosphere")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(y_plot, x_plot, fden[:, :, 0], cmap="bone")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.view_init(30, -115, 0)
plt.title("Full density at photosphere")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(y_plot, x_plot, fpres[:, :, 19], cmap="bone")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.view_init(30, -115, 0)
plt.title("Full pressure at z0")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(y_plot, x_plot, fden[:, :, 19], cmap="bone")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.view_init(30, -115, 0)
plt.title("Full density at z0")
plt.show()
