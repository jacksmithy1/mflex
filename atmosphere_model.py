import numpy as np
from mflex.model.plasma_parameters import bpressure, btemp, bdensity
import matplotlib.pyplot as plt

z0 = 0.2
deltaz = 0.02
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
b = 1.0

nresol_z = 80
zmin = 0.0
zmax = 1.0

z_arr: np.ndarray[np.float64, np.dtype[np.float64]] = (
    np.arange(nresol_z) * (zmax - zmin) / (nresol_z - 1) + zmin
)
backpres = 0.0 * z_arr
backtemp = 0.0 * z_arr
backden = 0.0 * z_arr

for iz in range(nresol_z):
    z = z_arr[iz]
    backpres[iz] = bpressure(z, z0, deltaz, h, t0, t1)
    backden[iz] = bdensity(z, z0, deltaz, h, t0, t1)
    backtemp[iz] = btemp(z, z0, deltaz, t0, t1)

plt.plot(z_arr, backpres, label="Background pressure", linewidth=0.5)
plt.plot(z_arr, backden, label="Background density", linewidth=0.5)
plt.plot(z_arr, backtemp, label="Background temperature", linewidth=0.5)
plt.legend()
plt.show()
