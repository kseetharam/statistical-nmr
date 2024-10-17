import numpy as np
import matplotlib.pyplot as plt

from stNMR.utils.io import read_mat

mat_path = "data/DFG_TwoSpinStochasticData.mat"
fids, ux, uy = read_mat(mat_path)

x_t = (ux + 1j * uy).squeeze()
y_t = fids[0, :] + 1j * fids[1, :]
Nsamp = len(x_t)
Nsamp1 = int(1e4)
Navg = int(Nsamp / Nsamp1)
h = 0

for i in range(Navg):
    idx_now = slice(Nsamp1 * i, Nsamp1 * (i + 1))
    y_tnow = y_t[idx_now]
    x_tnow = x_t[idx_now]
    h += (np.fft.fft(y_tnow) * np.conj(np.fft.fft(x_tnow))) / np.mean(np.abs(np.fft.fft(x_tnow))**2)

plt.plot(-np.real(h))
plt.show()
