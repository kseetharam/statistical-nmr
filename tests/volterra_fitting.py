from stNMR.volterra.volterra_nn import VolterraNetwork
from stNMR.utils.runtime import seed_everything
from stNMR.utils.io import read_mat

from scipy import signal, fft
from matplotlib import pyplot as plt

# Set seed
seed_everything(42)

# Simulate some data
mat_path = "data/DFG_TwoSpinStochasticData.mat"
fids, ux, uy = read_mat(mat_path)

y = fids[0] + 1j * fids[1]

complex_input = (ux + 1j * uy).squeeze()

out = signal.convolve(complex_input, y, mode="full")

fig, axes = plt.subplots(nrows=4, ncols=1)

axes[0].plot(complex_input.real)
axes[0].set_title("$x(t)$")

axes[1].plot(y.real)
axes[1].set_title("$y(t)$")

axes[2].plot(out.real)
axes[2].set_title(r"$(y \ast x)(t)$")

# xf = fft.fftfreq(out.shape[0], 1)
# xf = fft.fftshift(xf)

axes[3].plot(fft.fft(out).real)
axes[3].set_title("$k^{(1)}(\sigma_1)$")

plt.show()
plt.clf()


