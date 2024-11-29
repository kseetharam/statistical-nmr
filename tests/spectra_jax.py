import time
from pathlib import Path
from matplotlib import pyplot as plt

from stNMR.hamiltonian import NMRModelJax


s_time = time.time()
# file_path = Path("statistical-nmr/data/metabolites/urine/params") / "matrix3.csv"
file_path = Path("data/Difluoropropane.csv")
model = NMRModelJax.from_csv(file_path)

FID, time_series, FTspec, freq_series = model()

e_time = time.time()

print(FTspec.sum())
print(f"Time: {e_time - s_time}")

fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
axes.plot(freq_series, (FTspec).real)
axes.invert_xaxis()
axes.set_xlabel("Frequency (Hz)")
axes.tick_params(axis='both', labelsize=14)
axes.set_title("Simulated")

# plt.gcf().set_facecolor("w")
# plt.show()
plt.savefig((Path("figures/JAX+ODE") / file_path.name).with_suffix(".png"))
plt.clf()
