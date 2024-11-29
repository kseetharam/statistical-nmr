from pathlib import Path
from matplotlib import pyplot as plt

from stNMR.hamiltonian.simulation_module import NMRModel


file_path = Path("statistical-nmr/data/metabolites/urine/params") / "matrix1.csv"
model = NMRModel.from_csv(file_path)

FID, time_series, FTspec, freq_series = model()


fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
axes.plot(freq_series.numpy(), (FTspec).real.numpy())
axes.invert_xaxis()
axes.set_xlabel("Frequency (Hz)")
axes.tick_params(axis='both', labelsize=14)
axes.set_title("Simulated")

# plt.gcf().set_facecolor("w")
plt.show()
# plt.savefig((Path("figures") / file.name).with_suffix(".png"))
plt.clf()