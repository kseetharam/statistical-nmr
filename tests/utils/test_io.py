from stNMR.utils.io import read_mat


mat_path = "data/DFG_TwoSpinStochasticData.mat"

fids, ux, uy = read_mat(mat_path)

print(fids.shape)
print(ux.shape)
print(uy.shape)


from matplotlib import pyplot as plt

plt.plot(ux)
plt.grid()
plt.show()
