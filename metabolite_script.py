paramMat = np.loadtxt(matPath + '/' + mLabel + '.csv', delimiter=',')
paramMat[i, i] = paramMat[i, i] * toggleDict['refFreqHz'] * 1e-6  # actual parameter in Hz is given by multiplying by the reference frequency (in Hz) and ppm (10^-6). ***the off-diagonal components (Jij) are given directly in Hz
N = paramMat.shape[0]
hiList = [[paramMat[i, i], i] for i in np.arange(N)]  # extracts hi from parameter matrix (puts in form for QuSpin)
JijList = [[2 * paramMat[i, j], i, j] for i in np.arange(N) for j in np.arange(N) if (i != j) and (i < j) if not np.isclose(paramMat[i, j], 0)]  # extracts Jij from parameter matrix (puts in form for QuSpin); this list combines the Jij and Jji terms (Hermitian conjugates) into a single term

pMin = np.min(np.abs(paramMat[np.nonzero(paramMat)]))  # smallest parameter in the Hamiltonian - used to set the timescale of S(t|\theta) (smallest energy sets longest time needed to sample until)
pMax = np.max(np.abs(paramMat[np.nonzero(paramMat)]))  # largest parameter in the Hamiltonian - used to set the sampling rate of S(t|\theta) (largest energy sets shortest time interval to sample)
tMax = 1/pMin; dt = 1/pMax
tgrid = np.arange(0, tMax + dt, dt)

# Set up QuSpin Hamiltonian & Sz_tot operators

spinBasis = spin_basis_1d(N, pauli=False)
H_theta = hamiltonian([["x", hiList],["xx", JijList], ["yy", JijList], ["zz", JijList]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False)
Sz_Tot = np.array([bin(bint).count("1") - (spinBasis.L / 2) for bint in spinBasis.states])  # total z-magnetization is (N_up - N_down)/2 = (N_up - (L - N_up))/2.