from spinED import *

# helper functions

def encode_params(JAB, JBB, Delta):
    nA = JAB.shape[0]
    nB = JAB.shape[1]
    assert JBB.shape[0]==nB
    assert JBB.shape[1]==nB
    assert len(Delta) == nB

    dim = int(nA*nB+nB*(nB+1)/2)
    theta = np.zeros([dim])
    JAB = np.reshape(JAB, [nA, nB])
    theta[:nA*nB] = JAB
    counter = 0
    for i in range(nB):
        for j in range(i+1,nB):
            theta[nA*nB+counter] = JBB[i,j]
            counter += 1
    theta[nA*nB+counter:] = Delta
    return theta

def decode_params(theta, nA, nB):
    JAB = theta[:nA*nB]
    JAB = jnp.reshape(JAB, [nA, nB])
    JBB = jnp.zeros([nB, nB])
    counter = 0
    for i in range(nB):
        for j in range(i+1,nB):
            JBB = JBB.at[i,j].set(theta[nA*nB+counter])
            counter+=1
    Delta = theta[nA*nB+counter:]
    return JAB, JBB, Delta

def build_hadamards(n):

    def tensor_power(A0, L):
        A = A0
        for j in range(1,L):
            A = np.kron(A, A0)
        return A

    Hx = np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2)
    Hy = np.array([[1,-1j],[-1j,1]], dtype=complex)/np.sqrt(2)

    return tensor_power(Hx,n), tensor_power(Hy,n)

def get_trotter_energies(gAB, JAB, gBB, JBB, Delta):
    #g = np.array([1,1,-2])
    nA = JAB.shape[0]
    nB = JBB.shape[0]
    n = nA+nB
    
    indsA = np.arange(nA)
    indsB = np.arange(nA, n)
    JAB_pad = np.zeros([n, n])
    JAB_pad = JAB_pad.at[indsA[:,np.newaxis],indsB[np.newaxis,:]].set(JAB)
    JAB_pad = JAB_pad.at[indsB[:,np.newaxis],indsA[np.newaxis,:]].set(JAB.T)

    # Sensing Hamiltonian
    epAB = np.zeros([3, 2**n])
    z = 2*int2array(np.arange(2**n),n)-1 # (2**n, n)
    for mu in range(3):
        epAB = epAB.at[mu,:].set(gAB[mu]*np.diag(z @ JAB_pad @ z.T) / 4)

    # Molecule Hamiltonian
    epBB = np.zeros([3, 2**nB])
    z = 2*int2array(np.arange(2**nB), nB)-1 # (2**n, n)
    for mu in range(3):
        epBB = epBB.at[mu,:].set(gBB[mu]*np.diag(z @ JBB @ z.T) / 4)
        if mu ==2:
            epBB = epBB.at[mu,:].set(z @ Delta / 2)

    return epAB, epBB

def right_multiply(A, B):
    return np.swapaxes(np.tensordot(A, B, axes=((1),(0))),1,2)



class SensingSimulation:

    def __init__(self, nA, nB):
        self.nA = nA
        self.nB = nB
        self.n = nA + nB
        self.build_hadamards()
        self.secular = True # assume for now

    #### HAMILTONIAN BUILDING 

    def decode_params(self, theta):
        nA = self.nA
        nB = self.nB
        
        JAB = theta[:nA*nB]
        JAB = np.reshape(JAB, [nA, nB])
        JBB = np.zeros([nB, nB])
        #JBB = thetas[nA*nB+1:-1]
        counter = 0
        for i in range(nB):
            for j in range(i+1,nB):
                JBB[i,j]= theta[nA*nB+counter]
                counter+=1
        Delta = theta[nA*nB+counter:]
        return JAB, JBB, Delta

    def get_molecule_hamiltonian(self, g, JBB, Delta):
        # returns Hamiltonian acting on the molecule! (2**nB, 2**nB) matrix
        assert JBB.shape[0] == self.nB
        assert JBB.shape[1] == self.nB
        assert Delta.shape[0] == self.nB 
        nB = self.nB
        
        # assume secular dipolar Hamiltonian
        #g = np.array([1,1,-2])
        HBB = csc_matrix((2**nB, 2**nB), dtype='complex')
        for i in range(nB):
            HBB += 0.5*Delta[i]*build_local_operator(i, 'z', nB)
        for mu, p in enumerate(['x','y','z']):
            HBB += 0.5*g[mu]*build_interaction(JBB, p , p)

        return HBB

    def get_sensing_hamiltonian(self, g, JAB):
        nA = self.nA
        nB = self.nB
        n = self.n

        assert JAB.shape[0] == nA
        assert JAB.shape[1] == nB
        
        indsA = np.arange(nA)
        indsB = np.arange(nA, n)
        JAB_pad = np.zeros([n, n])
        JAB_pad[indsA[:,np.newaxis],indsB[np.newaxis,:]] = JAB
        JAB_pad[indsB[:,np.newaxis],indsA[np.newaxis,:]] = JAB.T

        # assume secular dipolar Hamiltonian
        # g = np.array([1,1,-2])

        HAB = csc_matrix((2**n, 2**n), dtype='complex')
        for mu, p in enumerate(['x','y','z']):
            HAB += 0.5*g[mu]*build_interaction(JAB_pad, p , p)

        return HAB

    #### EXACT TIME EVOLUTION AND STATE PREP

    def get_spectrum(self, theta):
        #nA = self.nA
        nB = self.nB
        n = self.n
        self.theta = theta
        JAB, JBB, Delta = self.decode_params(theta)
        
        HAB = self.get_sensing_hamiltonian(JAB)
        HBB = self.get_dipolar_molecule_hamiltonian(JBB, Delta)
        
        if self.secular:
            # AB system
            VAB = np.zeros([2**n, 2**n], dtype='complex')
            epAB = np.zeros(2**n)
            for inds in get_sec_inds(n):
                epAB_sec, VAB_sec = eigh(HAB[inds[:,np.newaxis],inds[np.newaxis,:]].toarray())
                VAB[inds[:,np.newaxis],inds[np.newaxis,:]] = VAB_sec
                epAB[inds] = epAB_sec

            # B system
            VBB = np.zeros([2**nB, 2**nB], dtype='complex')
            epBB = np.zeros(2**nB)
            for inds in get_sec_inds(nB):
                epBB_sec, VBB_sec = eigh(HBB[inds[:,np.newaxis], inds[np.newaxis,:]].toarray())
                VBB[inds[:,np.newaxis], inds[np.newaxis,:]] = VBB_sec
                epBB[inds] = epBB_sec

        else:
            VAB, epAB = eigh(HAB.toarray())
            VBB, epBB = eigh(HBB.toarray())

        self.spec = VAB, epAB, VBB, epBB
    
    def get_thermal_ensemble(self, specB, beta, M):
        VBB, epBB = specB
        psiB_list = []
        for m in range(M):
            psi0 = get_haar_random_state(2**self.nB, m)
            evolver = np.diag(np.exp(-beta*epBB/2))
            psiB = VBB @ evolver @ (VBB.conj().T @ psi0)
            psiB = psiB / norm(psiB)
            psiB_list.append(psiB)
        return psiB_list

    def propagate_exact(self, psi0_list, tau, ts, reverse_sense=True):
        # computes the quench (H_AB, H_BB, \pm H_AB) on a list of pure states
        # returns list of evolved pure states, each element is of shape (dimH, len(ts))
        nA = self.nA
        nB = self.nB
        n = self.n
        VAB, epAB, VBB, epBB = self.spec

        evolverAB = np.exp(-1j*epAB*tau) # (dimH)
        epBB_rep = np.repeat(epBB[np.newaxis,:], 2**nA, axis=0)
        evolverBB = np.exp(-1j*np.tensordot(epBB_rep, ts, axes=0)) # (dimA, dimB, N)
        
        psi_list = []
        for psi0 in psi0_list:
            # sense
            psi = VAB @ (evolverAB * (VAB.conj().T @ psi0))

            # time evolve molecule 
            Psi = np.reshape(psi, [2**nA, 2**nB])
            Psi = Psi @ VBB.conj() # (2**nA, 2**nB)
            Psi = np.repeat(Psi[:,:,np.newaxis], len(ts), axis=2) # (nA, nB, N)
            Psi = Psi*evolverBB 
            Psi = np.swapaxes(np.tensordot(Psi, VBB, axes=((1),(1))),1,2)# (2**nA, 2**nB, N)
            psi = np.reshape(Psi, [2**n, len(ts)]) # (dimH, N)

            # reverse sense
            if reverse_sense:
                psi = VAB @ (np.conj(np.diag(evolverAB)) @ (VAB.conj().T @ psi))
            
            psi_list.append(psi)

        return psi_list
    

    #### Measure sensor operators

    def measure_local_sensor_operators(self, psi_list):
        nA = self.nA
        S_list = []
        for psi in psi_list:
            S = np.zeros([3, nA, psi.shape[1]])
            for k in range(psi.shape[1]):
                rhoA, _ = self.partial_trace(psi[:,k])
                for j in range(nA):
                    for mu, p in enumerate(['x','y','z']):
                        O = build_local_operator(j,p,nA)
                        S[mu, j, k] =  np.trace(rhoA @ O)
            S_list.append(S)
        return S_list
    
    #### TROTTERIZED TIME EVOLUTION    

    # def set_trotter_energies(self, theta):
    #     # note this is only relavent for simulating Hamiltonians of the form Hx + Hy + Hz
    #     # will set diagonal trotter energy scales for both the sensing and molecule Hams
    #     # epAB is (3, 2**nA * 2**nB)
    #     # epBB is (3, 2**nB)
    #     nA = self.nA 
    #     nB = self.nB
    #     n = self.n
    #     g = np.array([1,1,-2])

    #     JAB, JBB, Delta = self.decode_params(theta)

    #     # pad extra zeros on coupling matrix
    #     indsA = np.arange(nA)
    #     indsB = np.arange(nA, n)
    #     JAB_pad = np.zeros([n, n])
    #     JAB_pad[indsA[:,np.newaxis],indsB[np.newaxis,:]] = JAB
    #     JAB_pad[indsB[:,np.newaxis],indsA[np.newaxis,:]] = JAB.T
        
    #     # Sensing Hamiltonian
    #     epAB = np.zeros([3, 2**n])
    #     z = 2*int2array(np.arange(2**n),n)-1 # (2**n, n)
    #     for mu in range(3):
    #         epAB[mu,:] = g[mu]*np.diag(z @ JAB_pad @ z.T) / 4

    #     # Molecule Hamiltonian
    #     epBB = np.zeros([3, 2**nB])
    #     z = 2*int2array(np.arange(2**nB),nB)-1 # (2**n, n)
    #     for mu in range(3):
    #         epBB[mu,:] = g[mu]*np.diag(z @ JBB @ z.T) / 4
    #         if mu ==2:
    #             epBB[mu,:] += z @ Delta / 2

    #     self.epAB = epAB
    #     self.epBB = epBB





    def apply_sensing_trotter_step(self, psi, dt):
        Hx = self.Hx_AB
        Hy = self.Hy_AB
        epAB = self.epAB
        DAB = np.exp(-1j * epAB * dt / 2) 

        # ZZ
        psi = DAB[2,:] * psi
        # XX
        psi = Hx.conj().T @ (DAB[0,:] * (Hx @ psi))
        # YY
        psi = Hy.conj().T @ (DAB[1,:]**2 * (Hy @ psi))
        # XX
        psi = Hx.conj().T @ (DAB[0,:] * (Hx @ psi))
        # ZZ
        psi = DAB[2,:] * psi
    
        return psi

    def apply_molecule_trotter_step(self, psi, dt):
        nA = self.nA
        nB = self.nB
        Hx = self.Hx_B
        Hy = self.Hy_B

        epBB = self.epBB

        DB = np.exp(-1j*epBB*dt/2)
        DB = np.repeat(DB[:,np.newaxis,:], 2**nA, axis=1)

        Psi = np.reshape(psi, [2**nA, 2**nB])
        # ZZ
        Psi = DB[2,:,:] * Psi
        # XX
        Psi =  ((Psi @ Hx.T) * DB[0,:,:]) @ Hx.conj()
        # YY
        Psi =  ((Psi @ Hy.T) * (DB[1,:,:]**2)) @ Hy.conj()
        # XX
        Psi =  ((Psi @ Hx.T) * DB[0,:,:]) @ Hx.conj()
        # ZZ
        Psi = DB[2,:,:] * Psi
        psi = np.ndarray.flatten(Psi)

        return psi # (2**nA, 2**nB) 

    def get_thermal_ensemble_trotter(self, dtau, beta, M):
        #VBB, epBB = specB
        psiB_list = []
        norm_list = []

        N_tau = int(beta/dtau)
        epBB = self.epBB
        #_, epBB = self.get_trotter_energies(theta)
        # right multipy the state
        #Psi = np.reshape(psi, [nA, nB])
        Hx = self.Hx_B
        Hy = self.Hy_B
        DB = np.exp(-epBB*dtau/2)
        for m in range(M):
            psi = get_haar_random_state(2**self.nB, m)
            for k in range(N_tau):
                # ZZ
                psi = DB[2,:] * psi
                # XX
                psi = Hx @ psi
                psi = Hx.conj().T @ (DB[0,:] * psi)
                # YY
                psi = Hy @ psi
                psi = Hy.conj().T @ (DB[1,:]**2 * psi)
                # XX
                psi = Hx @ psi
                psi = Hx.conj().T @ (DB[0,:] * psi)
                # ZZ
                psi = DB[2,:] * psi

            psiB_list.append(psi)
            norm_list.append(np.dot(psi.conj(), psi))
                    
        return psiB_list, norm_list


    def propagate_trotter(self, hyperparams):
        nA = self.nA
        nB = self.nB
        n = self.n

        # anistropy params
        gAB = hyperparams['gAB']
        gBB = hyperparams['gBB']

        # trotter params
        t = hyperparams['t']
        n_measure = hyperparams['n_measure']
        dt = hyperparams['dt']
        tau = hyperparams['tau']
        dtau = hyperparams['dtau']
        beta = hyperparams['beta']
        dbeta = hyperparams['dbeta']

        ts = np.arange(0, t, n_measure*dt)
        N_t = len(ts)
        N_tau = int(tau/dtau)
        N_beta = int((beta/2)/dbeta) # evolve the pure state by beta/2 to simulate finite temp!

        reverse_sense = hyperparams['reverse_sense']
        psi0 = hyperparams["psi0"]
        M = psi0.shape[1]

        Hx, Hy = build_hadamards(n)
        Hx_B, Hy_B = build_hadamards(nB)

        # any variational (input) parameters should be included in params.
        def apply_evolution(theta, args):
            
            # this is where the Ham params go! 
            JAB, JBB, Delta = decode_params(theta, nA, nB)
            epAB, epBB = get_trotter_energies(JAB, gAB, JBB, gBB, Delta) #  (3, 2**n), (3, 2**nB)
            epAB = np.repeat(epAB[:,:,np.newaxis], M, axis=2) #  (3, 2**n, M)
            epBB = np.repeat(epBB[:,:,np.newaxis], M, axis=2) # (3, 2**nB, M)

            # time evolution phases
            DAB = np.exp(-1j * epAB * dtau / 2) # (3, 2**n, M)
            DB = np.exp(-1j * epBB * dt / 2) # (3, 2**nB, M)
            DB = np.repeat(DB[:,np.newaxis,:,:], 2**nA, axis=1) # (3, 2**nA, 2**nB, M)

            # imag time evolution attenuators
            DB_imag = np.exp(- epBB * dbeta / 2) # (3, 2**nB, M)
            DB_imag = np.repeat(DB_imag[:,np.newaxis,:,:], 2**nA, axis=1) # (3, 2**nA, 2**nB, M)

            def apply_molecule_trotter_step_imag(tau0, psi):
                Psi = np.reshape(psi, [2**nA, 2**nB, M])
                Psi = DB_imag[2,:,:,:] * Psi # ZZ
                Psi = right_multiply(right_multiply(Psi, Hx_B.T) * DB_imag[0,:,:,:], Hx_B.conj()) # XX
                Psi = right_multiply(right_multiply(Psi, Hy_B.T) * DB_imag[1,:,:,:]**2, Hy_B.conj()) # YY
                Psi = right_multiply(right_multiply(Psi, Hx_B.T) * DB_imag[0,:,:,:], Hx_B.conj()) # XX
                Psi = DB_imag[2,:,:,:] * Psi # ZZ
                psi = np.reshape(Psi, [2**n, M])
                return psi 

            def apply_sensing_trotter_step(t0, psi): # psi is always (2**n, M)
                psi = DAB[2,:,:] * psi # ZZ
                psi = Hx.conj().T @ (DAB[0,:,:] * (Hx @ psi)) # XX
                psi = Hy.conj().T @ (DAB[1,:,:]**2 * (Hy @ psi)) # YY
                psi = Hx.conj().T @ (DAB[0,:,:] * (Hx @ psi)) # XX
                psi = DAB[2,:,:] * psi # ZZ
                return psi
            
            def apply_reverse_sensing_trotter_step(t0, psi): # psi is always (2**n, M)
                psi = DAB[2,:,:].conj() * psi # ZZ
                psi = Hx.conj().T @ (DAB[0,:,:].conj() * (Hx @ psi)) # XX
                psi = Hy.conj().T @ (DAB[1,:,:].conj()**2 * (Hy @ psi)) # YY
                psi = Hx.conj().T @ (DAB[0,:,:].conj() * (Hx @ psi)) #XX
                psi = DAB[2,:,:].conj() * psi # ZZ
                return psi

            def apply_molecule_trotter_step(t0, psi): # psi is always (2**n, M)
                Psi = np.reshape(psi, [2**nA, 2**nB, M])
                Psi = DB[2,:,:,:] * Psi # ZZ
                Psi = right_multiply(right_multiply(Psi, Hx_B.T) * DB[0,:,:,:], Hx_B.conj()) # XX
                Psi = right_multiply(right_multiply(Psi, Hy_B.T) * DB[1,:,:,:]**2, Hy_B.conj()) # YY
                Psi = right_multiply(right_multiply(Psi, Hx_B.T) * DB[0,:,:,:], Hx_B.conj()) # XX
                Psi = DB[2,:,:,:] * Psi # ZZ
                psi = np.reshape(Psi, [2**n, M])
                return psi 
            
            psi1 = psi0.copy()
            
            # imag time evolve
            psi1 = fori_loop(0, N_beta, apply_molecule_trotter_step_imag, psi1)
            psi1 = psi1/np.sqrt(np.trace(psi1.conj().T @ psi1)) # divide by estimator of partition function
            # note that now the columns of psi1 are not normalized, yet averaging uniformly over these states yields the correct density matrix

            # sense
            psi1 = fori_loop(0, N_tau, apply_sensing_trotter_step, psi1)

            # container for states to save
            psi_cont = np.zeros((N_t+1, 2**n, M), dtype='complex128')
            psi_cont = psi_cont.at[0,:,:].set(psi1)

            def evolve_molecule(t0, psi_cont):
                psi2 = psi_cont[t0,:,:]
                psi3 = fori_loop(0, N_tau, apply_reverse_sensing_trotter_step, psi2)
                psi2 = fori_loop(0, n_measure, apply_molecule_trotter_step, psi2)
                
                psi_cont = psi_cont.at[t0,:,:].set(psi3)
                psi_cont = psi_cont.at[t0+1,:,:].set(psi2)
                return psi_cont
            
            def evolve_molecule_noreverse(t0, psi_cont):
                psi2 = psi_cont[t0,:,:]
                psi2 = fori_loop(0, n_measure, apply_molecule_trotter_step, psi2)
                psi_cont = psi_cont.at[t0+1,:,:].set(psi2)
                return psi_cont
            
            if reverse_sense:
                psi_cont = fori_loop(0, N_t, evolve_molecule, psi_cont)
            else:
                psi_cont = fori_loop(0, N_t, evolve_molecule_noreverse, psi_cont)
            #psi_list.append(psi_cont[:-1,:,:])
            return psi_cont[:-1,:,:]

        return apply_evolution


        


    # def propagate_trotter(self, psi0_list, tau, dtau, t, dt, n_measure, reverse_sense=True):
    #     n = self.n
    #     ts = np.arange(0, t, n_measure*dt)
    #     N_t = len(ts)
    #     #N_t = int(n_evolve/n_measure)
    #     #print(N_t)
    #     #N_t = int(np.max(ts)/dt)
    #     N_tau = int(tau/dtau)
    #     #theta = self.theta

    #     psi_list = []
        
    #     for psi0 in psi0_list:
    #         psi = np.zeros([2**n, N_t], dtype='complex128')
    #         psi1 = psi0.copy()
            
    #         # sense
    #         for k in range(N_tau):
    #             psi1 = self.apply_sensing_trotter_step(psi1, dtau)

    #         # evolve
    #         psi2 = psi1.copy()
    #         counter = 0
    #         for l in range(int(t/dt)):
    #             print(l)
    #             psi2 = self.apply_molecule_trotter_step(psi2, dt)

    #             # reverse sense, then save the state
    #             if reverse_sense:
    #                 psi3 = psi2.copy()
    #                 print('time reversing...')
    #                 for k in range(N_tau):
    #                     psi3 = self.apply_sensing_trotter_step(psi3, -dtau)
    #                 if int(np.mod(l,n_measure))==0: 
    #                     print('measuring!')
    #                     psi[:, counter] = psi3
    #                     counter += 1
    #                     print()

    #             else:
    #                 if int(np.mod(l,n_measure))==0: 
    #                     psi[:,counter] = psi2
    #                     counter += 1
                
    #         psi_list.append(psi)

    #     return psi_list
            


