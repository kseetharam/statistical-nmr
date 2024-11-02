from spinED import *
import jax
from jax import jit
import jax.numpy as jnp
from jax import jit, jacfwd
from jax.lax import fori_loop
jax.config.update("jax_enable_x64", True)

def encode_params(JAB, JBB, Delta):
    nA = JAB.shape[0]
    nB = JAB.shape[1]
    assert JBB.shape[0]==nB
    assert JBB.shape[1]==nB
    assert len(Delta) == nB

    dim = int(nA*nB+nB*(nB+1)/2)
    theta = jnp.zeros([dim])
    #JAB = np.reshape(JAB, [nA, nB])
    theta = theta.at[:nA*nB].set(jnp.ravel(JAB))
    counter = 0
    for i in range(nB):
        for j in range(i+1,nB):
            #theta[nA*nB+counter] = JBB[i,j]
            theta = theta.at[nA*nB+counter].set(JBB[i,j])
            counter += 1
    theta = theta.at[nA*nB+counter:].set(Delta)
    #theta[nA*nB+counter:] = Delta
    return theta

# Helper Functions
def int2array(states, L):
    # states is (N, )
    N = states.shape[0]
    toret = jnp.zeros((N, L)).astype('int')
    for i in range(L):
        toret = toret.at[:,-i-1].set(states % 2)
        #toret[:, -i-1] = states % 2
        states = states // 2
    return toret

def array2int(states):
    # states is (N, L)
    N = states.shape[0]
    L = states.shape[1]

    toret = jnp.zeros(N).astype('int')
    for i in range(L):
        toret += 2**i * states[:, -i-1]
    return toret

def get_sec_inds(L):
    sec_inds_list = []
    basis = jnp.arange(2**L)
    ns = jnp.sum(int2array(basis,L),1) #(2**L, L)
    for a in range(L+1):
        inds = jnp.where(ns==a)[0]
        sec_inds_list.append(inds)
    return sec_inds_list

def build_hadamards(n):

    def tensor_power(A0, L):
        A = A0
        for j in range(1,L):
            A = jnp.kron(A, A0)
        return A

    Hx = jnp.array([[1,1],[1,-1]], dtype=complex)/jnp.sqrt(2)
    Hy = jnp.array([[1,-1j],[-1j,1]], dtype=complex)/jnp.sqrt(2)

    return tensor_power(Hx,n), tensor_power(Hy,n)

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

def get_padded_J(JAB):
    nA = JAB.shape[0]
    nB = JAB.shape[1]
    n = nA+nB
    indsA = jnp.arange(nA)
    indsB = jnp.arange(nA, n)
    JAB_pad = jnp.zeros([n, n])
    JAB_pad = JAB_pad.at[indsA[:,np.newaxis],indsB[np.newaxis,:]].set(JAB)
    JAB_pad = JAB_pad.at[indsB[:,np.newaxis],indsA[np.newaxis,:]].set(JAB.T)
    #JAB_pad[indsA[:,np.newaxis],indsB[np.newaxis,:]] = JAB
    #JAB_pad[indsB[:,np.newaxis],indsA[np.newaxis,:]] = JAB
    return JAB_pad

def partial_trace(psi,nA,nB):
    Psi = jnp.reshape(psi, [2**nA, 2**nB])
    return Psi @ Psi.conj().T

def get_trotter_energies(gAB, JAB, gB, JBB, Delta):
    nA = JAB.shape[0]
    nB = JBB.shape[0]
    n = nA+nB
    
    JAB_pad = get_padded_J(JAB)

    # Sensing Hamiltonian
    epAB = jnp.zeros([3, 2**n])
    z = 2*int2array(jnp.arange(2**n),n)-1 # (2**n, n)
    for mu in range(3):
        epAB = epAB.at[mu,:].set(gAB[mu]*jnp.diag(z @ JAB_pad @ z.T) / 4)

    # Molecule Hamiltonian
    epBB = jnp.zeros([3, 2**nB])
    z = 2*int2array(jnp.arange(2**nB), nB)-1 # (2**n, n)
    for mu in range(3):
        epBB = epBB.at[mu,:].set(gB[mu]*jnp.diag(z @ JBB @ z.T) / 4)
        if mu ==2:
            epBB = epBB.at[mu,:].set(z @ Delta / 2)

    return epAB, epBB

class SensorSim:

    def __init__(self,nA,nB):
        self.nA = nA
        self.nB = nB
        self.n = nA + nB
        self.secular = True

    def build_molecule_hamiltonian(self, gB, JBB, Delta):
        nB = self.nB
        assert JBB.shape == (nB,nB)
        assert Delta.shape[0] == nB
        
        HBB = csc_matrix((2**nB, 2**nB), dtype='complex')
        for i in range(nB):
            HBB += 0.5*Delta[i]*build_local_operator(i, 'z', nB)
        for mu, p in enumerate(['x','y','z']):
            HBB += 0.5*gB[mu]*build_interaction(JBB, p , p)

        return HBB

    def build_sensing_hamiltonian(self, gAB, JAB):
        nA = self.nA
        nB = self.nB
        n = self.n

        assert JAB.shape == (nA,nB)
        JAB_pad = get_padded_J(JAB)

        HAB = csc_matrix((2**n, 2**n), dtype='complex')
        for mu, p in enumerate(['x','y','z']):
            HAB += 0.5*gAB[mu]*build_interaction(JAB_pad, p , p)

        return HAB
    
    def build_exact_evolution(self, hyperparams):
        # in progress, low priority
        # computes the quench (H_AB, H_BB, \pm H_AB) on a list of pure states
        # returns list of evolved pure states, each element is of shape (dimH, len(ts))
        nA = self.nA
        nB = self.nB
        n = self.n

        t = hyperparams['t']
        n_measure = hyperparams['n_measure']
        dt = hyperparams['dt']
        tau = hyperparams['tau']
        beta = hyperparams['beta']

        ts = jnp.arange(0, t, n_measure*dt)
        N_t = len(ts)

        gAB = hyperparams['gAB']
        gB = hyperparams['gB']
        reverse_sense = hyperparams['reverse_sense']

        psi0_A = hyperparams["psi0_A"] # tensor of initial sampled states , (2**n, M)
        psi0 = jnp.array([jnp.kron(psi0_A, get_haar_random_state(2**nB, m)) for m in range(M)]).T # tensor of initial sampled states , (2**n, M)
        M = psi0.shape[1]

        def apply_evolution(theta, args):
            JAB, JBB, Delta = decode_params(theta, nA, nB)
            HAB = self.build_sensing_hamiltonian(gAB, JAB)
            HBB = self.build_molecule_hamiltonian(gB, JBB, Delta)

            # diagonalize the Hamiltonians
            evalsAB_list, VAB_list = get_evals_sectors(HAB, n)
            evalsB_list, VB_list = get_evals_sectors(HBB, nB)
            
            # imaginary time evolution
            psi0 = propagate_subsystem_sectors(psi0, evalsB_list, VB_list, -1j*beta)

            # sense
            psi0 = propagate_sectors(psi0, evalsAB_list, VAB_list, tau)

            # molecule evolve
            psi =  propagate_subsystem_sectors(psi0, evalsB_list, VB_list, ts)

            # reverse sense
            if reverse_sense:
                psi = propagate_sectors(psi, evalsAB_list, VAB_list, -tau)

            return psi # (N_t, 2**n, M)

        return apply_evolution
        # sense
            # psi = VAB @ (evolverAB * (VAB.conj().T @ psi0))

        # time evolve molecule 
        #Psi = np.reshape(psi, [2**nA, 2**nB])
        #Psi = Psi @ VBB.conj() # (2**nA, 2**nB)
        #Psi = np.repeat(Psi[:,:,np.newaxis], len(ts), axis=2) # (nA, nB, N)
        #Psi = Psi*evolverBB 
        #Psi = np.swapaxes(np.tensordot(Psi, VBB, axes=((1),(1))),1,2)# (2**nA, 2**nB, N)
        #psi = np.reshape(Psi, [2**n, len(ts)]) # (dimH, N)

        # reverse sense
        if reverse_sense:
            psi = VAB @ (np.conj(np.diag(evolverAB)) @ (VAB.conj().T @ psi))
        
        psi_list.append(psi)

        return psi_list


    def build_evolution(self, hyperparams):
        nA = self.nA
        nB = self.nB
        n = self.n

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

        gAB = hyperparams['gAB']
        gB = hyperparams['gB']
        reverse_sense = hyperparams['reverse_sense']

        psi0_A = hyperparams["psi0_A"] # single state, 2**nA
        M = hyperparams['M']
        psi0 = jnp.array([jnp.kron(psi0_A, get_haar_random_state(2**nB, m)) for m in range(M)]).T # tensor of initial sampled states , (2**n, M)
        #M = psi0.shape[1]

        Hx, Hy = build_hadamards(n)
        Hx_nB, Hy_nB = build_hadamards(nB)

        # any variational (input) parameters should be included in params.
        def apply_evolution(theta, args):

            # this is where the Ham params go! 
            JAB, JBB, Delta = decode_params(theta, nA, nB)
            epAB, epBB = get_trotter_energies(gAB, JAB, gB, JBB, Delta) #  (3, 2**n), (3, 2**nB)
            epAB = jnp.repeat(epAB[:,:,jnp.newaxis], M, axis=2) #  (3, 2**n, M)
            epBB = jnp.repeat(epBB[:,:,jnp.newaxis], M, axis=2) # (3, 2**nB, M)

            # time evolution phases
            DAB = jnp.exp(-1j * epAB * dtau / 2) # (3, 2**n, M)
            DB = jnp.exp(-1j * epBB * dt / 2) # (3, 2**nB, M)
            DB = jnp.repeat(DB[:,jnp.newaxis,:,:], 2**nA, axis=1) # (3, 2**nA, 2**nB, M)

            # imag time evolution attenuators
            DB_imag = jnp.exp(- epBB * dbeta / 2) # (3, 2**nB, M)
            DB_imag = jnp.repeat(DB_imag[:,jnp.newaxis,:,:], 2**nA, axis=1) # (3, 2**nA, 2**nB, M)

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

            def right_multiply(A, B):
                return jnp.swapaxes(jnp.tensordot(A, B, axes=((1),(0))),1,2)
            
            def apply_molecule_trotter_step_imag(tau0, psi):
                Psi = jnp.reshape(psi, [2**nA, 2**nB, M])
                Psi = DB_imag[2,:,:,:] * Psi # ZZ
                Psi = right_multiply(right_multiply(Psi, Hx_nB.T) * DB_imag[0,:,:,:], Hx_nB.conj()) # XX
                Psi = right_multiply(right_multiply(Psi, Hy_nB.T) * DB_imag[1,:,:,:]**2, Hy_nB.conj()) # YY
                Psi = right_multiply(right_multiply(Psi, Hx_nB.T) * DB_imag[0,:,:,:], Hx_nB.conj()) # XX
                Psi = DB_imag[2,:,:,:] * Psi # ZZ
                psi = jnp.reshape(Psi, [2**n, M])
                return psi 
            
            def apply_molecule_trotter_step(t0, psi): # psi is always (2**n, M)
                Psi = jnp.reshape(psi, [2**nA, 2**nB, M])
                Psi = DB[2,:,:,:] * Psi # ZZ
                Psi = right_multiply(right_multiply(Psi, Hx_nB.T) * DB[0,:,:,:], Hx_nB.conj()) # XX
                Psi = right_multiply(right_multiply(Psi, Hy_nB.T) * DB[1,:,:,:]**2, Hy_nB.conj()) # YY
                Psi = right_multiply(right_multiply(Psi, Hx_nB.T) * DB[0,:,:,:], Hx_nB.conj()) # XX
                Psi = DB[2,:,:,:] * Psi # ZZ
                psi = jnp.reshape(Psi, [2**n, M])
                return psi 
            
            psi1 = psi0.copy()
            
            # prepare thermal initial state
            # imag time evolve
            psi1 = fori_loop(0, N_beta, apply_molecule_trotter_step_imag, psi1)
            psi1 = psi1/jnp.sqrt(jnp.trace(psi1.conj().T @ psi1)) # divide by estimator of partition function
            # note that now the columns of psi1 are not normalized, yet averaging uniformly over these states yields the correct density matrix

            # sense
            psi1 = fori_loop(0, N_tau, apply_sensing_trotter_step, psi1)

            # container for states to save
            psi_cont = jnp.zeros((N_t+1, 2**n, M), dtype='complex128')
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

            return psi_cont[:-1,:,:]

        return jit(apply_evolution)


    def build_reduce(self):
        nA = self.nA
        nB = self.nB
        #nA = hyperparams['nA']
        #nB = hyperparams['nB']

        # This function takes as input a density matrix,
        # with shape (T,  ndim, M)
        def reduce(psi):
            psi = jnp.transpose(psi, [0,2,1]) # reshape this to (T, M, ndim)
            psi = jnp.reshape(psi, (psi.shape[0], psi.shape[1], 2**nA, 2**nB)) # expand ndim => (nA,nB)

            # then compute an outer product, returning (T, M, nA, nA')
            psi2 = jnp.transpose(psi, [0,1,3,2]) 
            rhos = jnp.sum(psi[:,:,:,:,jnp.newaxis] * psi2[:,:,jnp.newaxis,:,:].conj(), axis=3) 
            # finally, we will sum over M, returning (T, nA, nA')
            rho = jnp.mean(rhos, axis=1) 
            return rho
            #return jnp.stack([jnp.real(rho), jnp.imag(rho)])
        return jit(reduce)
    

    def simulate_sensor_state(self, hyperparams):
        reduce_func = self.build_reduce()
        sensing_evol = self.build_evolution(hyperparams)
        F = lambda theta, args : reduce_func(sensing_evol(theta, args))
        gradF = jit(jacfwd(F))
        return F, gradF
        #gradF(theta, args).shape
        #rhos = F(theta, args)
        #drhos = gradF(theta, args)