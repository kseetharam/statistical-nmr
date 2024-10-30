import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix
from scipy.linalg import eigh

# Building various sparse operators
from scipy.sparse import kron

I = csr_matrix([[1,0],[0,1]])
Z = csr_matrix([[1,0],[0,-1]])
X = csr_matrix([[0,1],[1,0]])
Y = csr_matrix([[0,-1j],[1j,0]])

# TENSOR PRODUCT CONSTRUCTIONS

def tensor(pauli_list):
    op = pauli_list[0]
    for i in range(1, len(pauli_list)):
        op = kron(op, pauli_list[i], format='csr')
    return op

def build_sparse_matrix(sites, paulis, L):
    oplist = [I] * L 
    for a in range(len(sites)):
        site = sites[a]
        if paulis[a] == 'x':
            oplist[site] = X
        elif paulis[a] == 'y':
            oplist[site] = Y
        elif paulis[a] == 'z':
            oplist[site] = Z
        else:
            print("dafuq!?")
    return tensor(oplist)

def build_local_operator(i, op, L):
    return build_sparse_matrix([i], [op], L)

def build_global_operator(op, L):
    global_op = csc_matrix((2**L, 2**L), dtype='complex')
    for i in range(L):
        global_op += build_local_operator(i, op, L)
    return global_op

def build_interaction(J, op1, op2):
    L = J.shape[0]
    Hint = csc_matrix((2**L, 2**L), dtype='complex')
    for i in range(L):
        for j in range(i+1,L):
            Hint += J[i,j] * build_sparse_matrix([i,j], [op1, op2], L)
    return Hint

#### SPIN SECTOR HELPERS ####

def int2array(states, L):
    # states is (N, )
    N = states.shape[0]
    toret = np.zeros((N, L)).astype('int')
    for i in range(L):
        toret[:, -i-1] = states % 2
        states = states // 2
    return toret

def array2int(states):
    # states is (N, L)
    N = states.shape[0]
    L = states.shape[1]

    toret = np.zeros(N).astype('int')
    for i in range(L):
        toret += 2**i * states[:, -i-1]
    return toret

def get_sec_inds(L):
    sec_inds_list = []
    basis = np.arange(2**L)
    ns = np.sum(int2array(basis,L),1) #(2**L, L)
    for a in range(L+1):
        inds = np.where(ns==a)[0]
        sec_inds_list.append(inds)
    return sec_inds_list

#### STATE PREPARATION ####

def get_spin_coherent_state(thetas, phis):
    L = len(thetas)
    assert len(phis) == L
    psi = np.array([np.cos(thetas[0]/2), np.exp(-1j*phis[0])*np.sin(thetas[0]/2)])
    for j in range(1, L):
        psi0 = np.array([np.cos(thetas[j]/2), np.exp(-1j*phis[j])*np.sin(thetas[j]/2)])
        psi = np.kron(psi, psi0)
    return psi

def get_haar_random_state(dimH, m):
    np.random.seed(m)
    psi0 = np.random.normal(0,1,dimH) + 1j * np.random.normal(0,1,dimH)
    return psi0/np.sqrt(np.dot(psi0.conj(), psi0))

##### BASIC LIN ALG ####

def inner(psi, phi):
    assert len(psi) == len(phi)
    return np.dot(psi.conj(), phi)

def norm(psi):
    return np.sqrt(inner(psi, psi))

def vec2op(rho_vec):
    dimH = int(np.sqrt(rho_vec.shape[0]))
    return np.reshape(rho_vec, [dimH, dimH])

def op2vec(rho_op):
    return np.ndarray.flatten(rho_op)

def compute_trace_distance(rho1_vec, rho2_vec):
    rho1 = vec2op(rho1_vec)
    rho2 = vec2op(rho2_vec)
    
    d12 = rho1-rho2
    return np.sum(np.linalg.evalsh(np.sqrt(np.d12.conj().T @ d12)))/2

def partial_trace(psi, nA, nB):
    # take the partial trace of a pure state psi
    #assert len(psi) == 2**nA * 2**nB
    Psi = np.reshape(psi, 2**nA, 2**nB)
    return Psi @ Psi.conj().T

##### EVOLVERS FOR PURE ED #####

def get_evals_sectors(H, L):
    evals_list = []
    V_list = []
    for inds in get_sec_inds(L):
        evals, V = eigh(H[inds[:,np.newaxis], inds[np.newaxis,:]].toarray())
        evals_list.append(evals)
        V_list.append(V)
    return evals_list, V_list

def propagate_sectors(psi0, ep_list, V_list, ts):
    # psi0_list is a list of tensors (dimH, M)
    psi0_list = [psi0[inds,:] for inds in get_sec_inds(n)]
    psi_list = []
    for psi0, evals, V in zip(psi0_list, ep_list, V_list):
        evolver = np.exp(-1j*np.tensordot(evals, ts, axes=0))
        if len(psi0.shape) == 1:
            psi = V.conj().T @ np.repeat(psi0[:,np.newaxis], len(ts), axis=1)
            psi = V@(evolver*psi)
        else:
            assert len(psi0.shape)==2 # (dimH, M)
            M = psi0.shape[1]
            evolver = np.repeat(evolver[:,:,np.newaxis], M, axis=2) # (dimH, N, M)
            np.repeat(psi0[:,np.newaxis,:], len(ts), axis=2) # (dimH, N, M)
            #propagate the state
            psi = V.conj().T @ np.repeat(psi0[:,np.newaxis], len(ts), axis=1)
            psi = V@(evolver*psi)


def propagate(psi0, evals, V, ts):
    # psi0 is (d, ) initial state
    # H is (d,d) hermitian
    # ts is (N,)
    # returns (d,N) array of propagated states

    N = len(ts)

    # exactly diagonalize
    evolver = np.exp(-1j*np.tensordot(evals,ts, axes=0)) # (d,N)
    
    #propagate the state
    psi = V.conj().T @ np.repeat(psi0[:,np.newaxis], N, axis=1)
    psi = V@(evolver*psi)
    
    return psi


def propagate_subsystem_sectors(psi0, evals_list, V_list, ts, nA, nB):
    # assume psi0 is (2**n, M)
    dimH = psi0.shape[0]
    M = psi0.shape[1]
    
    assert dimH == 2**nA * 2**nB
    Psi0 = np.reshape(psi0, [2**nA, 2**nB, M])
    
    Psi = np.zeros([2**nA, 2**nB, len(ts), M], dtype='complex128')
    for inds, evals, V in zip(get_sec_inds(nB), evals_list, V_list):
        #Psi0_sec = Psi0[:,inds,:,:].copy() # (2**nA, dB, len(ts), M)
        #evolver = np.exp(-1j * np.tensordot(evals, ts, axes=0)) # (dB, len(ts))
        #evolver = np.repeat(evolver[np.newaxis,:,:,np.newaxis], (2**nA, M), axis=(0,3))
        #Psi1_sec = evolver * np.transpose(np.tensordot(Psi0_sec, V.T, axes=((1), (0))), [0,3,1,2]) # (dA, dB, len(ts), M)
        #Psi[:,inds,:,:] = np.transpose(np.tensordot(Psi1_sec, V.conj(), axes=((1), (0))), [0,3,1,2])
        Psi[:,inds,:,:] = propagate_subsystem(Psi0[:,inds,:,:], evals, V, ts, nA, nB)
    
    return np.reshape(Psi, [dimH, len(ts), M])


def propagate_subsystem(Psi0, evalsB, VB, ts):
    # Psi0 is (dA, dB, M)
    dA = Psi0.shape[0]
    dB = Psi0.shape[1]
    
    Psi0 = np.repeat(Psi0[:,:,np.newaxis,:], len(ts), axis=2) # (2**nA, 2**nB, len(ts), M)
    evals = 
    evolver = np.exp(-1j * np.tensordot(evals, ts, axes=0)) # (dB, len(ts))
    evolver = np.repeat(evolver[np.newaxis,:,:,np.newaxis], (2**nA, M), axis=(0,3))
    Psi1_sec = evolver * np.transpose(np.tensordot(Psi0_sec, V.T, axes=((1), (0))), [0,3,1,2]) # (dA, dB, len(ts), M)
    
    return np.transpose(np.tensordot(Psi1_sec, V.conj(), axes=((1), (0))), [0,3,1,2])
    #evolver = np.exp(-1j*np.tensordot(evalsB, ts, axes=0))






