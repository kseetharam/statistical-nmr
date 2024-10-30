import numpy as np

from scipy.sparse import lil_matrix, diags, csc_matrix, triu



def norm(x):
    return np.sqrt(np.sum(np.abs(x)**2))

def int_to_array(states, L, p=1):
    # take a list of states in the "integer" rep (N,)
    # and convert it to a list of states in the spatial rep (N,L)

    toret = np.zeros((states.shape[0], (L//p))).astype('u1') # create states before hand

    for i in range(L//p):
        toret[:,i] = (states % (2**p))
        states = states // (2**p)
    return toret

def array_to_int(states):
    # take a list of states in spatial rep (N,L)
    # and convert it to the integer rep (N,)
    #assert(states.shape[1] == L)
    L = states.shape[1]
    toret = np.zeros((states.shape[0],)).astype('int')
    if L > 30:
        states = states.astype('int')
    for i in range(L):
        toret += states[:,i] * (2**i)
    return toret

class Lattice2D:

    def __init__(self, unit_cell, lattice_vector, lat_points, nrows, ncols, blockade_radius, PBC=True):

        self.unit_cell = unit_cell
        self.lat_points = lat_points
        self.unit_cell_size = unit_cell.shape[0]
        assert(unit_cell.shape[1] == 2)

        self.lattice_vector = lattice_vector

        self.blockade_radius = blockade_radius

        self.PBC = PBC

        self.nrows = nrows
        self.ncols = ncols

        self.L = nrows * ncols * self.unit_cell_size

        #self.constraints = self.build_constraints(blockade_radius, np.maximum(nrows, ncols)*2)

        self.make_positions()
        self.basis = self.build_basis()

    def make_positions(self):
        unit_cell = self.unit_cell
        lattice_vector = self.lattice_vector
        unit_cell_size = self.unit_cell_size
        lat_points = self.lat_points

        ncols = self.ncols
        nrows = self.nrows

        positions = []
        allpoints = []
        for row in range(nrows):
            for col in range(ncols):
                for uc in unit_cell:
                    v = row * lattice_vector[0] + col * lattice_vector[1] + uc
                    positions.append(v)
                for pt in lat_points:
                    v = row * lattice_vector[0] + col * lattice_vector[1] + pt
                    allpoints.append(v)
        positions = np.array(positions)
        allpoints = np.array(allpoints)
        self.positions = positions
        self.latpoints = allpoints

        # we need to calculate the correct distance between two sites.
        # the horizontal width is given by nrows * lat_vec[0][0]
        width = self.nrows * self.lattice_vector[0][0]
        # the vertical height is given by ncols * lat_vec[1][1]
        height = self.ncols * self.lattice_vector[1][1]

        if self.PBC:
            def distance(p1, p2):
                # let's try the four distances...
                # d1 = np.sqrt(np.sum(np.abs(p1 - p2)**2))
                # d2 = np.sqrt(np.sum((self.nrows * self.lattice_vector[0] - np.abs(p1 - p2))**2))
                # d3 = np.sqrt(np.sum((self.ncols * self.lattice_vector[1] - np.abs(p1 - p2))**2))
                # d4 = np.sqrt(np.sum(
                #     (self.nrows * self.lattice_vector[0] 
                #      + self.ncols * self.lattice_vector[1] 
                #      - np.abs(p1 - p2))**2)
                #      )
                d1 = np.sqrt(np.sum((p1 - p2)**2))
                d2 = np.sqrt(np.sum((self.nrows * self.lattice_vector[0] - (p1 - p2))**2))
                d3 = np.sqrt(np.sum((-self.nrows * self.lattice_vector[0] - (p1 - p2))**2))
                d4 = np.sqrt(np.sum((self.ncols * self.lattice_vector[1] - (p1 - p2))**2))
                d5 = np.sqrt(np.sum((-self.ncols * self.lattice_vector[1] - (p1 - p2))**2))
                d6 = np.sqrt(np.sum(
                    (self.nrows * self.lattice_vector[0] 
                     + self.ncols * self.lattice_vector[1] 
                     - (p1 - p2))**2)
                     )
                d7 = np.sqrt(np.sum(
                    (self.nrows * self.lattice_vector[0] 
                     - self.ncols * self.lattice_vector[1] 
                     - (p1 - p2))**2)
                     )
                d8 = np.sqrt(np.sum(
                    (-self.nrows * self.lattice_vector[0] 
                     + self.ncols * self.lattice_vector[1] 
                     - (p1 - p2))**2)
                     )
                d9 = np.sqrt(np.sum(
                    (-self.nrows * self.lattice_vector[0] 
                     - self.ncols * self.lattice_vector[1] 
                     - (p1 - p2))**2)
                     )
                return np.min([d1, d2, d3, d4,d5,d6,d7,d8,d9])#(np.sqrt(dx**2 + dy**2))
        else:
            def distance(p1, p2):
                dx = np.abs(p1[0] - p2[0])
                dy = np.abs(p1[1] - p2[1])
                return (np.sqrt(dx**2 + dy**2))
        
        L = self.L

        distances = np.zeros((L,L))
        for i in range(L):
            for j in range(i+1, L):
                distances[i,j] = distance(positions[i], positions[j])
                distances[j,i] = distance(positions[i], positions[j])
        self.distances = distances
    
    def build_basis(self):
        # easiest thing to do is just add spins one by one.
        PBC = self.PBC
        unit_cell = self.unit_cell
        unit_cell_size = self.unit_cell_size

        nrows = self.nrows
        ncols = self.ncols

        blockade_radius = self.blockade_radius

        basis = np.array([[0],[1]]).astype('int')
        for r in range(1, self.L):
            # now let's find all sites
            sites_blockade = np.argwhere(self.distances[:r,r] < blockade_radius)[:,0]
            states_noviolations   = np.argwhere(np.sum(basis[:,sites_blockade], axis=1) == 0)[:,0]

            newbasis = np.vstack(
                    [np.hstack([basis, np.zeros((len(basis),1)).astype('int') ]), 
                    np.hstack([basis[states_noviolations,:], np.ones((len(states_noviolations),1)).astype('int')])]
                )

            basis = newbasis
        
        #self.basis = basis
        #return basis
        return np.sort(array_to_int(basis))

    def build_operators(self):
        basis = self.basis
        L = self.L

        unit_cell_size = self.unit_cell_size

        basis_sites = int_to_array(basis, L)

        hx = lil_matrix((basis.shape[0], basis.shape[0])).astype('complex')

        hx_sitelist = []

        for site in range(0,L):
            b_site = basis_sites.copy()
            b_site[:,site] = 0
            b_site = array_to_int(b_site)

            inds = np.argwhere(b_site != basis)[:,0]

            inds_outputs = np.searchsorted(basis, b_site[inds])

            hx[inds, inds_outputs] += np.ones(len(inds))

            hxsite = lil_matrix((basis.shape[0], basis.shape[0])).astype('complex')
            hxsite[inds, inds_outputs] += np.ones(len(inds))

            hx_sitelist.append(hxsite + hxsite.T.conj())
        
        self.hx_sitelist = hx_sitelist

        self.hx = hx + hx.T.conj()

        ntot = np.sum(basis_sites, axis=1)

        self.ntot = diags(ntot)

        # add operator for each sublattice...
        n_sublat= []
        for i in range(unit_cell_size):
            n_lat = np.sum(basis_sites[:, i::unit_cell_size], axis=1)
            n_sublat.append(diags(n_lat))
        self.n_sublat = n_sublat

        n_sites = []
        for i in range(L):
            n_site = basis_sites[:,i]
            n_sites.append(diags(n_site))
        self.n_sites = n_sites

        # add operator 

    def site_index_to_coordinate(self, index):
        # label sites by (r, c, u) the row, column, and unit_cell
        nrows = self.nrows
        ncols = self.ncols
        nc = self.unit_cell_size

        corr3 = index % nc
        index = index // nc
        corr2 = index % ncols
        index = index // ncols
        corr1 = index % nrows
        index // nrows
        return (corr1, corr2, corr3)

    def site_coordinate_to_index(self, coor):
        # label sites by (r, c, u) the row, column, and unit_cell
        nrows = self.nrows
        ncols = self.ncols
        nc = self.unit_cell_size

        r = coor[0] % self.nrows
        c = coor[1] % self.ncols

        index = coor[2] + c * nc + r * (nc * ncols)

        return index

    def build_LR_hamiltonian(self):#, bl=self.blockade_radius):
        bl = self.blockade_radius
        basis = self.basis
        L = self.L
        positions = self.positions

        state_array = int_to_array(basis, L)

        distances = self.distances

        alpha=6 # for rydbergs!

        Jij = np.zeros((L,L))

        potential = np.zeros(basis.shape[0])
        sharp_potential = np.zeros(basis.shape[0])
        for i in range(L):
            for j in range(i+1, L):
                p1 = positions[i]
                p2 = positions[j]
                Jij[i,j] = (distances[i,j]/bl)**(-alpha)
                Jij[j,i] = (distances[i,j]/bl)**(-alpha)

                potential += Jij[i,j] * state_array[:,i] * state_array[:,j]
                if distances[i,j] < bl:
                    sharp_potential += state_array[:,i] * state_array[:,j]

        self.Jij = Jij
        self.potential = potential
        self.sharp_potential = sharp_potential


    def build_density_plaquettes(self, plaq_list):
        # assign plaquetes different signatures. 
        basis = self.basis
        L = self.L

        nrows = self.nrows
        ncols = self.ncols

        plaquette_operators = []

        state_array = int_to_array(basis, L)
        for r in range(nrows):
            for c in range(ncols):
                for plaq in plaq_list:
                    print(plaq)
                    indexes = []
                    for p in plaq:
                        row = (p[0]+r) % (nrows)
                        col = (p[1]+c) % (ncols)
                        index = self.site_coordinate_to_index((row, col, p[2]))
                        indexes.append(index)
                    print(indexes)
                    indexes = np.array(indexes)
                    vals = np.sum(state_array[:, indexes], axis=1)
                    plaquette_operators.append(vals.copy())
                    #parities = (-1)**vals
                    #plaquette_operators.append(parities)
        self.plaquette_operators = plaquette_operators
        return plaquette_operators
    
    def build_translation_isometry(self):
        assert self.PBC # translation only makes sense with PBC...

        basis = self.basis
        
        sym_list = []
        for tind in range(2):

            if tind == 0:
                tdim = self.nrows
            elif tind == 1:
                tdim = self.ncols
            else:
                println("we don't have three dims")
                assert False

            Top = np.zeros((self.L, self.L)).astype('int')
            for i in range(self.L):
                pos = self.positions[i] + self.lattice_vector[tind]
                partner_inds = np.argwhere(np.all(np.isclose(pos,self.positions), axis=1))[:,0]
                if len(partner_inds) != 1:
                    pos -= tdim * self.lattice_vector[tind]
                    partner_inds = np.argwhere(np.all(np.isclose(pos,self.positions), axis=1))[:,0]
                assert len(partner_inds) == 1
                Top[i,partner_inds[0]] = 1

            newbasis = array_to_int((Top @ int_to_array(self.basis, self.L).T).T)
            sym_perm = np.searchsorted(self.basis, newbasis)
            sym = lil_matrix((sym_perm.shape[0], sym_perm.shape[0]))
            sym[sym_perm, np.arange(sym_perm.shape[0])] = 1
            sym_list.append(csc_matrix(sym))
        self.sym_list = sym_list

        
        # let's construct the isometries from sym...
        # how should we build this isometry?

        nstates = basis.shape[0]

        temp = lil_matrix((nstates,nstates))
        temp[np.arange(nstates), np.arange(nstates)] = 1
        temp = csc_matrix(temp)

        sym_basis = lil_matrix((nstates,nstates))

        for a in range(self.nrows):
            for b in range(self.ncols):
                sym_basis = (sym_basis + temp)
                temp = sym_list[1] @ temp
            temp = sym_list[0] @ temp
        
        norms = (sym_basis.conj().T @ sym_basis).diagonal()
        sym_basis = sym_basis @ diags(1 / np.sqrt(norms))

        cols_to_keep = np.argwhere(triu(sym_basis, k=1).sum(axis=0) == 0)[:,1]

        sym_iso = sym_basis[:,cols_to_keep]
        self.sym_iso = sym_iso

    