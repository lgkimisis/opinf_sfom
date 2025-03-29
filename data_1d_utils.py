import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix 
import numpy.matlib as matl
from math import fabs
import random

def _find_simplexes(ind_fom, adjacency_order, mesh):
    """
    We find the simplexes that contain ind_fom in the mesh (fom nodes only)
    """
    kcore = [ind_fom]
    for _ in range(adjacency_order):
        for k in range(np.size(kcore)):
            k1 = np.where(mesh == kcore[k])[0]
            kcore = np.hstack([kcore, np.unique(mesh[k1])])
    # remove duplicate nodes
    return np.unique(kcore).astype(int)


def fom_connectivity(mesh, ind_rom, ind_rom_overlap, ind_fom, adjacency_order, n_variables):
    """
    Construct a connectivity matrix from the mesh (for the fom part)
    """
    n_fom = len(ind_fom)
    n_rom = len(ind_rom)
    n_rom_overlap = len(ind_rom_overlap)
    ntot = n_rom + n_fom
    #Initialize connectivity matrix for the fom part
    connectivity_matrix_sfom = np.zeros([n_fom, n_fom])
    connectivity_matrix_rom = np.zeros([n_fom, n_rom_overlap])
    connectivity_matrix = np.zeros([n_fom, ntot])
    
    for j in range(n_fom):
        k1glob = _find_simplexes(ind_fom=ind_fom[j],
                                 adjacency_order=adjacency_order,
                                 mesh=mesh)
        connectivity_matrix[j, k1glob] = 1    
        
        klocfom = np.argwhere(np.isin(ind_fom, k1glob)).ravel()   
        klocrom = np.argwhere(np.isin(ind_rom, k1glob)).ravel()
        
        connectivity_matrix_sfom[j, klocfom] = 1
        if len(klocrom) > 0:
            connectivity_matrix_rom[j, klocrom] = 1
            
    #Append connectivity matrix for all variables per DOF and make them sparse
    connectivity_matrix = csr_matrix(matl.repmat(connectivity_matrix, n_variables, n_variables))
    connectivity_matrix_sfom = csr_matrix(matl.repmat(connectivity_matrix_sfom, n_variables, n_variables))
    connectivity_matrix_rom = csr_matrix(matl.repmat(connectivity_matrix_rom, n_variables, n_variables))
    return connectivity_matrix, connectivity_matrix_rom, connectivity_matrix_sfom

def data_domain_decomp(xcutmin, xcutmax, xdata, Q, Qdot, n_variables):
    """
    Partition the data on the 1D domain to a FOM and a ROM subdomain
    """
    xmax = np.max(xdata)
    Nx = len(xdata)
    n_dofs = round(Q.shape[0]/n_variables)
    index_cut_min = round(xcutmin/xmax*n_dofs)
    index_cut_max = round(xcutmax/xmax*n_dofs)
    
    ind_rom = np.argwhere(xdata < xdata[index_cut_max]).flatten()
    ind_fom = np.argwhere(xdata >= xdata[index_cut_min]).flatten()

    #Find interface nodes (common entries in ind_rom, ind_fom) and corresponding indices
    op_ind = np.intersect1d(ind_rom, ind_fom, return_indices=True)[1]
    #Remove those indices from ind_rom
    ind_romS = np.delete(ind_rom, ind_rom[op_ind])
    
    #Break up the initial data matrix 
    ind_romt = np.array([ind_rom + i*Nx for i in range(n_variables)]).ravel()
    ind_romSt = np.array([ind_romS + i*Nx for i in range(n_variables)]).ravel()
    ind_fomt = np.array([ind_fom + i*Nx for i in range(n_variables)]).ravel()
    
    return Q[ind_romt], Qdot[ind_romt], ind_rom, ind_romt, ind_romS, ind_romSt, Q[ind_fomt], Qdot[ind_fomt], ind_fom, ind_fomt, len(ind_rom)
    
def compute_connectivity(n_fom, periodic_bc=True):    
    """
    Define the 1D mesh: Added a flag for periodic BCs
    """
    mesh = np.zeros([n_fom, 2])
    for i in range(n_fom-1):
        mesh[i, 0] = i
        mesh[i, 1] = i+1
    
    if periodic_bc:
        mesh[n_fom-1, 0] = n_fom-1
    return mesh

def training_snapshots(D, Ddt, D_rom, Ddt_rom, D_fom, ntrain, retained_qr=0):
    """
    Extract the training snapshot matrices from the original ones, also performing a QR decomposition.
    retained_qr = Percentage of retained timesteps
    """    
    if retained_qr > 0:
        # Perform QR decomposition on the training data.
        P = la.qr(D[:, :ntrain], pivoting=True)[2]
        # Choose the m timesteps with the most linearly independent state during training
        m = int(retained_qr * ntrain)
        # sort in chronological order. The last timestep should be included.
        # The realization of QR choses automatically the last column as the starting one.
        training_timesteps = np.sort(P[:m])
    else:
        training_timesteps = np.arange(ntrain)

    # The training data matrix will consist of these timesteps
    Dtr = D[:, training_timesteps]
    Ddttr = Ddt[:, training_timesteps]
    D_romtr = D_rom[:, training_timesteps]
    Ddt_romtr = Ddt_rom[:, training_timesteps]
    D_fomtr = D_fom[:, training_timesteps]
    q0 = D[:, 0]
    
    return q0, Dtr, Ddttr, D_romtr, Ddt_romtr, D_fomtr, training_timesteps


def gershgorin_circles(Arom, fig, ax, ci):
    """
    Plot Gershgorin circles of a linear operator
    """
    reg_circles = []
    for x in range(len(Arom)):
        piv = Arom[x][x]
        radius = sum(fabs(Arom[x][y]) for y in range(len(Arom)) if x != y)
        reg_circles.append([piv, radius])

    reg_eigs_rom = la.eig(Arom)[0]
    print(reg_eigs_rom)
    
    Xupper = 40
    Xlower = -100
    Ylimit = 45
    reg_index, reg_radi = zip(*reg_circles)
    print(reg_radi)
    ax.set_xlim((Xlower, Xupper))
    ax.set_ylim((-Ylimit, Ylimit))
    
    if ci>1:
        keyw1 = 'blue'
        keyw2 = 'o'
        keyw3 = [0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 0.35]
        keyw4 = r"$\mathbf{A}_{\mathrm{RR}}$"
    else:
        keyw1 = 'green'
        keyw2 = 'x'
        keyw3 = [0.15, 0.65, 0.20, 0.15]
        keyw4 = r"$\mathbf{A}_{\mathrm{FF}}$"
#Draw a sample of 10 eigs
    i_eigs = random.sample(range(np.shape(Arom)[0]), 5)
    print(i_eigs)
    for x in i_eigs:
        circ = plt.Circle((reg_eigs_rom[x], 0),
                          radius=reg_radi[x],
                          linestyle=':',
                          facecolor=(keyw3),
                          edgecolor=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 0.5)
                          )
        ax.add_artist(circ)
    
    plt.axvline(x=0, ymin=0, ymax=1, ls='--', color='#3c3c3c', lw=1, zorder=1)
    # plt.plot([Xlower, Xupper], [0, 0], '--', color='#3c3c3c', lw=1, zorder=1)
    plt.plot(reg_eigs_rom[i_eigs].real,
             reg_eigs_rom[i_eigs].imag,
             markersize=10, c=keyw1, alpha=0.7, linestyle='',
             fillstyle='none', marker=keyw2, markeredgecolor=keyw1, label = keyw4)
    plt.tight_layout()
