import numpy as np
import scipy.linalg as la
from scipy.sparse import lil_matrix
import numpy.random as rand

def quadvec_position(ind_location_fom, n_dofs):
    """
    Find the entry positions in the quadratic vector
    """
    entry_pos = []
    for jF, ind1 in enumerate(ind_location_fom):
        for kF in ind_location_fom[jF:]:
            # Storing these indices is: 0->(n-1), n->(2n-2), ...
            if ind1 <= kF:
                zstart = (2*n_dofs-ind1+1) * ind1/2 #unique DOF counting (global)
                entry_pos.append(zstart + (kF-ind1))
            else:
                zstart = (2*n_dofs-kF+1) * kF/2 #unique DOF counting (global)
                entry_pos.append(zstart + (ind1-kF))
    return np.hstack(np.array(entry_pos, dtype=object).astype(int))

def posvec_bilinear(ind_location_1, ind_location_2, n_dofs):
    """
    Find positions of the bilinear terms (sFOM-OpInf DOFs interaction)
    """
    #Find the entry positions in the quadratic vector
    entry_pos = []
    for ind1 in ind_location_1: #find the index positions
        for kF in ind_location_2:
            #Storing these indices is: 0->(n-1), n->(2n-2), ...
            if ind1 <= kF:
                zstart = (2*n_dofs-ind1+1)*ind1/2 #unique DOF counting (global)
                entry_pos.append(zstart + (kF-ind1))
            else:
                zstart = (2*n_dofs-kF+1)*kF/2 #unique DOF counting (global)
                entry_pos.append(zstart + (ind1-kF))
    return np.hstack(np.array(entry_pos, dtype=object).astype(int))

def sfom(n_variables,
         Q_train,
         Qdot_train,
         connectivity_matrix,
         ind_fom_tot,
         ind_fom,
         ind_rom_tot,
         ind_rom,
         imin,
         imax,
         nlog_levels_regularizer,
         upper_bound_regularizer,
         lower_bound_regularizer,
         quad_penalty_reg, 
         diagonal_penalty_reg,
         augment_pts,
         quad_model,
         periodic_BCs):
    """
    Compute the inferred sparse FOM
    """
    n_fom = len(ind_fom)
    n_rom = len(ind_rom)
    n_fom_tot = len(ind_fom_tot)
    n_rom_tot = len(ind_rom_tot)
    Nx = n_fom + n_rom

    ntrain = np.shape(Q_train)[1]

    L2reg_list = np.logspace(lower_bound_regularizer, upper_bound_regularizer, nlog_levels_regularizer)
    ind_internal_fom = ind_fom[imax:imin] #exclude boundary nodes
    
    #Complete vector of internal nodes
    n_internal_fom = np.size(ind_internal_fom)
    nDOFs = Nx*n_variables
    ivar = np.arange(n_variables)+1
    #Initialize matrices
    Afom = lil_matrix((n_fom_tot, n_fom_tot)) #linear matrix
    Bfom = np.zeros([n_fom_tot, n_rom_tot]) #Input from OpInf
    Cfom = np.zeros([n_fom_tot, n_variables]) #Input at x=L
    Hfom = lil_matrix((n_fom_tot, int(nDOFs*(nDOFs+1)/2))) #quadratic term

    # L-S problem for OpInf coupling
    for j in range(n_fom_tot):
        i = ind_fom_tot[j]
        
        # Input vector for all variables
        if np.isin(j, ivar*n_fom-1).any() and not periodic_BCs:
            ivv = int(np.floor(j/n_fom))
            Cfom[j, ivv] = 1
        #Internal DOFs in mesh
        else:
            #we classify into: internal-not walls, walls
            #For the internal, we treat the fom points with rom neighbours seperately
            ind_location_fom = np.argwhere(connectivity_matrix[j] > 0)[:, 1] #global indexing   
            #sort them like left/central/right
            ind_mods = ind_location_fom + np.floor_divide(np.max(ind_location_fom)-ind_location_fom,nDOFs-2)*nDOFs
            ind_mods_s = np.argsort(ind_mods)
            ind_location_fom = ind_location_fom[ind_mods_s]
            n_adjacent_fom = ind_location_fom.shape[0]
            #Position of self index in vector
            ind_self_position = np.argwhere(ind_location_fom == i)[:, 0]
            #initialization of LS terms
            #linear and quadratic terms
            if quad_model:
                alph1 = quad_train_vector(ntrain, ind_location_fom, Q_train)  #quadratic vector
            else:
                alph1 = Q_train[ind_location_fom] #linear vector
            beta1 = Qdot_train[i]
            
            if np.isin(i, ind_internal_fom): #For the internal DOFs and uniform meshes we augment the data of the LS problem
                irand = rand.randint(0, n_internal_fom, augment_pts)
                jrand = ind_internal_fom[irand]
                #initialize augment_ptsmented data terms
                alph_add = alph1
                beta_add = beta1
                #stack the data
                for kk in jrand: #n_rom is first in the indexing
                    iconnectivity_matrix = np.argwhere(connectivity_matrix[kk-n_rom] > 0)[:, 1] #rom entries on the left of fom entries
                    ind_mods = iconnectivity_matrix + np.floor_divide(np.max(iconnectivity_matrix)-iconnectivity_matrix,nDOFs-2)*nDOFs
                    ind_mods_s = np.argsort(ind_mods)
                    iconnectivity_matrix = iconnectivity_matrix[ind_mods_s]
                    if quad_model:
                        alphA = quad_train_vector(ntrain, iconnectivity_matrix, Q_train)  #quadratic system vector
                    else:
                        alphA = Q_train[iconnectivity_matrix]
                    alph_add = np.hstack([alph_add, alphA])
                    beta_add = np.hstack([beta_add, Qdot_train[kk]])
                
                alph1 = alph_add
                beta1 = beta_add
            
            n_dofs = len(alph1)
            #Initialize solution and error vectors
            solns_list = np.zeros([n_dofs, nlog_levels_regularizer])
            stability_residual = np.zeros(nlog_levels_regularizer)
            soln_error = np.zeros(nlog_levels_regularizer)
            soln_norm = np.zeros(nlog_levels_regularizer)

            for jreg in range(nlog_levels_regularizer):
                
                # We implement stability regularization
                penalization_vector = np.zeros([n_dofs])
                penalization_vector[ind_self_position] = diagonal_penalty_reg*L2reg_list[jreg]
                
                L2diag = np.concatenate([L2reg_list[jreg]*np.ones(n_adjacent_fom), quad_penalty_reg*L2reg_list[jreg]*np.ones(n_dofs-n_adjacent_fom)])
                L2term=np.diag(L2diag) #Diagonal matrix for regularization
                
                st_add = penalization_vector #Penalizes the value of the diagonal matrix elements

                beta_st = alph1 @ beta1 - st_add #st_add penalizes the diagonal element value
                
                alpha_matrix = alph1 @ alph1.T + L2term 
                
                # solve with L_2 regularization
                ls_solve = la.solve(alpha_matrix, beta_st, assume_a='sym')
        
                solns_list[:, jreg] = ls_solve
                linear_part_ls_soln = ls_solve[:n_adjacent_fom]
                #criterion for stability in continuous time
                soln_norm[jreg] = la.norm(ls_solve) # solution norm
                stability_residual[jreg] = np.sum(np.abs(linear_part_ls_soln))-(np.abs(linear_part_ls_soln[ind_self_position])-linear_part_ls_soln[ind_self_position])
                soln_error[jreg] = la.norm(beta1.T-alph1.T @ ls_solve) #computed only on the self node data!
        
            soln_index_stability = np.argwhere(stability_residual <= max(0,np.min(stability_residual))) #stable schemes
            max_residual = np.max(soln_error)
            max_norm = np.max(soln_norm)
            ind_L_curve = np.argmin(np.square(soln_error[soln_index_stability]/max_residual) + np.square(soln_norm[soln_index_stability]/max_norm)) 
            ind_stable = soln_index_stability[ind_L_curve]  #selected solution      
                                    
            if np.isin(ind_location_fom, ind_rom_tot).any():
                #find fom indices
                index_fom_pos = np.argwhere(np.isin(ind_location_fom, ind_fom_tot))
                ind_index_fom_pos = ind_location_fom[index_fom_pos] #global indexing
                index_fom_location = np.argwhere(np.isin(ind_fom_tot, ind_index_fom_pos)).ravel() #local fom indexing
                Afom[j, index_fom_location] = solns_list[index_fom_pos, ind_stable].T

                #We split the quadratic matrix when we do the projection. Here it is one matrix for both rom/fom
                
                #find rom indices
                index_rom = np.argwhere(np.isin(ind_location_fom, ind_rom_tot))
                index_rom_pos = ind_location_fom[index_rom] #global indexing
                index_rom_location = np.argwhere(np.isin(ind_rom_tot, index_rom_pos)).ravel() #local rom indexing
                #Transfer those indices to the global rom indices
                Bfom[j, index_rom_location] = solns_list[index_rom, ind_stable].T

            else:
                index_fom_location = np.argwhere(np.isin(ind_fom_tot, ind_location_fom)).ravel() #local fom indexing
                #From the solns_list, we use the smallest Gershgorin radius solution
                Afom[j, index_fom_location] = solns_list[:n_adjacent_fom, ind_stable].T
            
            if quad_model:
                # positions of quadratic term elements (in global indexing)
                pquad = quadvec_position(ind_location_fom, nDOFs)
                Hfom[j, pquad] = solns_list[n_adjacent_fom:, ind_stable].T

    return Afom, Hfom, Bfom, Cfom


def quad_train_vector(ntrain, ind_location_fom, Q_train):
    """
    Assemble the coefs for the quadratic terms
    """
    quadratic_vector = []

    for kk in range(ntrain):#timesteps (corresponds to u(2:M))
        longa = []
        #We assemble unique combinations of the quadratic entries
        for jj in range(len(ind_location_fom)): #only unique combinations
            long1 = Q_train[ind_location_fom[jj], kk] * Q_train[ind_location_fom[jj:], kk]
            longa.append(long1)
        longa = np.hstack(np.array(longa, dtype=object))
        quadratic_vector.append(longa)
        
    quadratic_vector = np.vstack(np.array(quadratic_vector, dtype=object))
    return np.hstack([Q_train[ind_location_fom].T, quadratic_vector]).astype(None).T

def pod_sfom(Afom,
             Bfom,
             Cfom,
             Hfom,
             Q_fom_train,
             U_rom,
             connectivity_matrix,
             connectivity_matrix_sfom,
             connectivity_matrix_rom,
             rrom,
             r_pod_fom,
             ind_fom_tot,
             ind_rom_tot,
             ind_rom_local_tot,
             nDOFs):
    """
    Formulate the sFOM matrices and potentially perform POD projection
    """
    n_fom_tot = len(ind_fom_tot)
    # fom or rom
    if r_pod_fom == Afom.shape[0]:
        Upr = np.eye(r_pod_fom) 
    else:
        Uf = la.svd(Q_fom_train, full_matrices=False)[0]
        Upr = Uf[:, :r_pod_fom]

    Aproj = Upr.T @ Afom @ Upr
    Bproj = Upr.T @ Bfom @ U_rom[ind_rom_local_tot]
    Cproj = Upr.T @ Cfom
    
    #Hfom needs special treatment for the projection (split to entries from OpInf or sFOM)
    Hint = np.zeros([n_fom_tot, r_pod_fom**2])
    #We split into quadratic and bilinear terms (rom-fom interaction)
    Kint = np.zeros([n_fom_tot, r_pod_fom*rrom])
    #There is also a quadratic rom term
    H2int = np.zeros([n_fom_tot, rrom**2])
    
    for i in range(n_fom_tot):
        ind_glob_dofs = np.argwhere(connectivity_matrix[i]>0)[:, 1] #global indexing for neighbouring DOFs 
        
        #split into fom and rom nodes (if any) 
        gindex_fom_pos = np.argwhere(np.isin(ind_glob_dofs, ind_fom_tot)).flatten()
        gindex_rom = np.argwhere(np.isin(ind_glob_dofs, ind_rom_tot)).flatten()

        #Local indexing fom/rom accordingly
        ind_location_rom = np.argwhere(connectivity_matrix_rom[i] > 0)[:, 1] #local connectivity matrix (OpInf)
        ind_location_fom = np.argwhere(connectivity_matrix_sfom[i] > 0)[:, 1] #local connectivity matrix (sfom)
        quadratic_vector = [np.kron(Upr[ind_location_fom[j]], Upr[ind_location_fom[j:]]) for j in range(ind_location_fom.shape[0])]
        
        #global indexing
        ind_globalfom = ind_glob_dofs[gindex_fom_pos]
        pos = quadvec_position(ind_globalfom, nDOFs)
        quadratic_vector = np.concatenate(quadratic_vector) 
        Hint[i] = Hfom[i, pos] @ quadratic_vector
        
        #bilinear term and quadratic from rom side
        if ind_location_rom.shape[0] > 0:
            bilinear_vector = []
            quadratic_rom_vector = []
            for j in range(ind_location_rom.shape[0]):
                #ind_location_rom are smaller values than ind_location_fom (fom entries are to the right of rom entries)
                bilinear_vector.append(np.kron(U_rom[ind_location_rom[j]], Upr[ind_location_fom]))
                quadratic_rom_vector.append(np.kron(U_rom[ind_location_rom[j]], U_rom[ind_location_rom[j:]]))
            #Global indexing for storage
            ind_globalrom = ind_glob_dofs[gindex_rom]
            pos_bilinear_rom = posvec_bilinear(ind_globalrom, ind_globalfom, nDOFs)
            pos_quad_rom = quadvec_position(ind_globalrom, nDOFs)
            
            Kint[i] = Hfom[i, pos_bilinear_rom] @ np.concatenate(bilinear_vector)
            H2int[i] = Hfom[i, pos_quad_rom] @ np.concatenate(quadratic_rom_vector)
    
    Hproj = Upr.T @ Hint   
    Kproj = Upr.T @ Kint
    H2proj = Upr.T @ H2int
    return Aproj, Bproj, Cproj, Hproj, Kproj, H2proj, Upr
