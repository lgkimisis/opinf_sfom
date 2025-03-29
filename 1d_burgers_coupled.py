"""
non-intrusive model reduction for localized dynamics:
OpInf-sFOM coupling for periodic, 1D Burgers' equation.
"""
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from findiff import FinDiff
import opinf
from time import time
from data_1d_utils import fom_connectivity, data_domain_decomp, compute_connectivity, training_snapshots, gershgorin_circles
from sFOM_1d_utils import sfom, pod_sfom
from sim_1d_utils import quad_vector, coupledsim, smoothen

#%% Import data
Q = np.load('./1d_burgers_soln.npy') #State data
times = np.load('./1d_burgers_time.npy') #Simulation timesteps
x = np.load('./1d_burgers_x.npy') #Spatial coordinates
dtmin = np.min(times[1:]-times[:-1]) 
t_0 = times[0]
t_final = times[-1]
nt = np.size(times)
L = np.max(x)
x_interval = 1 #interval for spatial points
Nx1 = np.size(x)
x = x[::x_interval]
Nx = np.size(x)
n_variables = 1 #number of variables per DOF
n_dofs = Nx*n_variables #Total number of DOFs
q_inter = np.array([np.arange(0, Nx1, x_interval) + i * Nx for i in range(n_variables)]).ravel()
Q = Q[q_inter]

#Data time derivatives
d_dt = FinDiff(0, times, acc=6)
Qdot = d_dt(Q.T).T

#If the BCs are not periodic, then we need corresponding input vectors for OpInf, sFOM.
periodic_BC = True #periodic BCs flag
quad_model = True #Flag for quadratic coupled model

# Construct a mesh from that data
mesh = compute_connectivity(n_dofs, periodic_BC) 

#%% Parameters for non-intrusive modeling (OpInf-sFOM)

#Partitioning of the data
xcutmin = 4.95/10*L #partion position
xcutmax = 5.0/10*L #xcutmin-dx

#OpInf hyperparameters
rank_rom = 10
#regularization parameters
n_regular = 20 #number of tested regularization values
regular_opinf = np.logspace(-3,0, n_regular) #number of logarithmically scaled regularization values 
penalty_quad = 200 #factor for quadratic term penalization
penalty_inp = 1 #factor for input term penalization
diag_factor = 0.05 #factor for linear diagonal entries penalization (Gershgorin regularization) 
ninputs_fom_rom = 1 #number of overlapping ROM/FOM nodes

#sFOM hyperparameters
adjacency_order = 1 #Adjacency order for sFOM sparsity patterns
n_log_regularize = 20 #number of logarithmically scaled regularization values 
upper_bound_reg = -3 #maximum reg. value (logarithmic scale) 
lower_bound_reg = -8 #minimum reg. value (logarithmic scale) 
augment_pts = 5 #augmentation points per DOF (for uniform mesh)
quad_penalty_reg = 10 #factor for quadratic term penalization
diagonal_penalty_reg = 50 #factor for linear diagonal entries penalization (Gershgorin regularization) 

#We select here a training time as Ttot/2
ktrain = int(nt/2)

#%% Partition the data matrix Q
Q_opinf, Qdot_opinf, ind_rom_overlap, ind_rom_overlap_tot, ind_rom, ind_rom_tot, Q_sfom, Q_sfomdot, ind_fom, ind_fom_tot, nrom = data_domain_decomp(xcutmin, xcutmax, x, Q, Qdot, n_variables)

#Adding the overlap:
n_overlap = len(ind_rom_overlap)-len(ind_rom)
n_fom = len(ind_fom)
train_opinf = ktrain
#Partition data to training and testing
q0, Q_train, Qdot_train, Q_opinf_train, Qdot_opinf_train, Q_sfom_train, train_timesteps = training_snapshots(Q, Qdot, Q_opinf, Qdot_opinf, Q_sfom, ntrain=train_opinf, retained_qr=0)

#Plot the singular value decay of every section (training time)
U, s1, Vh = la.svd(Q_train[ind_rom_overlap]) #OpInf data
U, s2, Vh = la.svd(Q_train[ind_fom]) #sFOM data
Utot, stot, Vh = la.svd(Q_train) #complete domain
    
np.save('svals_opinf_burgers.npy', s1)
np.save('svals_sfom_burgers.npy', s2)
np.save('svals_global_burgers', stot)

input('Enter for OpInf')
#%% OpInf for the first half of the domain (let's try linear at first)

# Construct the low-dimensional basis.
U_rom, svdvals = opinf.basis.pod_basis(Q_opinf_train, r=rank_rom)
rom = opinf.models.ContinuousModel(operators=[opinf.operators.LinearOperator(), opinf.operators.InputOperator(), opinf.operators.QuadraticOperator(), opinf.operators.StateInputOperator()])

Q_rom_train = U_rom.T @ Q_opinf_train
Qdot_rom_train = U_rom.T @ Qdot_opinf_train

#BCs are periodic so we need two input terms: x=0, x=xcut
#if BCs are not periodic, we only need an input at x=xcut
if periodic_BC:
    for j in range(n_variables):
        ind_input_rom = np.array([np.concatenate([np.arange(0, ninputs_fom_rom)+n_overlap, np.arange(n_fom-ninputs_fom_rom, n_fom)])+i*Nx for i in range(n_variables)]).ravel()

else:
    for j in range(n_variables):
        ind_input_rom = np.concatenate([np.arange(ninputs_fom_rom)+n_overlap+i*n_fom for i in range(n_variables)]).ravel()

ind_input_rom = np.hstack(np.array(ind_input_rom))        
input_vector_rom = Q_sfom[ind_input_rom]
len_input = len(ind_input_rom)


rom_norm = np.zeros(n_regular)
rom_ls_error = np.zeros(n_regular)

train_ret = np.shape(Q_opinf_train)[1]
Ukron = np.zeros([int(rank_rom*(rank_rom+1)/2), train_ret])
Ubil = np.zeros([int(rank_rom*len_input), train_ret])

for tt in range(train_ret):
    Ukron[:, tt] = quad_vector(U_rom.T @ Q_opinf_train[:, tt])
    Ubil[:, tt] = np.kron(input_vector_rom[:, tt], U_rom.T @ Q_opinf_train[:, tt])

#Stability-promoting solution
Amat = np.hstack([Q_opinf_train.T @ U_rom, input_vector_rom[:, train_timesteps].T, Ukron.T, Ubil.T])
p_Amat = la.pinv(Amat)   
#Implement stability regularization here by changing the b vector
p = np.zeros([np.size(Amat,1), rank_rom])

for j in range(rank_rom): #The first term in the model is the linear one
    p[j, j] = 1  

#L_curve criterion
for i in range(len(regular_opinf)):
    qq = penalty_quad*np.ones(int(rank_rom*(rank_rom+1)/2))
            
    st_vec = diag_factor*regular_opinf[i]*p.T @ p_Amat
    
    reg_diag = regular_opinf[i]*np.concatenate([np.ones(int(rank_rom)), penalty_inp*np.ones(int(len_input)), qq, penalty_quad*np.ones(int(rank_rom*len_input))]) #

    solver = opinf.lstsq.TikhonovSolver(regularizer=np.diagflat(reg_diag),method="normal") # Select a least-squares solver with regularization.
    #Qdot_opinf_train-st_vec.T
    
    rom.fit(states=Q_rom_train, ddts=Qdot_rom_train-st_vec, inputs=input_vector_rom[:, train_timesteps], solver=solver)

    Arom = rom.A_.entries
    Brom = rom.B_.entries
    Hrom = rom.H_.entries
    Krom = rom.N_.entries

    rom_norm[i] = la.norm(np.hstack([Arom, Brom, Hrom, Krom]))  #, Hrom
    rom_ls_error[i] = np.sum(opinf.lstsq.TikhonovSolver.misfit(solver, np.hstack([Arom, Brom, Hrom, Krom]).T))

i_lcurve_opt = np.argwhere(rom_norm+rom_ls_error == np.min(rom_norm+rom_ls_error))

#We solve the LS problem with the optimal regularization
opt_reg = regular_opinf[i_lcurve_opt].item()
qq = penalty_quad*np.ones(int(rank_rom*(rank_rom+1)/2))

reg_diag = opt_reg*np.concatenate([np.ones(int(rank_rom)), penalty_inp*np.ones(int(len_input)), qq, penalty_quad*np.ones(int(rank_rom*len_input))]) #
st_opt_vec = diag_factor*opt_reg*p.T @ p_Amat
solver = opinf.lstsq.TikhonovSolver(regularizer=np.diagflat(reg_diag),method="normal")  # Select a least-squares solver with regularization.


rom.fit(states=Q_rom_train, ddts=Qdot_rom_train-st_opt_vec, inputs=input_vector_rom[:, train_timesteps], solver=solver)

Arom = rom.A_.entries
Brom = rom.B_.entries
Hrom = rom.H_.entries
Krom = rom.N_.entries

input('Enter for sFOM')
#%% We want to infer a sparse fom from the complete domain
imax = adjacency_order
imin = -imax
nsten = imax-imin+1 #length of numerical stencil
ninputs_rom_fom = imin+1 #Right boundary nodes
nfom=len(ind_fom)

nt = np.size(times) #timesteps

#Compute connectivity on the 1D domain
ind_fom_tot = np.concatenate([ind_fom+i*Nx for i in range(n_variables)]).ravel()
ind_rom_overlap_tot = np.concatenate([ind_rom_overlap+i*Nx for i in range(n_variables)]).ravel()
ind_rom_local_tot = np.concatenate([ind_rom+i*len(ind_rom_overlap) for i in range(n_variables)]).ravel() 

connectivity_matrix, connectivity_matrix_rom, connectivity_matrix_fom = fom_connectivity(mesh, ind_rom, ind_rom_overlap, ind_fom, adjacency_order, n_variables) 

t10 = time() 
#The structure is dudt=Au+Huxu+B 
Afom, Hfom, Bfom, Cfom = sfom(n_variables=n_variables,
                              Q_train=Q_train,
                              Qdot_train=Qdot_train,
                              connectivity_matrix=connectivity_matrix,
                              ind_fom_tot=ind_fom_tot,
                              ind_fom=ind_fom,
                              ind_rom_tot=ind_rom_tot,
                              ind_rom=ind_rom,
                              imin=imin,
                              imax=imax,
                              nlog_levels_regularizer=n_log_regularize,
                              upper_bound_regularizer=upper_bound_reg,
                              lower_bound_regularizer=lower_bound_reg,
                              quad_penalty_reg=quad_penalty_reg,
                              diagonal_penalty_reg=diagonal_penalty_reg,
                              augment_pts=augment_pts,
                              quad_model=quad_model,
                              periodic_BCs=periodic_BC)
print(f'time sfom = {time() - t10} sec')


#Projected rom
rank_pod_fom = len(ind_fom)*n_variables
t10 = time()
#The POD basis is irrespective of the QR decomposition earlier
Aproj, Bproj, Cproj, Hproj, Kproj, H2proj, Upr = pod_sfom(Afom, 
                                                          Bfom, 
                                                          Cfom, 
                                                          Hfom, 
                                                          Q_sfom_train, 
                                                          U_rom, 
                                                          connectivity_matrix, 
                                                          connectivity_matrix_fom, 
                                                          connectivity_matrix_rom, 
                                                          rank_rom, 
                                                          rank_pod_fom, 
                                                          ind_fom_tot, 
                                                          ind_rom_tot,
                                                          ind_rom_local_tot, 
                                                          n_dofs)

print(f'time pod_sfom = {time() - t10} sec')

#%% Save the matrices
#Save OpInf operators
np.save('OpInf_operators_burg.npy',np.hstack([Arom, Brom, Hrom, Krom]))

#Save sFOM operators
np.save('sFOM_operators_burg.npy',np.hstack([Aproj, Bproj, Hproj, Kproj, H2proj, Cproj]))

#%% Gershgorin circles plot
fig, ax = plt.subplots(figsize=(6,4))
ci=1
gershgorin_circles(Aproj, fig, ax, ci)
ci=5
gershgorin_circles(Arom, fig, ax, ci)
ax.set_xticks([-80, -40, 0, 40]) 
ax.set_xticklabels([-80, -40, 0, 40], fontsize=11)
ax.set_yticks([-40, -20, 0, 20, 40])
ax.set_yticklabels([-40, -20, 0, 20, 40], fontsize=11)
plt.xlabel('Real axis', size=16)
plt.ylabel('Imaginary axis', size=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()


#%% Simulate the inferred, coupled rom/fom model
input('Enter for OpInf-sFOM simulation')
qt0 = np.concatenate([U_rom.T @ q0[ind_rom_overlap_tot], Upr.T @ q0[ind_fom_tot]]) 
dQin = Qdot[Nx*np.arange(1,n_variables+1)-1]

coupled_solution = solve_ivp(fun=lambda t, yfull: coupledsim(t, times, yfull, rank_rom, ind_input_rom, dQin, Arom, Brom, Hrom, Krom, Aproj, Hproj, Bproj, Cproj, H2proj, Kproj),
                             t_span=[t_0, t_final],
                             y0=qt0,
                             t_eval=times,
                             method="BDF",
                             max_step=dtmin).y

#Reprojected soln (due to OpInf projection)
reprojected_soln = np.zeros([Nx*n_variables, np.size(coupled_solution,1)])
for i in range(n_variables):  
    U_rom_reproj = U_rom[ind_rom+i*len(ind_rom_overlap)] @ coupled_solution[:rank_rom]
    U_fom_reproj = coupled_solution[rank_rom+i*nfom:rank_rom+nfom+i*nfom]
    reprojected_soln[np.arange(Nx)+i*Nx,:] = np.concatenate([ U_rom_reproj, U_fom_reproj])                                                                                                                                                                  

# As a post-processing step, one can apply a smoothing filter:
# We showcase this for a single variable
reprojected_soln_smooth = np.zeros([Nx*n_variables, np.size(coupled_solution,1)])
for i in range(n_variables):   
    ind_fom_per_variable = np.concatenate([np.arange(rank_rom), np.arange(i*nfom, (i+1)*nfom)+rank_rom])   
    i_var_overlap = ind_rom_overlap+i*len(ind_rom_overlap)
    i_var_rom = ind_rom +i*len(ind_rom_overlap)
    reprojected_soln_smooth[np.arange(Nx)+i*Nx] = smoothen(U_rom, i_var_overlap, i_var_rom, coupled_solution[ind_fom_per_variable], rank_rom)

# Projection error 
Q_reprojected = np.concatenate([U_rom[ind_rom] @ U_rom.T @ Q_opinf, Q_sfom[:nfom]])

r_global_rom = rank_rom #50
# Global projection error
Qreglob = Utot[:Nx, :r_global_rom] @ Utot[:, :r_global_rom].T @ Q

np.save('./times_plot_data.npy',times)
np.save('./x_plot_data.npy',x)
np.save('./soln_plot_data.npy',Q)
np.save('./soln_smooth_plot_data.npy',reprojected_soln_smooth)
np.save('./soln_reglob_plot_data.npy', Qreglob)