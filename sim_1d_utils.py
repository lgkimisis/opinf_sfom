import numpy as np
from scipy.interpolate import CubicSpline

def coupledsim(t, time_sample, yfull, rank_rom, ind_input_rom, dQin, Arom, Brom, Hrom, Krom, Aproj, Hproj, Bproj, Cproj, H2proj, Kproj): 
    """
    Simulation of the coupled fom/rom system
    """
    uOp = yfull[:rank_rom]
    usF = yfull[rank_rom:]
    invec = CubicSpline(time_sample, dQin, axis=1, bc_type='not-a-knot', extrapolate=None)

    inp = usF[ind_input_rom].T
    
    du_rom = Arom @ uOp + Brom @ inp + Hrom @ quad_vector(uOp) + Krom @ np.kron(inp, uOp)
    du_sfom = Aproj @ usF + Hproj @ np.kron(usF, usF) + Bproj @ uOp + Cproj @ invec(t) + H2proj @ np.kron(uOp, uOp) + Kproj @ np.kron(uOp, usF)
    
    return np.concatenate([du_rom, du_sfom])

def quad_vector(u):
    """
    Form quadratic vector with non-repeating entries
    """
    n=len(u)
    uq = [u[i] * u[i:] for i in range(n)]
    return np.hstack(np.array(uq, dtype=object))

def smoothen(U_rom, iop, iopS, FULLsol, rank_rom):
    """
    Smoothen the solution on the FOM/ROM interface
    """
    nt = np.shape(FULLsol)[1]
    tt = np.arange(iop[-1]-iopS[-1])    
    nsT = iop[-1]-iopS[-1]
    
    smooth = 1/2*np.matlib.repmat(1-np.cos(np.pi*tt/nsT), nt, 1).T #Continuous smoothing filter

    sol_over = np.multiply(smooth, FULLsol[rank_rom:rank_rom+nsT]) + np.multiply(1-smooth, U_rom[iopS[-1]+1:iop[-1]+1] @ FULLsol[:rank_rom])

    return np.concatenate([U_rom[iopS] @ FULLsol[:rank_rom], sol_over, FULLsol[rank_rom+nsT:]])