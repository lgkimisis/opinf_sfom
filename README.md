# opinf_sfom
Codes for the "Non-intrusive reduced-order modeling for dynamical systems with spatially localized features" by L. Gkimisis, N. Aretz, M. Tezzele, T. Richter, P. Benner, K. E. Willcox (https://arxiv.org/abs/2501.04400).

%--------------------------------------------------------------------------------------%

This repository includes the codes for predictions via OpInf-sFOM for the 1D Burgers' equation testcase in Section 5.1 of the above paper.

%--------------------------------------------------------------------------------------%

Brief repository files description:

%----- Simulation data -----%

1d_burgers_soln.npy: Simulation snapshot data

1d_burgers_x.npy: Spatial coordinate data

1d_burgers_time.npy: Temporal discretization data

%----- OpInf-sFOM inference -----%

environment.yml: Conda file with environment specifications.

1d_burgers_coupled.py: Main file that applies OpInf-sFOM on the simulation data.

data_1d_utils.py: Subroutines for data processing e.g. training/testing splitting, domain partitioning.

sFOM_1d_utils.py: Subroutines for sparse FOM inference.

sim_1d_utils.py: Subroutines for coupled OpInf-sFOM simulation.

%----- Results Visualization -----%

plot_burgers_solutions.py: Plot simulation data, OpInf-sFOM predictions and reprojected data.

plot_sval_decay_burgers.py: Plot the singular values of the OpInf, sFOM subdomain and complete domain snapshot matrices.
