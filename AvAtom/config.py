"""
Configuration file to store global parameters
"""

# physical parameters

# model parameters
spinpol = False  # spin-polarized functional
xfunc_id = "lda_x"  # exchange functional (libxc ref)
cfunc_id = "lda_c_pw"  # correlation functional (libxc ref)
bc = "dirichlet"  # boundary condition: Dirchlet means X(r_s)=0, Neumann means [dX(r)/dr]_(r=r_s)=0
unbound = "ideal"  # treatment for unbound electrons


# numerical grid for static calculations
grid_params = {"ngrid": 1000, "x0": -12}
# convergence parameters for static calculations
conv_params = {"econv": 1e-5, "nconv": 1e-3, "vconv": 1e-3}
# scf parameters
scf_params = {"maxscf": 30, "mixfrac": 0.5}

# variables
nmax = 6  # maximum quantum number n of eigenfunctions
lmax = 5  # maximum quantum number l of eigenfunctions
