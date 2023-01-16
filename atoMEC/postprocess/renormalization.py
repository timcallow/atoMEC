import copy

import numpy as np
from scipy.interpolate import interp1d
from scipy import signal

from atoMEC import numerov, arrays


def renorm_orbs(orbs, potential, RVS_new, ngrid_new, valence_orbs, renorm=False):

    # first we get the eigfuncs and the old xgrid
    eigfuncs = orbs.eigfuncs
    nbands, nspin, lmax, nmax, ngrid_o = np.shape(eigfuncs)
    eigvals = orbs.eigvals
    xgrid_o = orbs._xgrid
    x_R_o = xgrid_o[-1]
    ngrid_o = np.size(xgrid_o)
    v_s = potential.v_s

    # make a copy of the orbs object
    orbs_new = copy.deepcopy(orbs)

    # extend the potential and xgrid
    x_R = np.log(RVS_new)
    dx = xgrid_o[1] - xgrid_o[0]
    xgrid_add = np.arange(x_R_o + dx, x_R + dx, dx, dtype=np.float32)
    xgrid_tmp = np.concatenate((xgrid_o, xgrid_add))
    v_tmp = arrays.zeros32((1, len(xgrid_tmp)))
    v_tmp[:, : ngrid_o - 1] = v_s[:, :-1]
    v_tmp[:, ngrid_o - 1 :] = v_s[:, -2]

    # set up the final xgrid
    xgrid_new = np.linspace(xgrid_o[0], x_R, ngrid_new, dtype=np.float32)

    # interpolate the potential onto the new grid
    v_func = interp1d(xgrid_tmp, v_tmp)
    v_new = v_func(xgrid_new)

    # propagate numerov for eigenfunctions
    eigfuncs_new = numerov.calc_wfns_e_grid(xgrid_new, v_new, eigvals, norm_orbs=renorm)

    # deal with the valence orbitals
    for vpair in valence_orbs:
        eigfunc = eigfuncs[:, :, vpair[0], vpair[1]]
        interp_func = interp1d(xgrid_o, eigfunc, fill_value="extrapolate")
        eigfunc_new = interp_func(xgrid_new)
        eigfuncs_new[:, :, vpair[0], vpair[1]] = eigfunc_new

    # do some magic
    x_R_o_loc = np.where(xgrid_new <= x_R_o)[0][-1]
    eigfuncs_l = eigfuncs_new[:, :, :, :, :x_R_o_loc]
    eigfuncs_r = eigfuncs_new[:, :, :, :, x_R_o_loc:]

    for k in range(nbands):
        for sp in range(nspin):
            for l in range(lmax):
                for n in range(nmax):
                    minlocs = signal.argrelmin(eigfuncs_r[k, sp, l, n] ** 2, axis=-1)[0]
                    if len(minlocs) > 0:
                        minloc_0 = minlocs[0]
                        eigfuncs_r[k, sp, l, n, minloc_0:] = 0.0

    # set the orbital attributes
    eigfuncs_new = np.concatenate([eigfuncs_l, eigfuncs_r], axis=-1)
    # renormalize the orbitals
    psi_tmp = np.where(xgrid_new < x_R_o, eigfuncs_new, 0.0)
    psi_sq = np.exp(-xgrid_new) * psi_tmp**2  # convert from P_nl to X_nl and square
    integrand = 4.0 * np.pi * np.exp(3.0 * xgrid_new) * psi_sq
    norm = (np.trapz(integrand, x=xgrid_new)) ** (-0.5)
    eigfuncs_new = np.einsum("ijkl,ijklm->ijklm", norm, eigfuncs_new)

    orbs_new._eigfuncs = eigfuncs_new
    orbs_new._xgrid = xgrid_new

    return orbs_new
