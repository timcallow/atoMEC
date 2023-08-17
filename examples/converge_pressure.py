#!/usr/bin/env python3
"""
Automatic workflow for completely converging a calculation in atoMEC.

Saves the converged results and the convergence parameters as pickled dictionaries.
In this example, we focus on the pressure and some related quantities. One could
replace the function `calc_pressure_dict` with any function that returns a dictionary
of the quantities that should be converged.
"""

from atoMEC import Atom, models, config, mathtools, staticKS
from atoMEC.unitconv import ha_to_gpa
from atoMEC.postprocess import pressure

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import sys

# use all cores
config.numcores = -1

atom_species = "Al"  # helium
density = 2.7  # Wigner-Seitz radius of room-temp Be
temperature = 0.5  # temperature in eV
grid_type = "sqrt"

# initialize the atom object
atom = Atom(atom_species, density=density, temp=temperature, units_temp="eV")

# initialize the model
model = models.ISModel(atom, bc="bands", unbound="quantum")

r0 = 1e-8
ngrid_init = 500
nconv_init = 1e-3
kpts_init = 20
s0 = np.sqrt(r0)

# hard limits - calculation will likely fail before these are breached
nmax_lim = 200
lmax_lim = 300

# how much to increase by
nmax_diff = 5
lmax_diff = 5

# buffer for orbitals
# if increasing grid size or allowing more scf cycles requires more orbitals
norbs_buffer = 3
lorbs_buffer = 3

# initial orbitals
nmax = 5
lmax = 5

# calculations considered converged provided all occupations are below this
threshold = 1e-3
conv_hardlim = 2
conv_softlim = 5


def calc_pressure_dict(atom, model, nconv, ngrid, nkpts):
    out = model.CalcEnergy(
        nmax,
        lmax,
        grid_params={"ngrid": ngrid, "s0": s0},
        conv_params={"nconv": nconv, "vconv": 1e-1, "econv": 1e-2},
        band_params={"nkpts": nkpts},
        scf_params={"maxscf": 30},
        write_info=False,
        grid_type=grid_type,
    )

    orbs = out["orbitals"]
    pot = out["potential"]
    dens = out["density"]

    MIS = dens.MIS[0]
    n_R = dens.total[0, -1]
    V_R = pot.v_s[0, -1]
    xgrid = orbs._xgrid
    dn_dR = gradient(dens.total[0], xgrid)[-1]
    dV_dR = gradient(pot.v_s[0], xgrid)[-1]

    # finite difference pressure
    P_fd_A = pressure.finite_diff(
        atom,
        model,
        orbs,
        pot,
        method="A",
        conv_params={"nconv": nconv, "vconv": 1e-1, "econv": 1e-2},
        write_info=False,
    )

    # stress-tensor pressure
    P_st_tr = pressure.stress_tensor(atom, model, orbs, pot, only_rr=False)  # trace
    P_st_rr = pressure.stress_tensor(
        atom, model, orbs, pot, only_rr=True
    )  # rr comp only

    # compute virial pressure
    P_vir_corr_A = pressure.virial(
        atom, model, out["energy"], dens, orbs, pot, use_correction=True, method="A"
    )
    P_vir_nocorr_A = pressure.virial(
        atom, model, out["energy"], dens, orbs, pot, use_correction=False, method="A"
    )

    # compute ideal pressure
    chem_pot = mathtools.chem_pot(orbs)
    P_elec_id = pressure.ideal_electron(atom, chem_pot)

    E_free = out["energy"].F_tot

    P_dict = {
        "P_fd_A": P_fd_A * ha_to_gpa,
        "P_st_tr": P_st_tr * ha_to_gpa,
        "P_st_rr": P_st_rr * ha_to_gpa,
        "P_vir_corr_A": P_vir_corr_A * ha_to_gpa,
        "P_vir_nocorr_A": P_vir_nocorr_A * ha_to_gpa,
        "P_id": P_elec_id * ha_to_gpa,
        "MIS": MIS,
        "n_R": n_R,
        "V_R": V_R,
        "dn_dR": dn_dR,
        "dV_dR": dV_dR,
        "E_free": E_free,
        "chem_pot": chem_pot[0],
    }

    return P_dict


def compute_relative_differences(A, B):
    relative_differences = {}
    for key in A:
        if key in B:
            # Compute the relative error
            if A[key] != 0 or B[key] != 0:
                relative_error = (
                    abs(A[key] - B[key]) / (abs(A[key]) + abs(B[key])) * 100
                )
            else:
                relative_error = 0.0  # Avoid division by zero if both values are 0
            relative_differences[key] = round(
                relative_error, 2
            )  # Rounded to 2 decimal places
        else:
            print(f"Key {key} not found in second dictionary.")
            return None

    return relative_differences


def are_differences_below_threshold(
    relative_differences, threshold_hard, threshold_soft
):
    below_threshold_count = 0

    for value in relative_differences.values():
        if value < threshold_hard:
            below_threshold_count += 1

    if below_threshold_count == len(relative_differences):
        return True
    elif below_threshold_count == len(relative_differences) - 1:
        max_value = max(relative_differences.values())
        if max_value < threshold_soft:
            return True

    return False


def gradient(f, xgrid):
    return np.gradient(f, xgrid) / (2 * xgrid)


# P_dict = calc_pressure_dict(atom, model, 1e-4, 1000, 30)


# increase orbitals to convergence
while nmax <= nmax_lim:
    breakloop = False
    while lmax <= lmax_lim:
        try:
            out_test = model.CalcEnergy(
                nmax,
                lmax,
                grid_params={"ngrid": ngrid_init, "s0": s0},
                band_params={"nkpts": kpts_init},
                scf_params={"maxscf": 5},
                grid_type=grid_type,
                write_info=False,
            )
        except:
            sys.exit("Too many orbitals, aborting calculation")
        xgrid = out_test["orbitals"]._xgrid
        norbs_ok, lorbs_ok = staticKS.Orbitals(xgrid, "sqrt").check_orbs(
            out_test["orbitals"].occnums_w, threshold
        )
        if norbs_ok and lorbs_ok:
            breakloop = True
            break
        else:
            if not norbs_ok:
                nmax += nmax_diff
            if not lorbs_ok:
                lmax += lmax_diff
    if breakloop:
        break

nmax += norbs_buffer
lmax += lorbs_buffer

ngrid = ngrid_init

P_dict_inner = calc_pressure_dict(atom, model, nconv_init, ngrid_init, kpts_init)
nconv = nconv_init
while nconv >= 1e-7:
    nconv /= 10
    P_dict = calc_pressure_dict(atom, model, nconv, ngrid, kpts_init)
    diffs_inner = compute_relative_differences(P_dict, P_dict_inner)
    P_dict_inner = P_dict
    print(nconv, diffs_inner)

    if are_differences_below_threshold(diffs_inner, conv_hardlim, conv_softlim):
        print("break")
        break

ngrid = ngrid_init
while ngrid < 30000:
    ngrid = int(ngrid * 1.5)
    P_dict_outer = P_dict
    P_dict = calc_pressure_dict(atom, model, nconv, ngrid, kpts_init)
    diffs_outer = compute_relative_differences(P_dict, P_dict_outer)
    print(ngrid, diffs_outer)

    if are_differences_below_threshold(diffs_outer, conv_hardlim, conv_softlim):
        print("break")
        break

nkpts = kpts_init
while nkpts < 550:
    nkpts = int(nkpts * 1.5)
    P_dict_outer = P_dict
    P_dict = calc_pressure_dict(atom, model, nconv, ngrid, nkpts)
    diffs_outer = compute_relative_differences(P_dict, P_dict_outer)
    print(nkpts, diffs_outer)

    if are_differences_below_threshold(diffs_outer, conv_hardlim, conv_softlim):
        break

# ion pressure
P_dict["P_ion"] = pressure.ions_ideal(atom) * ha_to_gpa
P_dict["rho"] = density
P_dict["T"] = temperature
P_dict["Z"] = atom.at_chrg

# save the converged results
conv_dict = {"ngrid": ngrid, "nkpts": nkpts, "nconv": nconv, "nmax": nmax, "lmax": lmax}

with open("conv_dict.pkl", "wb") as f:
    pkl.dump(conv_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

with open("pressure_dict.pkl", "wb") as f:
    pkl.dump(P_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
