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

atom_species = "H"  # helium
density = 0.000983  # Wigner-Seitz radius of room-temp Be
temperature = 1.346  # temperature in eV
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
nmax_diff = 3
lmax_diff = 5

# buffer for orbitals
# if increasing grid size or allowing more scf cycles requires more orbitals
norbs_buffer = 2
lorbs_buffer = 4

# initial orbitals
nmax = 4
lmax = 4

# calculations considered converged provided all occupations are below this
threshold = 1e-3
conv_hardlim = 2
conv_softlim = 5


def calc_pressure_dict(atom, model, nconv, ngrid, nkpts):
    out = model.CalcEnergy(
        nmax,
        lmax,
        grid_params={"ngrid": ngrid, "s0": s0},
        conv_params={"nconv": 1e-6, "vconv": 1e-1, "econv": 1e-2},
        band_params={"nkpts": nkpts},
        scf_params={"maxscf": 30},
        write_info=False,
        grid_type=grid_type,
    )

    orbs = out["orbitals"]
    pot = out["potential"]
    dens = out["density"]
    nconv = out["conv_vals"]["drho"][0]
    econv = out["conv_vals"]["dE"]
    vconv = out["conv_vals"]["dpot"][0]
    conv_arr = [econv, nconv, vconv]

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
        method="B",
        conv_params={"nconv": 1e-7, "vconv": vconv / 10, "econv": econv / 10},
        write_info=True,
    )

    # stress-tensor pressure
    P_st_tr = pressure.stress_tensor(atom, model, orbs, pot, only_rr=False)  # trace
    P_st_rr = pressure.stress_tensor(
        atom, model, orbs, pot, only_rr=True
    )  # rr comp only

    # compute virial pressure
    P_vir_corr_A = pressure.virial(
        atom, model, out["energy"], dens, orbs, pot, use_correction=True, method="B"
    )
    P_vir_nocorr_A = pressure.virial(
        atom, model, out["energy"], dens, orbs, pot, use_correction=False, method="B"
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
    }

    full_dict = {
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

    return P_dict, full_dict, conv_arr


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


def are_differences_below_threshold(pressure_dict, full_dict, hard_thresh, soft_thresh):
    # Counters for values that are no more than 1 above the thresholds
    pressure_counter = 0
    full_counter = 0

    # Check the values in pressure_dict
    for value in pressure_dict.values():
        if value < hard_thresh:
            continue
        elif value <= hard_thresh * 1.2:
            pressure_counter += 1
        else:
            return False

    # If more than one value is no more than 1 above hard_thresh, return False
    if pressure_counter > 1:
        return False

    # Check the values in full_dict
    for value in full_dict.values():
        if value < soft_thresh:
            continue
        elif value <= soft_thresh * 1.2:
            full_counter += 1
        else:
            return False

    # If more than one value is no more than 1 above soft_thresh, return False
    if full_counter > 1:
        return False

    return True


def converge_pressure(scf_out, scf_out_old, atom, model):
    P_dict, full_dict = calc_pressure_dict_small(scf_out, atom, model)
    P_dict_old, full_dict_old = calc_pressure_dict_small(scf_out_old, atom, model)
    P_diffs = compute_relative_differences(P_dict, P_dict_old)
    full_diffs = compute_relative_differences(full_dict, full_dict_old)
    return are_differences_below_threshold(
        P_diffs, full_diffs, conv_hardlim, conv_softlim
    )


def calc_pressure_dict_small(scf_output, atom, model):
    orbs = scf_output["orbitals"]
    dens = scf_output["density"]
    pot = scf_output["potential"]
    energy = scf_output["energy"]
    # stress-tensor pressure
    P_st_tr = pressure.stress_tensor(atom, model, orbs, pot, only_rr=False)  # trace
    P_st_rr = pressure.stress_tensor(
        atom, model, orbs, pot, only_rr=True
    )  # rr comp only

    # compute virial pressure
    P_vir_corr_A = pressure.virial(
        atom, model, energy, dens, orbs, pot, use_correction=True, method="A"
    )
    P_vir_nocorr_A = pressure.virial(
        atom, model, energy, dens, orbs, pot, use_correction=False, method="A"
    )

    MIS = dens.MIS[0]
    n_R = dens.total[0, -1]
    V_R = pot.v_s[0, -1]
    xgrid = orbs._xgrid
    dn_dR = gradient(dens.total[0], xgrid)[-1]
    dV_dR = gradient(pot.v_s[0], xgrid)[-1]

    # compute ideal pressure
    chem_pot = mathtools.chem_pot(orbs)
    P_elec_id = pressure.ideal_electron(atom, chem_pot)

    E_free = energy.F_tot

    P_dict = {
        "P_st_tr": P_st_tr * ha_to_gpa,
        "P_st_rr": P_st_rr * ha_to_gpa,
        "P_vir_corr_A": P_vir_corr_A * ha_to_gpa,
        "P_vir_nocorr_A": P_vir_nocorr_A * ha_to_gpa,
        "P_id": P_elec_id * ha_to_gpa,
    }

    full_dict = {
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

    return P_dict, full_dict


def gradient(f, xgrid):
    return np.gradient(f, xgrid) / (2 * xgrid)


# P_dict = calc_pressure_dict(atom, model, 1e-4, 1000, 30)


nmax += norbs_buffer
lmax += lorbs_buffer

ngrid = ngrid_init
nconv = nconv_init


def converge_bands_occnums(threshold, nmax_in, lmax_in):
    nmax = nmax_in
    lmax = lmax_in
    while nmax <= nmax_lim:
        breakloop = False
        while lmax <= lmax_lim:
            # try:
            out_test = model.CalcEnergy(
                nmax,
                lmax,
                grid_params={"ngrid": ngrid_init, "s0": s0},
                band_params={"nkpts": kpts_init},
                scf_params={"maxscf": 5},
                grid_type=grid_type,
                write_info=False,
            )
            # except:
            #     sys.exit("Too many orbitals, aborting calculation")
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

        return nmax, lmax


P_dict, full_dict, nconv = calc_pressure_dict(atom, model, 1e-4, ngrid, kpts_init)

threshold = threshold_init


def converge_bands_pressure():
    while threshold > 1e-6:
        threshold /= 10
        nmax_out, lmax_out = converge_bands_occnums(threshold, nmax, lmax)
        nmax, lmax = nmax_out, lmax_out
        nmax += norbs_buffer
        lmax += lorbs_buffer
        P_dict_outer = P_dict
        full_dict_outer = full_dict
        P_dict, full_dict, nconv = calc_pressure_dict(
            atom, model, 1e-4, ngrid, kpts_init
        )
        P_diffs_outer = compute_relative_differences(P_dict, P_dict_outer)
        full_diffs_outer = compute_relative_differences(full_dict, full_dict_outer)
        print(
            "nmax final",
            nmax,
            "lmax_final",
            lmax,
            "P_diffs",
            P_diffs_outer,
            "P_dict",
            P_dict,
        )

        if are_differences_below_threshold(
            P_diffs_outer, full_diffs_outer, conv_hardlim, conv_softlim
        ):
            break


while ngrid < 20000:
    ngrid = int(ngrid * 1.5)
    P_dict_outer = P_dict
    full_dict_outer = full_dict
    P_dict, full_dict, nconv = calc_pressure_dict(atom, model, 1e-4, ngrid, kpts_init)
    P_diffs_outer = compute_relative_differences(P_dict, P_dict_outer)
    full_diffs_outer = compute_relative_differences(full_dict, full_dict_outer)
    print("ngrid final", ngrid, "P_diffs", P_diffs_outer, "P_dict", P_dict)

    if are_differences_below_threshold(
        P_diffs_outer, full_diffs_outer, conv_hardlim, conv_softlim
    ):
        break

nkpts = kpts_init

while nkpts < 400:
    nkpts = int(nkpts * 1.5)
    P_dict_outer = P_dict
    full_dict_outer = full_dict
    P_dict, full_dict, conv_arr = calc_pressure_dict(atom, model, 1e-4, ngrid, nkpts)
    P_diffs_outer = compute_relative_differences(P_dict, P_dict_outer)
    full_diffs_outer = compute_relative_differences(full_dict, full_dict_outer)
    print("nkpts final", nkpts, "P_diffs", P_diffs_outer, "P_dict", P_dict)

    if are_differences_below_threshold(
        P_diffs_outer, full_diffs_outer, conv_hardlim, conv_softlim
    ):
        break

# ion pressure
full_dict["P_ion"] = pressure.ions_ideal(atom) * ha_to_gpa
full_dict["rho"] = density
full_dict["T"] = temperature
full_dict["Z"] = atom.at_chrg

# save the converged results
conv_dict = {
    "ngrid": ngrid,
    "nkpts": nkpts,
    "conv_arr": conv_arr,
    "nmax": nmax,
    "lmax": lmax,
    "orbs_threshold": threshold,
}

print(full_dict)
print(conv_dict)

with open("conv_params.pkl", "wb") as f:
    pkl.dump(conv_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

with open("output.pkl", "wb") as f:
    pkl.dump(full_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
