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


def calc_pressure_dict(
    atom, model, nmax, lmax, nconv=1e-4, ngrid=1000, nkpts=30, s0=1e-4, maxscf=30
):
    out = model.CalcEnergy(
        nmax,
        lmax,
        grid_params={"ngrid": ngrid, "s0": s0},
        conv_params={"nconv": nconv, "vconv": 1e-1, "econv": 1e-2},
        band_params={"nkpts": nkpts},
        scf_params={"maxscf": maxscf},
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
        method="B",
        conv_params={"nconv": nconv, "econv": 1e-2, "vconv": 1e-3},
        write_info=False,
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

    return P_dict, full_dict


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


def gradient(f, xgrid):
    return np.gradient(f, xgrid) / (2 * xgrid)


def converge_pressure_nconv(
    atom,
    model,
    nconv_max=1e-3,
    nconv_min=1e-7,
    orb_params={
        "norbs_min": 4,
        "norbs_max": 100,
        "lorbs_min": 4,
        "lorbs_max": 200,
        "norbs_buffer": 3,
        "lorbs_buffer": 3,
        "norbs_diff": 2,
        "lorbs_diff": 2,
        "maxscf": 5,
        "threshold_max": 1e-3,
        "threshold_min": 1e-7,
    },
    grid_params={"ngrid_min": 500, "ngrid_max": 20000, "s0": 1e-4},
    kpt_params={"nkpts_min": 20, "nkpts_max": 400},
    conv_hardlim=2,
    conv_softlim=5,
):
    nconv = nconv_max
    print("Running nconv convergence")
    counter = 0

    while nconv >= nconv_min:
        nmax, lmax, thresh, _, _ = converge_pressure_norbs(
            atom,
            model,
            orb_conv_params=orb_params,
            ngrid=grid_params["ngrid_min"],
            nkpts=kpt_params["nkpts_min"],
            conv_hardlim=conv_hardlim,
            conv_softlim=conv_softlim,
            nconv=nconv,
            s0=grid_params["s0"],
        )
        # update the starting guesses for nmax and lmax
        orb_params["norbs_min"] = nmax - orb_params["norbs_buffer"]
        orb_params["lorbs_min"] = lmax - orb_params["lorbs_buffer"]
        print(nmax, orb_params["norbs_min"])

        ngrid, _, _ = converge_pressure_ngrid(
            atom,
            model,
            nmax,
            lmax,
            ngrid_min=grid_params["ngrid_min"],
            ngrid_max=grid_params["ngrid_max"],
            s0=grid_params["s0"],
            nconv=nconv,
            nkpts=kpt_params["nkpts_min"],
            conv_hardlim=conv_hardlim,
            conv_softlim=conv_softlim,
        )
        nkpts, P_dict, full_dict = converge_pressure_nkpts(
            atom,
            model,
            nmax,
            lmax,
            nkpts_min=kpt_params["nkpts_min"],
            nkpts_max=kpt_params["nkpts_max"],
            s0=grid_params["s0"],
            nconv=nconv,
            ngrid=ngrid,
            conv_hardlim=conv_hardlim,
            conv_softlim=conv_softlim,
        )

        print("nconv = ", nconv)
        print("Pressure dict = ", P_dict)
        if counter > 0:
            P_diffs = compute_relative_differences(P_dict, P_dict_save)
            full_diffs = compute_relative_differences(full_dict, full_dict_save)
            if are_differences_below_threshold(
                P_diffs, full_diffs, conv_hardlim, conv_softlim
            ):
                break
        P_dict_save, full_dict_save = P_dict, full_dict
        counter += 1

        nconv_save = nconv
        nconv /= 10

    conv_dict = {
        "nconv": nconv,
        "orbs_threshold": thresh,
        "nmax": nmax,
        "lmax": lmax,
        "ngrid": ngrid,
        "nkpts": nkpts,
    }

    return conv_dict, full_dict_save


def converge_pressure_norbs(
    atom,
    model,
    orb_conv_params={
        "norbs_min": 4,
        "norbs_max": 100,
        "lorbs_min": 4,
        "lorbs_max": 200,
        "norbs_buffer": 3,
        "lorbs_buffer": 3,
        "norbs_diff": 2,
        "lorbs_diff": 2,
        "maxscf": 5,
        "threshold_max": 1e-3,
        "threshold_min": 1e-7,
    },
    ngrid=1000,
    nkpts=30,
    s0=1e-4,
    nconv=1e-4,
    conv_hardlim=2,
    conv_softlim=5,
):
    norbs_min = orb_conv_params["norbs_min"]
    norbs_max = orb_conv_params["norbs_max"]
    lorbs_min = orb_conv_params["lorbs_min"]
    lorbs_max = orb_conv_params["lorbs_max"]
    norbs_buffer = orb_conv_params["norbs_buffer"]
    lorbs_buffer = orb_conv_params["lorbs_buffer"]
    norbs_diff = orb_conv_params["norbs_diff"]
    lorbs_diff = orb_conv_params["lorbs_diff"]
    maxscf = orb_conv_params["maxscf"]
    threshold_min = orb_conv_params["threshold_min"]
    threshold_max = orb_conv_params["threshold_max"]
    norbs = norbs_min
    lorbs = lorbs_min
    threshold = threshold_max
    counter = 0
    while threshold >= threshold_min:
        P_dict, full_dict = calc_pressure_dict(
            atom, model, norbs, lorbs, nconv=nconv, ngrid=ngrid, nkpts=nkpts, s0=s0
        )
        print("norbs, lorbs = ", norbs, lorbs)
        print("threshold = ", threshold)
        print("Pressure dict = ", P_dict)
        # FIRST increase orbitals to convergence for threshold
        while norbs <= norbs_max:
            breakloop = False
            while lorbs <= lorbs_max:
                try:
                    out_test = model.CalcEnergy(
                        norbs,
                        lorbs,
                        grid_params={"ngrid": ngrid, "s0": s0},
                        band_params={"nkpts": nkpts},
                        scf_params={"maxscf": maxscf},
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
                        norbs += norbs_diff
                    if not lorbs_ok:
                        lorbs += lorbs_diff
            if breakloop:
                break
        norbs += norbs_buffer
        lorbs += lorbs_buffer

        if counter > 0:
            P_diffs = compute_relative_differences(P_dict, P_dict_save)
            full_diffs = compute_relative_differences(full_dict, full_dict_save)
            if are_differences_below_threshold(
                P_diffs, full_diffs, conv_hardlim, conv_softlim
            ):
                break
        P_dict_save, full_dict_save = P_dict, full_dict
        counter += 1
        threshold_save = threshold
        threshold /= 10

    return norbs, lorbs, threshold_save, P_dict_save, full_dict_save


def converge_pressure_ngrid(
    atom,
    model,
    nmax,
    lmax,
    ngrid_min=500,
    ngrid_max=20000,
    nkpts=30,
    nconv=1e-4,
    s0=1e-4,
    conv_hardlim=2,
    conv_softlim=5,
):
    ngrid = ngrid_min
    print("Running ngrid convergence")
    counter = 0
    while ngrid < ngrid_max:
        P_dict, full_dict = calc_pressure_dict(
            atom, model, nmax, lmax, nconv=nconv, ngrid=ngrid, nkpts=nkpts, s0=s0
        )
        print("ngrid = ", ngrid)
        print("Pressure dict = ", P_dict)
        if counter > 0:
            P_diffs = compute_relative_differences(P_dict, P_dict_save)
            full_diffs = compute_relative_differences(full_dict, full_dict_save)
            if are_differences_below_threshold(
                P_diffs, full_diffs, conv_hardlim, conv_softlim
            ):
                break
        P_dict_save, full_dict_save = P_dict, full_dict
        counter += 1

        ngrid_save = ngrid
        ngrid = int(ngrid * 1.5)

    return ngrid_save, P_dict_save, full_dict_save


def converge_pressure_nkpts(
    atom,
    model,
    nmax,
    lmax,
    nkpts_min=20,
    nkpts_max=400,
    ngrid=1000,
    nconv=1e-4,
    s0=1e-4,
    conv_hardlim=2,
    conv_softlim=5,
):
    nkpts = nkpts_min
    print("Running nkpts convergence")
    counter = 0
    while nkpts < nkpts_max:
        P_dict, full_dict = calc_pressure_dict(
            atom, model, nmax, lmax, nconv=nconv, ngrid=ngrid, nkpts=nkpts, s0=s0
        )
        print("nkpts = ", nkpts)
        print("Pressure dict = ", P_dict)
        if counter > 0:
            P_diffs = compute_relative_differences(P_dict, P_dict_save)
            full_diffs = compute_relative_differences(full_dict, full_dict_save)
            if are_differences_below_threshold(
                P_diffs, full_diffs, conv_hardlim, conv_softlim
            ):
                break
        P_dict_save, full_dict_save = P_dict, full_dict
        counter += 1

        nkpts_save = nkpts
        nkpts = int(nkpts * 1.5)

    return nkpts_save, P_dict_save, full_dict_save


if __name__ == "__main__":
    # use all cores
    config.numcores = -1
    config.suppress_warnings = True

    atom_species = "Al"  # helium
    density = 2.7  # Wigner-Seitz radius of room-temp Be
    temperature = 50  # temperature in eV
    grid_type = "sqrt"

    # initialize the atom object
    atom = Atom(atom_species, density=density, temp=temperature, units_temp="eV")

    # initialize the model
    model = models.ISModel(atom, bc="bands", unbound="quantum")

    ### CONVERGENCE PARAMETERS ###
    ngrid_min = 500
    ngrid_max = 20000
    nconv = 1e-6
    nkpts_min = 20
    nkpts_max = 400
    s0 = 1e-4
    conv_hardlim = 2
    conv_softlim = 5
    ### END CONVERGENCE PARAMETERS ###

    # nmax, lmax, orbs_threshold, P_dict = converge_pressure_norbs(
    #     atom,
    #     model,
    #     ngrid=ngrid_min,
    #     nkpts=nkpts_min,
    #     conv_hardlim=conv_hardlim,
    #     conv_softlim=conv_softlim,
    #     nconv=nconv,
    #     s0=s0,
    # )

    # ngrid, P_dict = converge_pressure_ngrid(
    #     atom,
    #     model,
    #     nmax,
    #     lmax,
    #     ngrid_min=ngrid_min,
    #     ngrid_max=ngrid_max,
    #     nkpts=nkpts_min,
    #     nconv=nconv,
    #     s0=s0,
    #     conv_hardlim=conv_hardlim,
    #     conv_softlim=conv_softlim,
    # )

    # nkpts, P_dict = converge_pressure_nkpts(
    #     atom,
    #     model,
    #     nmax,
    #     lmax,
    #     nkpts_min=nkpts_min,
    #     nkpts_max=nkpts_max,
    #     ngrid=ngrid,
    #     nconv=nconv,
    #     s0=s0,
    #     conv_hardlim=conv_hardlim,
    #     conv_softlim=conv_softlim,
    # )

    conv_dict, full_dict_save = converge_pressure_nconv(atom, model)

    print(conv_dict)
    print(full_dict_save)
