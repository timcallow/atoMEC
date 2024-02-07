"""
Contains classes and functions used to compute and store aspects related to convergence.

So far, the only procedure requring convergence is the static SCF cycle. More will
be added in future.

Classes
-------
* :class:`SCF` : holds the SCF convergence attributes and calculates them for the \
                 given cycle
"""

# standard libraries

# external libraries
import numpy as np

# internal modules
from . import mathtools
from . import config


class SCF:
    """
    Convergence attributes and functions related to SCF energy procedures.

    Notes
    -----
    Contains the private attributes _energy, _potential and _density

    """

    def __init__(self, xgrid, grid_type):
        self._xgrid = xgrid
        self._energy = np.zeros((2))
        self._density = np.zeros((2, config.spindims, config.grid_params["ngrid"]))
        self.grid_type = grid_type

    def check_conv(self, E_free, dens, iscf):
        """
        Compute and check the changes in energy and integrated density.
=
        If the convergence tolerances are all simultaneously satisfied, the `complete` \
        variable returns `True` as the SCF cycle is complete.

        Parameters
        ----------
        E_free : float
            the total free energy
        dens : ndarray
            the electronic density
        iscf : int
            the iteration number

        Returns
        -------
        conv_vals : dict
            Dictionary of convergence parameters as follows:
            {
            `conv_energy` : ``float``,   `conv_rho` : ``ndarray``,
            `complete` : ``bool``
            }
        """
        conv_vals = {}

        # first update the energy, potential and density attributes
        self._energy[0] = E_free
        self._density[0] = dens

        # compute the change in energy
        conv_vals["dE"] = abs((self._energy[0] - self._energy[1]) / self._energy[0])

        # compute the change in density
        dn = np.abs(self._density[0] - self._density[1])
        # integrate over sphere to return a number
        # add a small constant to avoid errors if no electrons in one spin channel
        conv_vals["drho"] = mathtools.int_sphere(dn, self._xgrid, self.grid_type) / (
            config.nele + 1e-3
        )

        # reset the energy, potential and density attributes
        self._energy[1] = E_free
        self._density[1] = dens

        # see if the convergence criteria are satisfied
        conv_energy = False
        conv_rho = False
        conv_vals["complete"] = False

        if iscf > 1:
            if conv_vals["dE"] < config.conv_params["econv"]:
                conv_energy = True
            if np.all(conv_vals["drho"] < config.conv_params["nconv"]):
                conv_rho = True
            if conv_energy and conv_rho:
                conv_vals["complete"] = True

        return conv_vals
