"""
Tools for reading and writing spectrum and parameter files.

Authored by Grace Telford 02/13/16.
"""

# TODO: make sure variable names are consistent between *kwargs and local.cfg

from __future__ import absolute_import, print_function, division

import numpy as np
from astropy.io import fits
import h5py
import sys


def get_local_params():
    """
    Read local parameters from local.cfg file in the pwd.
    Each line contains a parameter name followed by ": " and the parameter.

    Returns
    -------
    """
    local_params = {}
    with open('local.cfg') as f:
        for raw_line in f:
            parsed_line = raw_line.strip().split(': ')
            if len(parsed_line) == 2:
                param_name, param_value = parsed_line
                local_params[param_name] = param_value
    return local_params


class FitsData(object):
    """
    Extracts relevant information from spectrum .fits files.
    User may specify directory containing spectra in local.cfg file or using path_to_files keyword.

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self, filename, path_to_files=None):
        if path_to_files:
            self.path_to_files = path_to_files

        else:
            try:
                self.path_to_files = get_local_params()['spectra_directory']
            except KeyError:
                sys.exit('Specify path_to_files in *kwargs or spectra_directory in local.cfg file')

        self.filename = filename

        try:
            fitsfile = fits.open(self.path_to_files + self.filename)
        except IOError:
            sys.exit('Invalid filename or spectra directory')

        self.z = fitsfile[2].data['z']
        self.fluxes = fitsfile[1].data['flux']
        self.wavelengths = 10 ** fitsfile[1].data['loglam']
        self.ivars = fitsfile[1].data['ivar']
        self.andmask = fitsfile[1].data['and_mask']

        fitsfile.close()


def get_galaxy_params(galaxy_parameters_file=None, columns=(2,3,4,5,7), indices=None):
    """
    Read in a table of parameters

    Parameters
    ----------
    galaxy_parameters_file: str
    columns: tuple
        Column numbers to read in. Default is (2,3,4,5,7), corresponding to
        ('PLATEID', 'MJD', 'FIBER', 'Z', 'EBV') assuming that galaxy_parameters_file was generated
        by the notebook galaxyData.ipynb within the /Research/ManifoldLearning/code directory
    indices: int, list, or ndarray

    Returns
    -------
    """
    if galaxy_parameters_file is None:
        try:
            galaxy_parameters_file = get_local_params()['galaxy_parameters_file']
        except KeyError:
            sys.exit('Specify galaxy_parameters_file in *kwargs or in local.cfg file')

    if columns:
        try:
            galaxyparams = np.genfromtxt(galaxy_parameters_file, delimiter=',', names=True, dtype=float, usecols=columns)
        except IOError:
            sys.exit('Invalid galaxy_parameters_file')
    else:
        try:
            galaxyparams = np.genfromtxt(galaxy_parameters_file, delimiter=',', names=True, dtype=float)
        except IOError:
            sys.exit('Invalid galaxy_parameters_file')

    if indices:
        return galaxyparams[indices]
    else:
        return galaxyparams



