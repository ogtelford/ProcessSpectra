"""
Tools for reading and writing spectrum and parameter files.

Authored by Grace Telford 02/13/16.
"""

#TODO: add function to write a STARLIGHT input file

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
    def __init__(self, filename, spectra_directory=None):
        if spectra_directory:
            self.spectra_directory = spectra_directory

        else:
            try:
                self.spectra_directory = get_local_params()['spectra_directory']
            except KeyError:
                sys.exit('Specify spectra_directory in *kwargs or in local.cfg file')

        self.filename = filename

        try:
            fitsfile = fits.open(self.spectra_directory + self.filename)
        except IOError:
            sys.exit('Invalid filename or spectra directory')

        self.z = fitsfile[2].data['z']
        self.plate = fitsfile[2].data['plate']
        self.mjd = fitsfile[2].data['mjd']
        self.fiber = fitsfile[2].data['fiberid']
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


def save_spectra_hdf5(loglam_grid, spectra, weights, output_filename=None, dataset_dict=None):
    """
    Save arrays of wavelengths, spectra, and weights to an HDF5 file.

    Can specify output_filename in local.cfg file in pwd or in *kwargs.
    Can specify additional dataset names and data using the dataset_dict keyword.
    Useful for storing identifying information for spectra.

    Parameters
    ----------
    loglam_grid:
    spectra:
    weights:
    output_filename:
    dataset_dict:
    """
    if output_filename is None:
        output_filename=get_local_params()['output_filename']

    f = h5py.File(output_filename, 'w')

    f.create_dataset('log_wavelengths', data=loglam_grid)
    f.create_dataset('spectra', data=spectra)
    f.create_dataset('ivars', data=weights)

    if dataset_dict:
        for name in dataset_dict.keys():
            f.create_dataset(name, data=dataset_dict[name])

    f.close()


def read_spectra_hdf5(filename=None):
    """
    Read data saved in HDF5 format

    Parameters
    ----------
    filename:
    """
    if filename is None:
        try:
            filename=get_local_params()['output_filename']
        except KeyError:
            sys.exit('Specify filename in *kwargs or in or output_filename in local.cfg file')

    try:
        f = h5py.File(filename,'r')
    except IOError:
        sys.exit('Invalid filename')

    return f