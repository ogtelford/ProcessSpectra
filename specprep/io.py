"""
Tools for reading and writing spectrum and parameter files.

Authored by Grace Telford 02/13/16.
"""

from __future__ import absolute_import, print_function, division

import numpy as np
from astropy.io import fits
from scipy import interp
import h5py
import sys

class FitsData(object):
    """
    Extracts relevant information from spectrum .fits files.
    User may specify directory containing spectra in local.cfg file or using path_to_files keyword.

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self, filename,
                 spectra_directory='/Users/ogtelford/Documents/UW/Research/StackingSpectra/data/M9.8_SFR0/'):
        if spectra_directory:
            self.spectra_directory = spectra_directory
        else:
            sys.exit('Specify spectra_directory in *kwargs')

        self.filename = filename

        try:
            fitsfile = fits.open(self.spectra_directory + self.filename, memmap=False)
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


def get_galaxy_params(galaxy_parameters_file='/Users/ogtelford/Documents/UW/Research/StackingSpectra/\
                            code/data_M9.8_SFR0.csv', columns=None, indices=None):
    """
    Read in a table of parameters

    Parameters
    ----------
    galaxy_parameters_file: str
        Specify the file containing galaxy parameters. Must include Plate/MJD/Fiber for each galaxy.
    columns: tuple
        Optional. Specify a subset of columns to read. Default None reads in all columns.
    indices: int, list, or ndarray
        Optional. Specify rows of parameters table to read in. Default None reads in all rows.

    Returns
    -------
    """
    if galaxy_parameters_file is None:
        sys.exit('Specify galaxy_parameters_file in *kwargs')

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

def read_filenames(spectrum_filenames_file):
    """
    Simple function to return an array of spectrum filenames from a file.

    NB: must be the same number of filenames as spectra you wish to stack!
    """
    return np.loadtxt(spectrum_filenames_file, dtype=str)


def make_files_list(galaxy_params, indices):
    filenames = []
    for ii in indices:
        filenames.append('spec-%04d-%05d-%04d.fits' % (galaxy_params['PLATEID'][ii],
                                                       galaxy_params['MJD'][ii],
                                                       galaxy_params['FIBER'][ii]))

    return filenames

def save_spectra_hdf5(wavelength_grid, spectra, weights, output_filename='spectra.hdf5', dataset_dict=None):
    """
    Save arrays of wavelengths, spectra, and weights to an HDF5 file.
    Must specify output_filename in *kwargs.

    Can specify additional dataset names and data using the dataset_dict keyword.
    Useful for storing identifying information for spectra.

    Parameters
    ----------
    wavelength_grid:
    spectra:
    weights:
    output_filename:
    dataset_dict:
    """
    if output_filename is None:
        sys.exit('Specify output_filename in *kwargs')

    f = h5py.File(output_filename, 'w')

    f.create_dataset('wavelengths', data=wavelength_grid)
    f.create_dataset('spectra', data=spectra)
    f.create_dataset('ivars', data=weights)

    if dataset_dict:
        for name in dataset_dict.keys():
            f.create_dataset(name, data=dataset_dict[name])

    f.close()


def read_spectra_hdf5(filename=None):
    """
    Read data saved in HDF5 format.

    Parameters
    ----------
    filename: str
        Name of HDF5 file that contains the spectra.

    Returns
    -------
    f:
    """
    if filename is None:
        sys.exit('Specify filename in *kwargs or in or output_filename in local.cfg file')

    try:
        f = h5py.File(filename,'r')
    except IOError:
        sys.exit('Invalid filename')

    return f


def save_spectrum_starlight(wavelength_grid, spectrum, errors, minlam=3700, maxlam=8200, flags=None,
                            output_filename='spectrum_starlight.txt'):
    """
    Save arrays of wavelengths, spectra, and weights to a text file formatted for use with STARLIGHT.

    Can specify output_filename in local.cfg file in pwd or in *kwargs.
    NB: STARLIGHT requires uniform, 1 angstrom spacing of wavelengths! Need to resample arrays here.

    Parameters
    ----------
    wavelength_grid:
    spectrum:
    weights:
    minlam:
    maxlam:
    flags:
    output_filename:
    """
    if output_filename is None:
        sys.exit('Specify output_filename in *kwargs or in local.cfg file')

    # interpolate spectrum, errors, and flags (f > 1 == bad pixel) to linear wavelength grid
    lam_grid = np.arange(minlam, maxlam, 1.)

    specData = np.zeros((len(lam_grid), 4))
    specData[:,0] = lam_grid
    specData[:,1] = interp(lam_grid, wavelength_grid, spectrum, left=0., right=0.)
    specData[:,2] = interp(lam_grid, wavelength_grid, errors, left=0., right=0.)
    if flags is not None:
        specData[:,3] = interp(lam_grid, wavelength_grid, flags, left=0., right=0.)

    np.savetxt(output_filename, specData)
