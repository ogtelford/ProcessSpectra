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
    Extract relevant information from spectrum .fits files.

    Attributes
    ----------
        filename: string
            Filename of SDSS .fits file contatining galaxy spectrum
        spectra_directory: string
            Path of directory contatining the spectrum .fits file
        z: float
            Redshift of the galaxy
        plate: integer
            SDSS plate id
        mjd: integer
            SDSS MJD
        fiber: integer
            SDSS fiber number
        fluxes: ndarray
            Spectrum (flux measured at each wavelength) of the galaxy
        wavelengths: ndarray
            Wavelengths corresponding to each measurement in the spectrum
        ivars: ndarray
            Inverse variance at each wavelength, that is, 1 / (measurement error) ** 2
        andmask: ndarray
            Value of mask at each wavelength, > 0 if set in any exposure of a given galaxy
    """
    def __init__(self, filename, spectra_directory=None):
        self.filename = filename

        if spectra_directory is None:
            self.spectra_directory = './'
        else:
            self.spectra_directory = spectra_directory

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


def get_galaxy_params(galaxy_parameters_file, columns=None, indices=None):
    """
    Read in a table of galaxy parameters.

    Parameters
    ----------
    galaxy_parameters_file: string
        Specify the file containing galaxy parameters. Must include Plate/MJD/Fiber for each galaxy.
    columns: tuple
        Optional. Specify a subset of columns to read. Default None reads in all columns.
    indices: integer, list, or ndarray
        Optional. Specify rows of parameters table to read in. Default None reads in all rows.

    Returns
    -------
    galaxy_params: numpy record array
        Array of properties for each galaxy, where each column is named according to column heading in the .csv file.

    Raises
    ------
    IOError:
        An error occurred when attempting to read the galaxy parameters .csv file.
    """
    if columns:
        try:
            galaxyparams = np.genfromtxt(galaxy_parameters_file, delimiter=',',
                                         names=True, dtype=float, usecols=columns)
        except IOError:
            sys.exit('Invalid galaxy_parameters_file')
    else:
        try:
            galaxyparams = np.genfromtxt(galaxy_parameters_file, delimiter=',',
                                         names=True, dtype=float)
        except IOError:
            sys.exit('Invalid galaxy_parameters_file')

    if indices:
        return galaxyparams[indices]
    else:
        return galaxyparams


def read_filenames(spectrum_filenames_file):
    """
    Simple function to return an array of spectrum filenames from a file.

    NB: if using with SpecStacker, must be the same number of filenames as spectra you wish to stack!

    Parameters
    ----------
    spectrum_filenames_file: string
        Path and filename of text file containing a list of filenames of SDSS spectrum .fits files.

    Returns
    -------
    filenames: ndarray of strings
        Array of filenames of SDSS spectrum .fits files.
    """
    try:
        return np.loadtxt(spectrum_filenames_file, dtype=str)
    except IOError:
        sys.exit('Invalid spectrum_filenames_file')


def make_files_list(galaxy_params, indices=None):
    """
    Generate list of spectrum filenames given a galaxy_params file.

    NB: galaxy_params MUST contain Plate/MJD/Fiber for this to work!

    Parameters
    ----------
    galaxy_params: numpy record array
        Array of properties for each galaxy, where each column is named according to column heading in the .csv file.
        Output of get_galaxy_params function.
    indices: integer, list, or ndarray
        Optional. Specify a subset of rows in galaxy_params for which to generate filenames.
    Returns
    -------
    filenames: list of strings
        List of filenames of SDSS spectrum .fits files.
    """
    if indices is None:
        indices = np.arange(len(galaxy_params))

    filenames = []
    for ii in indices:
        filenames.append('spec-%04d-%05d-%04d.fits' % (galaxy_params['PLATEID'][ii],
                                                       galaxy_params['MJD'][ii],
                                                       galaxy_params['FIBER'][ii]))

    return filenames


def save_spectra_hdf5(wavelength_grid, spectra, weights, output_filename='spectra.hdf5', dataset_dict=None):
    """
    Save arrays of wavelengths, spectra, and weights to an HDF5 file.

    Can specify additional dataset names and data using the dataset_dict keyword.
    Useful for storing identifying information for spectra.

    Parameters
    ----------
    wavelength_grid: ndarray
        n_samples length array, common wavelength grid corresponding to flux measurements for all spectra
    spectra: ndarray
        n_spectra x n_samples array, where each row is the spectrum for a given galaxy
    weights: ndarray
        n_spectra x n_samples array, where each row is the inverse variance at each wavelength for a given galaxy
    output_filename: string
        HDF5 filename where the arrays are to be stored.
    dataset_dict: dictionary
        Optional. Dictionary where keys are additional dataset names and values are additional data to store
        in the HDF5 file.
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


def save_spectrum_starlight(wavelength_grid, spectrum, errors, minlam=3700, maxlam=8200, flags=None,
                            output_filename='spectrum_starlight.txt'):
    """
    Save arrays of wavelengths, spectra, and weights to a text file formatted for use with STARLIGHT.

    NB: STARLIGHT requires uniform, 1 angstrom spacing of wavelengths! Need to resample arrays here.

    Parameters
    ----------
    wavelength_grid: ndarray
        Linearly spaced wavelength array. Any spacing allowed, but will be resampled to 1 Angstrom.
    spectrum: ndarray
        Flux measurements at each wavelength.
    errors: ndarray
        Measurement error on each flux measurement.
    minlam: integer
        Minimum wavelength of resampled wavelength grid.
    maxlam: integer
        Maximum wavelength of resampled wavelength grid.
    flags: ndarray
        Optional. Set > 0 at each wavelength where STARLIGHT should ignore that pixel in the fit.
    output_filename: string
        Filename of text file where STARLIGHT observation file should be saved.
    """
    if output_filename is None:
        sys.exit('Specify output_filename in *kwargs or in local.cfg file')

    # interpolate spectrum, errors, and flags (f > 1 == bad pixel) to linear wavelength grid
    lam_grid = np.arange(minlam, maxlam, 1.)

    data = np.zeros((len(lam_grid), 4))
    data[:, 0] = lam_grid
    data[:, 1] = interp(lam_grid, wavelength_grid, spectrum, left=0., right=0.)
    data[:, 2] = interp(lam_grid, wavelength_grid, errors, left=0., right=0.)
    if flags is not None:
        data[:, 3] = interp(lam_grid, wavelength_grid, flags, left=0., right=0.)

    np.savetxt(output_filename, data)
