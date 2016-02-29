"""
Tools to process galaxy spectra .fits files from SDSS-II Legacy survey.

Authored by Grace Telford 02/13/16
"""

#TODO: add option to save to HDF5 file to the processing function

from __future__ import absolute_import, print_function, division

import numpy as np
from scipy import interp
import time
import sys

from .io import FitsData, get_local_params, get_galaxy_params


class SpecProcessor(object):
    """
    Perform basic processing of raw spectra.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>>
    """
    def __init__(self, Nsamples=5000, loglam_grid=None, spectrum_filenames_file=None):
        if loglam_grid:
            self.loglam_grid = loglam_grid
            self.Nsamples = len(loglam_grid)
        else:
            self.loglam_grid = 3.5 + 0.0001 * np.arange(Nsamples)
            self.Nsamples = Nsamples

        if spectrum_filenames_file:
            self.spectrum_filenames_file = spectrum_filenames_file
        else:
            try:
                self.spectrum_filenames_file = get_local_params()['spectrum_filenames_file']
            except (KeyError, IOError):
                sys.exit('Specify spectrum_filenames_file in *kwargs or in local.cfg file')

        self.filenames = np.loadtxt(self.spectrum_filenames_file, dtype=str)
        self.Nspectra = len(self.filenames)

        try:
            self.index_offset = get_local_params()['index_offset']
        except (KeyError, IOError):
            self.index_offset = 0

    @staticmethod
    def k(wavelength, r_v=3.1):
        """
        Return A_wavelength/A_V using CCM 1989 extincton law.

        Parameters
        ----------
        wavelength : float or array-like
            Wavelength(s) at which to compute the reddening correction.
        r_v : float (default=3.1)
            R_V value assumed in reddening law.

        Returns
        -------
        k : float or array-like
            Value(s) of k(lambda) at the specified wavelength(s).
        """

        x = 1. / (wavelength / 10000.)
        """
        Valid for 1.1 < x < 3.3 - all wavelengths in this code are between 1.35 and 2.7.
        """
        y = x - 1.82
        a = 1. + 0.17699 * y - 0.50447 * (y ** 2) - 0.02427 * (y ** 3) + 0.72085 * (y ** 4) + 0.01979 * (
            y ** 5) - 0.77530 * (y ** 6) + 0.32999 * (y ** 7)
        b = 1.41338 * y + 2.28305 * (y ** 2) + 1.07233 * (y ** 3) - 5.38434 * (y ** 4) - 0.62251 * (
            y ** 5) + 5.30260 * (y ** 6) - 2.09002 * (y ** 7)
        return a + b / r_v

    def deredden(self, log_wavelength, flux, ebv):
        """
        Correct flux at specified wavelength(s) for reddening using CCM 1989 extinction law.

        Parameters
        ----------
        log_wavelength : float or array-like
            Wavelength(s) at which to compute the reddening correction.
        flux : float or array-like
            Uncorrected flux(es).
        ebv : float
            Value of E(B-V).

        Returns
        -------
        flux_corr : float or array-like
            Flux(es) corrected for reddening.
        """

        return flux * 10 ** (0.4 * self.k(10**log_wavelength) * ebv)

    def normalize(self, spectra, weights):
        """
        Normalize the array of spectra to mean value of each spectrum between 4400 and 4450 A

        Multiply inverse variances by the square of the normalization

        Parameters
        ----------

        Returns
        -------
        """
        norm = np.mean(spectra[:, (10**self.loglam_grid > 4400.) * (10**self.loglam_grid < 4450.)], axis=1)
        spectra /= norm[:, None]
        weights *= norm[:, None] ** 2

        return spectra, weights

    def process_fits(self, normalize=False, mask=False, indices=None, missing_params=False, missing_spec=False, return_ID=False):
        """
        Iterate over all .fits filenames, read in and process spectra.

        Check that redshift in header matches redshift in parameters file.
        """
        start_time = time.time()
        counter = 0

        spectra = np.zeros((self.Nspectra, self.Nsamples))
        weights = np.zeros((self.Nspectra, self.Nsamples))
        missing = []
        redshifts = []
        plates = []
        mjds = []
        fibers = []

        galaxyparams = get_galaxy_params()

        if indices is not None:
            index_list = indices
        else:
            index_list = np.arange(self.Nspectra)

        for ind in index_list:
            data = FitsData(self.filenames[ind])

            # sanity check
            #print("fits: ", data.z)
            #print("table: ", galaxyparams['Z'][ind + self.index_offset])

            redshifts.append(data.z)
            plates.append(data.plate)
            mjds.append(data.mjd)
            fibers.append(data.fiber)

            if missing_params:
                # this code handles the case where there are more filenames than rows in table
                if galaxyparams['Z'][ind + self.index_offset] != data.z:
                    self.index_offset -= 1
                    continue

            if missing_spec:
                # this code handles the case where there are more rows in the table than filenames
                while galaxyparams['Z'][ind + self.index_offset] != data.z:
                    print("Galaxy %d missing" % (ind + self.index_offset))
                    missing.append(ind + self.index_offset)
                    self.index_offset += 1
                    if self.index_offset > 1e6:
                        break

            if mask:
                data.ivars[data.andmask > 0] = np.nan

            # Shift to restframe, apply mask, correct for reddening
            loglam = np.log10(data.wavelengths / (1. + data.z))
            ebv = galaxyparams['EBV'][ind + self.index_offset]
            data.fluxes = self.deredden(loglam, data.fluxes, ebv)

            # Interpolate spectrum/ivars & resample to common grid; set all NaNs in ivar array to 0 (masked)
            spectra[ind, :] = interp(self.loglam_grid, loglam, data.fluxes, left=0., right=0.)
            weights[ind, :] = interp(self.loglam_grid, loglam, data.ivars, left=0., right=0.)
            weights[ind, np.isnan(weights[ind,:])] = 0.

            # Progress report
            if counter % 10 == 0:
                current_time = time.time()
                print('Time to iteration %d: %g' % (counter, current_time - start_time))

            counter += 1

        # Remove any rows where all fluxes are 0 -- this occurs when there are more spectra than rows in table
        keep = np.sum(spectra, axis=1) != 0
        spectra = spectra[keep,:]
        weights = weights[keep,:]

        if normalize:
            spectra, weights = self.normalize(spectra, weights)

        end_time = time.time()
        print('Total time:', end_time - start_time)

        if return_ID:
            ID_dict = {'redshifts': redshifts, 'plates': plates, 'mjds': mjds, 'fibers': fibers}
            return spectra, weights, ID_dict, missing
        else:
            return spectra, weights