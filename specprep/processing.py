"""
Tools to process raw galaxy spectra from SDSS-II Legacy survey.

Authored by Grace Telford 02/13/16
"""

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
    def __init__(self, Nsamples=5000, loglam_grid=None, files_list=None):
        if loglam_grid:
            self.loglam_grid = loglam_grid
            self.Nsamples = len(loglam_grid)
        else:
            self.loglam_grid = 3.5 + 0.0001 * np.arange(Nsamples)
            self.Nsamples = Nsamples

        if files_list:
            self.files_list = files_list
        else:
            try:
                self.files_list = get_local_params()['spectrum_filenames_file']
            except KeyError:
                sys.exit('Specify files_list in *kwargs or spectrum_filenames_file in local.cfg file')

        self.filenames = np.loadtxt(self.files_list, dtype=str)
        self.Nspectra = len(self.filenames)

        try:
            self.index_offset = get_local_params()['index_offset']
        except KeyError:
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
        wavelength : float or array-like
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


    def process_fits(self, missing_params=False):
        """
        Iterate over all .fits filenames, read in and process spectra.
        Check that redshift in header matches redshift in parameters file.
        """
        start_time = time.time()

        spectra = np.zeros((self.Nspectra, self.Nsamples))
        weights = np.zeros((self.Nspectra, self.Nsamples))
        missing = []
        redshifts = []

        galaxyparams = get_galaxy_params()

        for ind in np.arange(self.Nspectra):
            data = FitsData(self.filenames[ind])

            #z = data.z
            #fluxes = data.fluxes
            #wavelengths = data.wavelengths
            #ivars = data.ivars
            #andmask = data.andmask

            # sanity check
            #print("fits: ", data.z)
            #print("table: ", galaxyparams['Z'][ind + self.index_offset])

            if missing_params:
                # this code handles the case where there are more filenames than rows in table
                if galaxyparams['Z'][ind + self.index_offset] != data.z:
                    self.index_offset -= 1
                    continue
            else:
                # this code handles the case where there are more rows in the table than filenames
                while galaxyparams['Z'][ind + self.index_offset] != data.z:
                    print("Galaxy %d missing" % (ind + self.index_offset))
                    missing.append(ind + self.index_offset)
                    self.index_offset += 1
                    if self.index_offset > 10:
                        break

            # Shift to restframe, apply mask, correct for reddening
            loglam = np.log10(data.wavelengths / (1. + data.z))
            data.ivars[data.andmask > 0] = np.nan
            ebv = galaxyparams['EBV'][ind + self.index_offset]
            data.fluxes = self.deredden(loglam, data.fluxes, ebv)

            # Interpolate spectrum/ivars & resample to common grid; set all NaNs in ivar array to very small ivar
            spectra[ind, :] = interp(self.loglam_grid, loglam, data.fluxes, left=0., right=0.)
            weights[ind, :] = interp(self.loglam_grid, loglam, data.ivars, left=0., right=0.)
            weights[ind, np.isnan(weights[ind,:])] = 0.

            if ind % 10 == 0:
                current_time = time.time()
                print('Time to index %d: %g' % (ind, current_time - start_time))

            # Normalize the spectrum -- commenting out for now! Really want mean of 0 and variance of 1 for PCA
            #norm = np.mean(spectra[:, (self.lam_grid > 4400.) * (self.lam_grid < 4450.)], axis=1)
            #spectra /= norm[:, None]
            #variances /= norm[:, None] ** 2

        keep = np.sum(spectra, axis=1) != 0

        end_time = time.time()
        print('Total time:', end_time - start_time)
        print(missing)

        return spectra[keep,:], weights[keep,:], missing