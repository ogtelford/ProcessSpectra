"""
Tools to bin and stack galaxy spectra.

Authored by Grace Telford 02/22/16
"""

from __future__ import absolute_import, print_function, division

import numpy as np
import sys
from .io import get_local_params, get_galaxy_params, save_spectrum_starlight
from .processing import SpecProcessor


class SpecStacker(object):
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
    def __init__(self, selection_dict=None, columns=None):

        self.galaxy_params = get_galaxy_params(columns=columns)

        self.names = []
        self.mins = []
        self.maxs = []
        if selection_dict:
            for name in selection_dict.keys():
                self.names.append(name)
                self.mins.append(selection_dict[name][0])
                self.maxs.append(selection_dict[name][1])

        keep = np.ones(len(self.galaxy_params), dtype=bool)

        for ind in range(len(self.names)):
            keep *= (self.galaxy_params[self.names[ind]] < self.maxs[ind]) * \
                    (self.galaxy_params[self.names[ind]] > self.mins[ind])

        self.stack_inds = np.where(keep > 0)[0]

        self.galaxy_params = self.galaxy_params[self.stack_inds]
        self.Nspectra = len(self.galaxy_params)

    def make_files_list(self):
        filenames = []
        for ii in self.stack_inds:
            filenames.append('spec-%04d-%05d-%04d.fits' % (self.galaxy_params['PLATEID'][ii],
                                                           self.galaxy_params['MJD'][ii],
                                                           self.galaxy_params['FIBER'][ii]))

        return filenames

    def get_stacked_spectrum(self, spec_array=None, weights_array=None, method='mean', err_method='rms', \
                             mcmc_samples=100, missing_params=False, missing_spec=False, spec_filenames_file=None):
        """
        Calculate mean or median stack of spectra.

        Parameters
        ----------
        spectra : ndarray
            An N_spectra x N_wavelengths array containing all spectra interpolated to common,
            de-redshifted wavelength grid.
        method : string (default='mean')
            Method of stacking. Must be either 'mean' or 'median'
        plot : boolean (default=True)

        Returns
        -------
        stack
        """

        # Get array of spectra and weights (ivars)
        if spec_array:
            wavelengths = None
            spectra = spec_array
            weights = weights_array
        else:
            sp = SpecProcessor(spectrum_filenames_file=spec_filenames_file)
            spectra, weights = sp.process_fits(indices=self.stack_inds, normalize=True,
                                               missing_params=missing_params, missing_spec=missing_spec)
            wavelengths = 10 ** sp.loglam_grid
            keep = (wavelengths > 3700) * (wavelengths < 8400)
            wavelengths = wavelengths[keep]
            spectra = spectra[:,keep]
            weights = weights[:,keep]

        # check for number of spectra with masked pixels at each wavelength
        #frac_masked = (weights==0).mean(0)

        # Perform stacking of spectra
        spectra[spectra == 0] = np.nan

        if method == 'mean':
            stack = np.nanmean(spectra, axis=0)

        elif method == 'median':
            stack = np.nanmedian(spectra, axis=0)

        else:
            sys.exit('method keyword must be either "mean" or "median"')

        # Calculate uncertainties on each pixel in stacked spectrum
        # NB -- THIS IS UNDER DEVELOPMENT; current method of using nans to ignore pixels with no data isn't great...
        if err_method == 'rms':
            errs = np.sqrt(np.nansum((spectra - stack) ** 2, axis=0) / self.Nspectra)

        elif err_method == 'mcmc':
            sigmas = 1. / np.sqrt(weights)
            sigmas[np.isinf(sigmas)] = 10. # better way to choose this value??

            resampled_stacks = np.zeros((mcmc_samples, len(wavelengths)))

            for samp in range(mcmc_samples):
                resampled = np.zeros(np.shape(spectra))

                for ii in range(self.Nspectra):
                    for jj in range(len(wavelengths)):
                        resampled[ii,jj] = np.random.normal(loc=spectra[ii,jj], scale=sigmas[ii,jj])

                resampled_stacks[samp,:] = np.nanmean(resampled, axis=0)

            errs = np.nanstd(resampled_stacks, axis=0)


        else:
            sys.exit('err_method keyword must be either "rms" or "mcmc"')


        if wavelengths is not None:
            return wavelengths, stack, errs
        else:
            return stack, errs