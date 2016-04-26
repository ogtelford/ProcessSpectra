"""
Tools to bin and stack galaxy spectra.

Authored by Grace Telford 02/22/16
"""

from __future__ import absolute_import, print_function, division

import numpy as np
import sys
from .io import get_galaxy_params, read_filenames, make_files_list
from .processing import SpecProcessor


class SpecStacker(object):
    """
    Perform basic processing of raw spectra.

    Attributes
    ----------
    galaxy_params: numpy record array
    names: list
    mins: list
    maxs: list
    stack_inds: ndarray
    Nspectra: integer
    """

    def __init__(self, selection_dict=None, columns=None, galaxy_parameters_file=None,
                 spectra_directory=None):

        if galaxy_parameters_file:
            self.galaxy_params = get_galaxy_params(columns=columns, galaxy_parameters_file=galaxy_parameters_file)
        else:
            self.galaxy_params = get_galaxy_params(columns=columns)

        self.spectra_directory = spectra_directory

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

    def get_stacked_spectrum(self, spec_array=None, weights_array=None, spec_filenames_file=None,
                             method='mean', err_method='rms', mcmc_samples=100):
        """
        Calculate mean or median stack of spectra.

        May be used to process and stack spectra from .fits in a single step; otherwise, can pass arrays
        of spectra and ivars to this function and generate a stack directly from preprocessed data.

        Parameters
        ----------
        spec_array: ndarray (default=None)
            Optional. An Nspectra x Nwavelengths array containing all spectra interpolated to common,
            de-redshifted wavelength grid.
        weights_array: ndarray (default=None)
            Optional. Inverse variance associated with each flux measurement in spec_array.
        spec_filenames_file: string
            NB: this will ONLY work if the filenames in this file are line-by-line matched to self.galaxy_params;
            DO NOT use a selection_dict and pass a list of filenames!
        method: string (default='mean')
            Method of stacking. Must be either 'mean' or 'median'
        err_method: string (default='rms')
            Method of computing flux errors in the stacked spectrum. Must be either 'rms' or 'mcmc'
        mcmc_samples: integer
            Number of resamplings in the MCMC error estimate

        Returns
        -------
        wavelengths: ndarray
            Only returned if no spectrum/weights arrays were passed to the function.
        stack: ndarray
            Stacked spectrum
        errs: ndarry
            Uncertainty in the stacked spectrum at each wavelength
        """
        if spec_filenames_file:
            filenames = read_filenames(spec_filenames_file)
        else:
            filenames = make_files_list(self.galaxy_params, indices=None)

        # Get array of spectra and weights (ivars)
        if spec_array:
            wavelengths = None
            spectra = spec_array
            weights = weights_array
        else:
            sp = SpecProcessor(spectrum_filenames_file=None, filenames=filenames, spectra_directory=self.spectra_directory)
            spectra, weights = sp.process_fits(indices=None, normalize=True)
            wavelengths = 10 ** sp.loglam_grid
            keep = (wavelengths > 3700) * (wavelengths < 8200)
            wavelengths = wavelengths[keep]
            spectra = spectra[:, keep]
            weights = weights[:, keep]

        # check for number of spectra with masked pixels at each wavelength
        # frac_masked = (weights==0).mean(0)

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
            sigmas[np.isinf(sigmas)] = 10.  # better way to choose this value??

            resampled_stacks = np.zeros((mcmc_samples, len(wavelengths)))

            for samp in range(mcmc_samples):
                resampled = np.zeros(np.shape(spectra))

                for ii in range(self.Nspectra):
                    for jj in range(len(wavelengths)):
                        resampled[ii, jj] = np.random.normal(loc=spectra[ii, jj], scale=sigmas[ii, jj])

                resampled_stacks[samp, :] = np.nanmean(resampled, axis=0)

            errs = np.nanstd(resampled_stacks, axis=0)

        else:
            sys.exit('err_method keyword must be either "rms" or "mcmc"')

        if wavelengths is not None:
            return wavelengths, stack, errs
        else:
            return stack, errs
