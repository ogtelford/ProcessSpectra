"""
Tools to bin and stack galaxy spectra.

Authored by Grace Telford 02/22/16
"""

# TODO: Handle case of using chunks of larger table w/ index_offset

from __future__ import absolute_import, print_function, division

import numpy as np
import sys
from .io import get_local_params, get_galaxy_params


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
    def __init__(self, selection_dict=None, columns=None, indices=None):
        self.names = []
        self.mins = []
        self.maxs = []
        if selection_dict:
            for name in selection_dict.keys():
                self.names.append(name)
                self.mins.append(selection_dict[name][0])
                self.maxs.append(selection_dict[name][1])

        self.columns = columns
        self.indices = indices

        self.galaxy_params = get_galaxy_params(columns=self.columns, indices=self.indices)

    def get_selection_indices(self):

        keep = np.ones(len(self.galaxy_params), dtype=bool)

        for ind in range(len(self.names)):
            keep *= (self.galaxy_params[self.names[ind]] < self.maxs[ind]) * \
                    (self.galaxy_params[self.names[ind]] > self.mins[ind])

        return keep