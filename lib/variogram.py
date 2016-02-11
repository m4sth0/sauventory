#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016

# Author(s):

#   Thomas Leppelt <thomas.leppelt@dwd.de>

# This file is part of sauventory.
# Spatial Autocorrelated Uncertainty of Inventories

# sauventory is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# sauventory is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# sauventory comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
# This is free software, and you are welcome to redistribute it
# under certain conditions; type `show c' for details.
"""
This module defines a semivariogram class with methods to create empirical
variograms and derive theoretical variogram functions. In addition the class
provides methos for variogram plots including empirical and theoretical
semivariograms.

The three standard types of variogram functions: spherical, exponential, and
the Gaussian are supported.
"""

import logging
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Configure logger.
logger = logging.getLogger('semivariogram')


class Variogram(object):
    """Semivariogram class

    The class methods are influenced by work of Johnson Connor:
    http://connor-johnson.com/2014/03/20/simple-kriging-in-python/
    """

    def __init__(self):
        pass

    def semivvarh(self, P, h, bw):
        '''Experimental semivariogram for a single lag.

        Keyword arguments:
            P    Input data array
            h    Lag distance
            bw    Bandwise

        Returns:
            Semivariance
        '''
        pd = squareform(pdist(P[:, :2]))
        N = pd.shape[0]
        print(N)
        Z = list()
        for i in range(N):
            for j in range(i+1, N):
                if(pd[i, j] >= h - bw) and (pd[i, j] <= h + bw):
                    Z.append((P[i, 2] - P[j, 2])**2.0)
        return np.sum(Z) / (2.0 * len(Z))

    def semivvar(self, P, hs, bw):
        '''
        Experimental variogram for a collection of lags
        '''
        sv = list()
        for h in hs:
            sv.append(self.semivvarh(P, h, bw))
        sv = [[hs[i], sv[i]] for i in range(len(hs)) if sv[i] > 0]
        return np.array(sv).T

    def c(self, P, h, bw):
        '''
        Calculate the sill
        '''
        c0 = np.var(P[:, 2])
        if h == 0:
            return c0
        return c0 - self.semivvarh(P, h, bw)
