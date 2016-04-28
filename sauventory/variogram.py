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
from math import exp
# Configure logger.
logger = logging.getLogger('semivariogram')


class Variogram(object):
    """Semivariogram class

    The class methods are influenced by work of Johnson Connor:
    http://connor-johnson.com/2014/03/20/simple-kriging-in-python/
    """

    def __init__(self):
        pass

    def semivarh(self, P, h, bw):
        """Experimental semivariogram for a single lag.

        Keyword arguments:
            P    Input data array n x (x , y, v) with n number of rows with two
                 dimensional coordinates (x, y) and corresponding value (v) for
                 local observation.
            h    Lag distance
            bw    Bandwise

        Returns:
            Semivariance
        """
        pd = squareform(pdist(P[:, :2]))
        N = pd.shape[0]
        Z = list()
        for i in range(N):
            for j in range(i+1, N):
                if(pd[i, j] >= h) and (pd[i, j] < h + bw):
                    Z.append((P[i, 2] - P[j, 2])**2.0)
        # Calculate semivariance.
        nnan = len(np.invert(np.isnan(Z)))
        sv = np.nansum(Z) / (2.0 * nnan)

        return(sv, nnan)

    def covarh(self, P, h, bw):
        """Experimental covariance for a single lag.

        Keyword arguments:
            P    Input data array n x (x , y, v) with n number of rows with two
                 dimensional coordinates (x, y) and corresponding value (v) for
                 local observation.
            h    Lag distance
            bw    Bandwise

        Returns:
            Semivariance
        """
        pd = squareform(pdist(P[:, :2]))
        N = pd.shape[0]
        T = list()  # Tail values f(i)
        H = list()  # Head values f(i+h)
        for i in range(N):
            for j in range(i+1, N):
                if(pd[i, j] >= h) and (pd[i, j] < h + bw):
                    T.append(P[i, 2])
                    H.append(P[j, 2])
        # Calculate covariance.
        nnan = len(np.invert(np.isnan(T)))
        # Get covariance matrix
        covmat = np.cov(np.array([T, H]))
        # Extract covariance
        cov = covmat[0, -1]

        return(cov, nnan)

    def semivar(self, P, hs, bw):
        """
        Experimental variogram for a collection of lags
        """
        sv = list()
        n = list()
        for h in hs:
            svh, nh = self.semivarh(P, h, bw)
            sv.append(svh)
            n.append(nh)
        sv = [[hs[i], sv[i], n[i]] for i in range(len(hs)) if sv[i] > 0]
        return np.array(sv).T

    def covar(self, P, hs, bw):
        """
        Experimental covariance function for a collection of lags
        """
        cov = list()
        n = list()
        for h in hs:
            covh, nh = self.covarh(P, h, bw)
            cov.append(covh)
            n.append(nh)
        cov = [[hs[i], cov[i], n[i]] for i in range(len(hs))]

        return np.array(cov).T

    def c(self, P, h, bw):
        """Calculate the sill"""
        c = np.nanvar(P[:, 2])
        if h == 0:
            return c
        else:
            return c - self.semivarh(P, h, bw)

    def c0(self, P, bw):
        """Calculate the nugget"""
        c0, n = self.semivarh(P, 0, bw)

        return c0

    def opt(self, fct, x, y, c, c0, parameterRange=None, meshSize=1000):
        if parameterRange is None:
            parameterRange = [x[1], x[-1]]
        mse = np.zeros(meshSize)
        a = np.linspace(parameterRange[0], parameterRange[1], meshSize)
        for i in range(meshSize):
            mse[i] = np.mean((y - fct(x, a[i], c, c0))**2.0)
        return a[mse.argmin()]

    def nugget(self, h, *args):
        """ Simple nugget model for the semivariogram

        Keyword arguments:
            h    lag distance
            c    Sill value
            c0    Nugget value
        """
        if len(args) != 2:
            c = args[1]
            c0 = args[2]
        else:
            c = args[0]
            c0 = args[1]
        # if h is a single digit
        if type(h) in [np.float64, float]:
            if h == 0:
                return c0
            else:
                return c
        # if h is an iterable
        else:
            # Calcualte the spherical function for all elements
            c = np.ones(h.size) * c
            c0 = np.ones(h.size) * c0
            return map(self.nugget, h, c, c0)

    def spherical(self, h, *args):
        """ Spherical model of the semivariogram

        Keyword arguments:
            h    lag distance
            a    (practical) range
            c    Sill value
            """
        # if h is a single digit
        a = args[0]
        c = args[1]
        # TODO: Check performance of float type.
        if type(h) in [np.float64, float]:
            # Calculate the spherical function
            if h <= a:
                return c*(1.5*h/a - 0.5*(h/a)**3.0)
            else:
                return c
        # if h is an iterable
        else:
            # Calcualte the spherical function for all elements
            a = np.ones(h.size) * a
            c = np.ones(h.size) * c
            return map(self.spherical, h, a, c)

    def gaussian(self, h, *args):
        """ Gaussian model of the semivariogram

        Keyword arguments:
            h    lag distance
            a    (practical) range
            c    Sill value
            """
        # if h is a single digit
        a = args[0]
        c = args[1]
        # TODO: Check performance of float type.
        if type(h) in [np.float64, float]:
            # Calculate the gaussian function
            return c*(1. - exp((-3.*h**2.)/a**2.))
        # if h is an iterable
        else:
            # Calcualte the gaussian function for all elements
            a = np.ones(h.size) * a
            c = np.ones(h.size) * c
            return map(self.gaussian, h, a, c)

    def exponential(self, h, *args):
        """ Exponential model of the semivariogram

        Keyword arguments:
            h    lag distance
            a    (practical) range
            c    Sill value
            """
        # if h is a single digit
        a = args[0]
        c = args[1]
        # TODO: Check performance of float type.
        if type(h) in [np.float64, float]:
            # Calculate the exponential function
            return c*(1. - exp((-3.*h)/a))
        # if h is an iterable
        else:
            # Calcualte the exponential function for all elements
            a = np.ones(h.size) * a
            c = np.ones(h.size) * c
            return map(self.exponential, h, a, c)

    def cvmodel(self, P, hs, bw, model):
        """ Model semivariances with specific functions.

        Keyword arguments:
            P    ndarray with data
            hs     ndarray with distances
            bw     Bandwidth - hs +- bw / 2
            model  Optional modeling function: spherical, exponential,
                   or gaussian. Default is spherical.

        Return:
            covfct    Covariance function
            sv    semivariogram
            c0    sill value
        """
        if not callable(model):
            msg = "ERROR: Variogram model type is not callable"
            raise RuntimeError(msg)
        # Calculate the semivariogram
        sv = self.semivar(P, hs, bw)
        # Calculate the sill
        c = self.c(P, hs[0], bw)
        # Calculate the nugget
        c0 = self.c0(P, bw)
        # Calculate the optimal parameters
        param = self.opt(model, sv[0], sv[1], c, c0)
        args = (param, c, c0)
        # Return a covariance function
        covfct = lambda h: model(h, *args)

        return covfct, sv, c, c0
