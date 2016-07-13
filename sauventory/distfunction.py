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
This module defines classes with step function support. On this basis empirical
cumulative distribution functions can be created, imported and handed over to
Monte Carlo algorithms.
"""

import numpy as np
import scipy
import scipy.stats


class StepFunction(object):
    """
    The StepFunction class provide basic functionality to create, handle and
    convert stepwise defined functions.

    This base class allows contruction of statistical meassures, e. g.
    empirical cumulative distribution functions (ECDF).

    Keyword Arguments:
        x, y    Array-like objects.
        zval    Value for values left of x[0].
        sorted    Boolean, if input is sorted. Default is False.
        side    {'left', 'right'} Shape of the input to construct the steps.
                'right' correspond to [a, b) intervals and 'left' to (a, b]
    """

    def __init__(self, x, y, zval=0, sort=False, side='left'):
        """ Constructor """
        if side.lower() not in {'left', 'right'}:
            msg = "'side' can only take values 'left and 'right'"
            raise ValueError(msg)
        self.side = side

        _x = np.asarray(x)
        _y = np.asarray(y)

        if _x.shape != _y.shape:
            msg = "x and y should have the same shape."
            raise ValueError(msg)

        if len(_x.shape) != 1:
            msg = "x and y must be 1-dimensional."
            raise ValueError(msg)

        self.x = np.r_[-np.inf, _x]
        self.y = np.r_[zval, _y]

        if not sort:
            asort = np.argsort(self.x)
            self.x = np.take(self.x, asort, 0)
            self.y = np.take(self.y, asort, 0)
        self.n = self.x.shape[0]

    def __call__(self, val):
        """ Instance call assign index of step to given value.
        """
        index = np.searchsorted(self.x, val, self.side) - 1
        return(self.y[index])


class ECDF(StepFunction):
    """ Return empirical cumulative distribution function

    This function is fitting a step function to given array that will
    represent the probabilities of provided values.

    Keyword arguments:
        x    Array like object
        side    {'left', 'right'} Shape of the input to construct the steps.
                'right' correspond to [a, b) intervals and 'left' to (a, b]

    Returns:
        Empirical CDF as stepwise function
    """
    def __init__(self, x, side='right'):
        """ Constructor """
        _x = np.asarray(x)
        _x.sort()
        n = len(_x)
        _y = np.linspace(1./n, 1, n)
        super(ECDF, self).__init__(_x, _y, side=side, sort=True)


# The following functions are taken from geostatsmodles package from
# Connor Johnson:
# https://github.com/cjohnson318/geostatsmodels/blob/master/
# geostatsmodels/zscoretrans.py

def cdf(d):
    '''
    Input:  (d)    iterable, a data set
    Output: (f)    NumPy array with (bins) rows and two columns
                   the first column are values in the range of (d)
                   the second column are CDF values
                   alternatively, think of the columns as the
                   domain and range of the CDF function
            (finv) inverse of (f)
    ---------------------------------------------------------
    Calculate the cumulative distribution function
    and its inverse for a given data set
    '''
    # the number of data points
    N = float(len(d))
    # sorted array of data points
    xs = np.sort(d)
    # array of unique data points
    xu = np.unique(xs)
    # number of unique data points
    U = len(xu)
    # initialize an array of U zeros
    cdf = np.zeros((U))
    # for each unique data point..
    for i in range(U):
        # count the number of points less than
        # this point, and then divide by the
        # total number of data points
        cdf[i] = len(xs[xs < xu[i]]) / N
    # f : input value --> output percentage describing
    # the number of points less than the input scalar
    # in the modeled distribution; if 5 is 20% into
    # the distribution, then f[5] = 0.20
    f = np.vstack((xu, cdf)).T
    # inverse of f
    # finv : input percentage --> output value that
    # represents the input percentage point of the
    # distribution; if 5 is 20% into the distribution,
    # then finv[0.20] = 5
    finv = np.fliplr(f)
    return f, finv


def fit(d):
    '''
    Input:  (d) NumPy array with two columns,
                a domain and range of a mapping
    Output: (f) function that interpolates the mapping d
    ----------------------------------------------------
    This takes a mapping and returns a function that
    interpolates values that are missing in the original
    mapping, and maps values outside the range* of the
    domain (d) to the maximum or minimum values of the
    range** of (d), respectively.
    ----------------------------------------------------
    *range - here, I mean "maximum minus minimum"
    **range - here I mean the output of a mapping
    '''
    x, y = d[:, 0], d[:, 1]

    def f(t):
        # return the minimum of the range
        if t <= x.min():
            return y[np.argmin(x)]
        # return the maximum of the range
        elif t >= x.max():
            return y[np.argmax(x)]
        # otherwise, interpolate linearly
        else:
            intr = scipy.interpolate.interp1d(x, y)
            return intr(t)
    return f


def to_norm(data):
    '''
    Input  (data) 1D NumPy array of observational data
    Output (z)    1D NumPy array of z-score transformed data
           (inv)  inverse mapping to retrieve original distribution
    '''
    # look at the dimensions of the data
    dims = data.shape
    # if there is more than one dimension..
    if len(dims) > 1:
        # take the third column of the second dimension
        z = data[:, 2]
    # otherwise just use data as is
    else:
        z = data
    # grab the number of data points
    N = len(z)
    # grab the cumulative distribution function
    f, inv = cdf(z)
    # h will return the cdf of z
    # by interpolating the mapping f
    h = fit(f)
    # ppf will return the inverse cdf
    # of the standard normal distribution
    ppf = scipy.stats.norm(0, 1).ppf
    # for each data point..
    for i in range(N):
        # h takes z to a value in [0,1]
        p = h(z[i])
        # ppf takes p (in [0,1]) to a z-score
        z[i] = ppf(p)
    # convert positive infinite values
    posinf = np.isposinf(z)
    z = np.where(posinf, np.nan, z)
    z = np.where(np.isnan(z), np.nanmax(z), z)
    # convert negative infinite values
    neginf = np.isneginf(z)
    z = np.where(neginf, np.nan, z)
    z = np.where(np.isnan(z), np.nanmin(z), z)
    # if the whole data set was passed, then add the
    # transformed variable and recombine with data
    if len(dims) > 1:
        z = np.vstack((data[:, :2].T, z)).T
    return z, inv


def from_norm(data, inv):
    '''
    Input:  (data) NumPy array of zscore data
            (inv)  mapping that takes zscore data back
                   to the original distribution
    Output: (z)    Data that should conform to the
                   distribution of the original data
    '''
    # convert to a column vector
    d = data.ravel()
    # create an interpolating function
    # for the inverse cdf, mapping zscores
    # back to the original data distribution
    h = fit(inv)
    # convert z-score data to cdf values in [0,1]
    f = scipy.stats.norm(0, 1).cdf(d)
    # use inverse cdf to map [0,1] values to the
    # original distribution, then add the mu and sd
    z = np.array([h(i) for i in f])
    # reshape the data
    z = np.reshape(z, data.shape)

    return z

if __name__ == '__main__':
    pass
