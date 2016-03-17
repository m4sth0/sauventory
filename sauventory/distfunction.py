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

if __name__ == '__main__':
    pass
