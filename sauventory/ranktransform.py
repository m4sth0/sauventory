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
This module is able to perform rank transformation for
Monte Carlo sampling. Matrices with independent random samples for any type of
distribution are rearranged to fit to a given correlation matrix.

The methodology is adapted from Iman & Conover 1981.

https://www.uio.no/studier/emner/matnat/math/STK4400/v05/undervisningsmateriale
/A%20distribution-free%20approach%20to%20rank%20correlation.pdf
"""

import numpy as np
import scipy.stats as sstats
import logging

# Configure logger.
logger = logging.getLogger('ranktransform')
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - '
                           '%(message)s', level=logging.INFO)


def sort_rank(array, rarray):
    """ This function sort an array by specific ranks.
    Args:
        array  numpy array that will be sorted by ranks.
                The columns represent n number of independent samples
                for k variables.

        rarray  array of same length whose ranks are used for sorting.
    """
    # Get array ranks.
    array_rank = np.apply_along_axis(sstats.rankdata, 0, array)
    rarray_rank = np.apply_along_axis(sstats.rankdata, 0, rarray)
    # Get index list for array based on given ranks.
    indexlist = [np.where(array_rank == i)[0][0].astype(int)
                 for i in rarray_rank]
    # Order array by rank list.
    sorted_array = array[indexlist]

    return(sorted_array)


def transform_by_corrmat(array, cormat):
    """ The function utilize an given correlation matrix to rearrange a data
    matrix to fullfill the covariant relationships.

    Args:
        array  Matrix (R) with n independent sampeld of k variables,  R = kxn
        cormat  Target Spearman rank correlation matrix (C*) as numpy array.
    """
    # Calculate lower trianglular matrix by cholesky factorization of target
    # correlation matrix. C = C* = PP' (C must be positive definite and
    # symmetric)
    lowtri = np.linalg.cholesky(cormat)

    # Calculate the sample correlation matrix T for the independent sampled
    # data.
    t = np.corrcoef(array, rowvar=0)
    logger.debug("Sample correlation matrix T for R: %s" % (t))
    # Calculate R* matrix with multivariate distribution close to C by matrix
    # multiplication of R with transposed lower triangular covariance matrix
    # R* = RP'.
    array_cor = np.dot(array, np.transpose(lowtri))

    # Calculate spearman rank correlation matrix M of R* matrix.
    m = sstats.spearmanr(array_cor)
    logger.debug("Spearman rank correlation matrix M for R*: %s" % (m[0]))

    # Use rank sort function to rearrange input matrix R by calculated R*
    # matrix.
    sorted_array = [sort_rank(array[:, i], array_cor[:, i])
                    for i in range(array.shape[1])]

    result = np.transpose(np.vstack(tuple(sorted_array)))

    # Calculate spearman rank correlation matrix M* for rearranged R matrix.
    m_star = sstats.spearmanr(result)

    logger.info("Spearman rank correlation matrix M*")
    logger.debug("Spearman rank correlation matrix M* for R:\n %s" %
                 (m_star[0]))

    return(result)
