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
This module perform unittests for the rank transformation module.

Testing the module with example from Iman & Conover 1981:

https://www.uio.no/studier/emner/matnat/math/STK4400/v05/undervisningsmateriale
/A%20distribution-free%20approach%20to%20rank%20correlation.pdf
"""

import numpy as np
import unittest

import ranktransform


class Test(unittest.TestCase):

    def setUp(self):

        # Matrix R with n independent sampeld columns k,  R = kxn.
        self.r = np.array([[1.534, 1.534, -1.534, -1.534, .489, -.319],
                          [-.887, -.489, .887, -.887, -.157, .674],
                          [-.489, .674, -.489, 1.150, 1.534, -.489],
                          [.887, 0.000, -.674, .319, 0.000, -1.534],
                          [1.150, -.319, .489, .674, .157, 1.150],
                          [.157, -1.534, -.887, -.674, -.319, .157],
                          [-1.150, -.674, -.157, .157, -1.534, -.157],
                          [0.000, -.887, .157, -.319, -.674, .887],
                          [.319, -.157, .674, .887, .574, 1.534],
                          [-.319, .157, -.319, -1.150, 1.150, -.887],
                          [-1.534, .887, 1.150, 1.534, -.489, -1.150],
                          [-.157, -1.150, 1.534, -.157, -1.150, -.674],
                          [.489, .489, -1.150, .489, -.887, 0.000],
                          [.674, .319, .319, 0.000, .887, .319],
                          [-.674, 1.150, 0.000, -.489, .319, .489]])

        # Example target correlation matrix.
        self.c_star = np.array([[1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 1, .75, -.70],
                               [0, 0, 0, .75, 1, -.95],
                               [0, 0, 0, -.70, -.95, 1]])
        # Result sample columns arranged to given correlation amtrix.
        self.res = np.array([[1.534,  1.534, -1.534, -1.534, -0.887,  0.489],
                            [-0.887, -0.489,  0.887, -0.887, -0.674,  1.15],
                            [-0.489,  0.674, -0.489,  1.15,  1.534, -1.534],
                            [0.887,  0., -0.674, 0.319, 0.319, -0.887],
                            [1.15, -0.319, 0.489, 0.674, 0.574, -0.319],
                            [0.157, -1.534, -0.887, -0.674, -0.489, 0.674],
                            [-1.15, -0.674, -0.157, 0.157, -1.534, 0.887],
                            [0., -0.887, 0.157, -0.319, -0.319, 1.534],
                            [0.319, -0.157,  0.674,  0.887,  1.15, -0.674],
                            [-0.319, 0.157, -0.319, -1.15, 0.157, -0.157],
                            [-1.534, 0.887, 1.15, 1.534, 0.887, -1.15],
                            [-0.157, -1.15, 1.534, -0.157, -1.15, 0.319],
                            [0.489, 0.489, -1.15, 0.489, -0.157, 0.],
                            [0.674, 0.319, 0.319, 0., 0.489, -0.489],
                            [-0.674, 1.15, 0., -0.489, 0., 0.157]])

    def tearDown(self):
        pass

    def test_conover(self):
        r_cor = ranktransform.transform_by_corrmat(self.r, self.c_star)
        compare = r_cor == self.res
        self.assertEqual(compare.all(), True)


if __name__ == "__main__":
    unittest.main()
