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
This module perform unittests for supplementary functions within the
sauventory package.
"""

import numpy as np
import os
import unittest

import spatialinventory
import stepfunction as sf


class Test(unittest.TestCase):

    def setUp(self):
        # Set basic array
        self.x = np.arange(20)
        self.y = np.arange(20)
        # Sample from normal distribution.
        self.a = np.random.normal(1, 0.3, 10000)

        # Create artificial raster arrays.
        self.maxrast = np.concatenate((np.ones((1, 50)), np.zeros((1, 50))
                                       )).reshape(10, 10)
        self.nanrast = np.hstack((np.ones((1, 50)),
                                  np.zeros((1, 50)),
                                  np.empty((1, 10)) * np.nan)).reshape(11, 10)

        # Setup test raster file names and location.
        cwd = os.getcwd()
        self.invin = cwd + "/data/model_peat_examp_1.tiff"
        self.uncertin = cwd + "/data/uncert_peat_examp_1.tiff"

        # Setup test vector file names and location.
        self.invvector = cwd + "/data/n2o_eu_2010_inventory/"
        "n2o_eu_2010_inventory.shp"

    def test_stepfunction(self):
        f = sf.StepFunction(self.x, self.y)
        f2 = sf.StepFunction(self.x, self.y, side='right')

        self.assertEqual(f(5), 4)
        self.assertEqual(f2(5), 5)

    def test_ecdf(self):
        ecdf = sf.ECDF(self.a)
        self.assertAlmostEqual(ecdf(1), 0.5, 1)

    def test_weight_raster_queen(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")
        si.import_inventory(self.maxrast)
        w = si.get_weight_matrix(si.inv_array)
        self.assertEqual(w.n, 100)
        self.assertListEqual(w.neighbors[0], [10, 1, 11])
        self.assertListEqual(w.weights[0], [1.0, 1.0, 1.0])

    def test_weight_raster_rook(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")
        si.import_inventory(self.maxrast)
        w = si.get_weight_matrix(si.inv_array, rook=True)
        self.assertEqual(w.n, 100)
        self.assertListEqual(w.neighbors[0], [10, 1])
        self.assertListEqual(w.weights[0], [1.0, 1.0])

    def test_moran(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")
        si.import_inventory(self.maxrast)
        mi = si.check_moran()
        self.assertEqual(round(mi, 3), 0.848)

    def test_moran_nan(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")
        si.import_inventory(self.nanrast)
        mi = si.check_moran()
        self.assertEqual(round(mi, 3), 0.848)

    def test_moran_raster(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_raster(self.invin, self.uncertin)
        mi = si.check_moran()
        self.assertEqual(round(mi, 3), 0.603)

if __name__ == "__main__":
    unittest.main()
