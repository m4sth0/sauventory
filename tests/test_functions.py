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
import timeit
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
        self.invvector = cwd + "/data/n2o_eu_2010_inventory/" \
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

    def test_weight_vector_queen(self):
        si = spatialinventory.VectorInventory("N2O-Agrar-EU-2010", "Gg",
                                              "N2O inventory for EU-27"
                                              "emissions from agriculture",
                                              ("2010-01-01 00:00:00",
                                               "2011-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_vector(self.invvector, 'n2o_Gg',
                                      index='NUTS_ID')

        w = si.get_weight_matrix(si.inv_array)
        self.assertEqual(w.n, 27)
        self.assertListEqual(w.neighbors['DE'], [u'BE', u'AT', u'CZ', u'CH',
                                                 u'FR', u'NL', u'PL'])
        self.assertEqual([i for i in w.neighbors][4], 'DE')
        self.assertEqual(si.inv_array[4], 41628)
        self.assertEqual([i for i in w.neighbors][1], 'FR')
        self.assertEqual(si.inv_array[1], 51690)

    def test_weight_vector_rook(self):
        si = spatialinventory.VectorInventory("N2O-Agrar-EU-2010", "Gg",
                                              "N2O inventory for EU-27"
                                              "emissions from agriculture",
                                              ("2010-01-01 00:00:00",
                                               "2011-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_vector(self.invvector, 'n2o_Gg',
                                      index='NUTS_ID')
        # TODO: Queens' case histogram is not different to rook's case. Why?
        w = si.get_weight_matrix(si.inv_array, rook=True)
        self.assertEqual(w.n, 27)
        self.assertListEqual(w.neighbors['DE'], [u'BE', u'AT', u'CZ', u'CH',
                                                 u'FR', u'NL', u'PL'])
        self.assertEqual([i for i in w.neighbors][4], 'DE')
        self.assertEqual(si.inv_array[4], 41628)
        self.assertEqual([i for i in w.neighbors][1], 'FR')
        self.assertEqual(si.inv_array[1], 51690)

    def test_remove_nan_raster(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")
        si.import_inventory(self.nanrast)
        w = si.get_weight_matrix(si.inv_array)
        nw, na, n = si.rm_nan_weight(w, si.inv_array)
        self.assertEqual(w.n, 110)
        self.assertEqual(nw.n, 100)
        self.assertEqual(n, 10)

    def test_remove_nan_vector(self):
        si = spatialinventory.VectorInventory("N2O-Agrar-EU-2010", "Gg",
                                              "N2O inventory for EU-27"
                                              "emissions from agriculture",
                                              ("2010-01-01 00:00:00",
                                               "2011-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_vector(self.invvector, 'n2o_Gg',
                                      uncert='uncert_Gg', index='NUTS_ID',
                                      relative=True)
        w = si.get_weight_matrix(si.inv_array)
        nw, na, n = si.rm_nan_weight(w, si.inv_array)
        self.assertEqual(w.n, 27)
        self.assertEqual(nw.n, 22)
        self.assertEqual(n, 5)

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

    def test_moran_vector(self):
        si = spatialinventory.VectorInventory("N2O-Agrar-EU-2010", "Gg",
                                              "N2O inventory for EU-27"
                                              "emissions from agriculture",
                                              ("2010-01-01 00:00:00",
                                               "2011-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_vector(self.invvector, 'n2o_Gg',
                                      uncert='uncert_Gg', index='NUTS_ID',
                                      relative=True)
        mi = si.check_moran()
        self.assertEqual(round(mi, 3), 0.265)

    def test_get_variogram_raster(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_raster(self.invin, self.uncertin)
        sv, svm, c0 = si.get_variogram(10)
        self.assertEqual(round(np.max(sv[1]), 3), 0.228)
        self.assertEqual(round(np.min(si.inv_sv[1]), 3), 0.168)

    def test_plot_variogram_raster(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_raster(self.invin, self.uncertin)
        sv, svm, c0 = si.get_variogram(10, 80, True)
        self.assertEqual(round(np.max(sv[1]), 3), 0.245)
        self.assertEqual(round(np.min(si.inv_sv[1]), 3), 0.168)
        si.plot_variogram()

    def test_get_cov_matrix_raster(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_raster(self.invin, self.uncertin)
        si.get_cov_matrix()
        self.assertEqual(round(np.max(si.inv_covmat), 3), 0.34)
        self.assertEqual(round(np.min(si.inv_covmat), 3), 0.)
        self.assertEqual(round(si.inv_c0, 3), 0.34)

if __name__ == "__main__":
    unittest.main()
