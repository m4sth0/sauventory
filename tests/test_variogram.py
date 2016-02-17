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
This module perform unittests for variogram functions within the
sauventory package.
"""
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import unittest

import spatialinventory
import variogram


class Test(unittest.TestCase):

    def setUp(self):
        self.r = np.array([random.randrange(1, 1000) for _ in range(0, 1000)])

        # Setup test raster file names and location.
        cwd = os.getcwd()
        self.invin = cwd + "/data/model_peat_examp_1.tiff"
        self.uncertin = cwd + "/data/uncert_peat_examp_1.tiff"

        # Setup test vector file names and location.
        self.invvector = cwd + "/data/n2o_eu_2010_inventory/" \
                               "n2o_eu_2010_inventory.shp"

    def test_single_variogram_raster(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_raster(self.invin, self.uncertin)
        v = variogram.Variogram()
        coords = si.get_coord()
        data = np.hstack((coords, si.inv_array.reshape((si.inv_array.size,
                                                        1))))
        sv1 = v.semivvarh(data, 5, 5)
        sv2 = v.semivvarh(data, 50, 5)
        self.assertEqual(round(sv1, 3), 0.168)
        self.assertEqual(round(sv2, 3), 0.217)

    def test_single_variogram_vector(self):
        si = spatialinventory.VectorInventory("N2O-Agrar-EU-2010", "Gg",
                                              "N2O inventory for EU-27"
                                              "emissions from agriculture",
                                              ("2010-01-01 00:00:00",
                                               "2011-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_vector(self.invvector, 'n2o_Gg',
                                      uncert='uncert_Gg', index='NUTS_ID',
                                      relative=True)
        v = variogram.Variogram()
        coords = si.get_coord()
        data = np.hstack((coords, si.inv_array.reshape((si.inv_array.size,
                                                        1))))
        sv1 = v.semivvarh(data, 10, 8)
        sv2 = v.semivvarh(data, 20, 8)
        self.assertEqual(round(sv1, 3), 136703201.471)
        self.assertEqual(round(sv2, 3), 145110190.277)

    def test_empirical_variogram_raster(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_raster(self.invin, self.uncertin)
        v = variogram.Variogram()
        coords = si.get_coord()
        data = np.hstack((coords, si.inv_array.reshape((si.inv_array.size,
                                                        1))))
        # Define variogram parameters
        bw = 10  # Bandwidth
        hs = np.arange(0, 80, bw)  # Distance intervals
        svario = v.semivvar(data, hs, bw)
        self.assertEqual(round(np.max(svario[1]), 3), 0.245)
        self.assertEqual(round(np.min(svario[1]), 3), 0.168)

    def test_empirical_variogram_vector(self):
        si = spatialinventory.VectorInventory("N2O-Agrar-EU-2010", "Gg",
                                              "N2O inventory for EU-27"
                                              "emissions from agriculture",
                                              ("2010-01-01 00:00:00",
                                               "2011-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_vector(self.invvector, 'n2o_Gg',
                                      uncert='uncert_Gg', index='NUTS_ID',
                                      relative=True)
        v = variogram.Variogram()
        coords = si.get_coord()
        data = np.hstack((coords, si.inv_array.reshape((si.inv_array.size,
                                                        1))))

        # Define variogram parameters
        bw = 10  # Bandwidth
        hs = np.arange(0, 80, bw)  # Distance intervals
        svario = v.semivvar(data, hs, bw)
        self.assertEqual(round(np.max(svario[1]), 3), 142924111.258)
        self.assertEqual(round(np.min(svario[1]), 3), 16520.533)

        # Plot variogram.
        plt.plot(svario[0], svario[1], '.-')
        plt.xlabel('Lag [m]')
        plt.ylabel('Semivariance')
        plt.title('Semivariogram for vector sample')
        axes = list(plt.axis())
        axes[2] = 0
        plt.axis(axes)
        plt.show()

    def test_get_variogram_raster(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_raster(self.invin, self.uncertin)
        sv, svm = si.get_variogram(10, 80, False)
        self.assertEqual(round(np.max(sv[1]), 3), 0.245)
        self.assertEqual(round(np.min(si.inv_sv[1]), 3), 0.168)

    def test_plot_variogram_raster(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_raster(self.invin, self.uncertin)
        sv, svm = si.get_variogram(10, 80, True)
        self.assertEqual(round(np.max(sv[1]), 3), 0.245)
        self.assertEqual(round(np.min(si.inv_sv[1]), 3), 0.168)
        si.plot_variogram()

    def test_spherical_variogram_raster(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_raster(self.invin, self.uncertin)
        v = variogram.Variogram()
        coords = si.get_coord()
        data = np.hstack((coords, si.inv_array.reshape((si.inv_array.size,
                                                        1))))
        # Define variogram parameters
        bw = 10  # Bandwidth
        hs = np.arange(0, 80, bw)  # Distance intervals
        # svario = v.semivvar(data, hs, bw)
        svmodel, svario = v.cvmodel(data, hs, bw, model=v.spherical)
        self.assertEqual(round(svmodel(svario[0][0]), 3), 0.)
        self.assertEqual(round(svmodel(svario[0][7]), 3), 0.340)

    """
    sp = cvmodel( P, model=spherical, hs=np.arange(0,10500,500), bw=500 )
    plot( sv[0], sv[1], '.-' )
    plot( sv[0], sp( sv[0] ) ) ;
    savefig('semivariogram_model.png',fmt='png',dpi=200)
    """
if __name__ == "__main__":
    unittest.main()
