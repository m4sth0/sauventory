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
This module perform unittests for the inventory module.

Testing the module with example the agricultural N2O inventory for 2012:

https://www.umweltbundesamt.de/sites/default/files/medien/378/publikationen/
climate-change_24_2014_nationaler_inventorybericht_0.pdf
"""

import unittest
import numpy as np
import os
import time

import inventory
import spatialinventory


class Test(unittest.TestCase):

    def setUp(self):

        # N2O inventory of agricultural source categories for 2012.
        self.n2o_inv = [958.4, 1092.7, 497.8, 42.2, 7.3, 135.9, 1.7, 0.8,
                        51.1, 9539.9, 4693.5, 472.6, 6171.3, 4750.5, 1315,
                        2213.5, 11596.4, 162.8]

        # Corresponding list of source category descriptions.
        self.n2o_index = ["Manure management, dairy cows",
                          "Manure management, other cattle",
                          "Manure management, pigs",
                          "Manure management, sheep",
                          "Manure management, goats",
                          "Manure management, horses",
                          "Manure management, mules, asses",
                          "Manure management, buffalo",
                          "Manure management, poultry",
                          "Soils, mineral fertilizers",
                          "Soils, application of manure",
                          "Soils, N fixing crops",
                          "Soils, crop residues",
                          "Soils, organic soils",
                          "Soils, grazing",
                          "Soils, indirect emissions (deposition)",
                          "Soils, indirect emissions (leaching, run-off)",
                          "Soils, sewage sludge emissions"]
        # Uncertainty of inventory in %, half the 95 % confidence interval.
        self.n2o_percent = [100.1, 100.1, 100.1, 300.2, 300.7, 300.2, 316.2,
                            100.5, 100.5, 80, 100, 94.3, 94.3, 200, 201, 111.8,
                            416.3, 82.5]

        # Convert to absolute values in Gg.
        self.n2o_uncert = [a*b/100 for a, b in zip(self.n2o_inv,
                                                   self.n2o_percent)]
        self.n2o_inv_uncert = np.sqrt(np.sum(map(np.square, self.n2o_uncert)))

        # Hypothetic covariance matrix for N2O emissions.
        l = len(self.n2o_percent)
        self.n2o_covmat = np.zeros(shape=(l, l))
        np.fill_diagonal(self.n2o_covmat, np.square(self.n2o_uncert))
        indu = np.triu_indices_from(self.n2o_covmat, 1)
        indl = np.tril_indices_from(self.n2o_covmat, -1)
        # Calculate covariances for an assumed correlation coefficientof 0.5.
        self.n2o_covmat[indu] = 0.5 * np.sqrt(self.n2o_covmat[(indu[0],
                                                               indu[0])] *
                                              self.n2o_covmat[(indu[1],
                                                               indu[1])])
        self.n2o_covmat[indl] = 0.5 * np.sqrt(self.n2o_covmat[(indl[0],
                                                               indl[0])] *
                                              self.n2o_covmat[(indl[1],
                                                               indl[1])])
        self.n2ocovsum = np.sqrt(self.n2o_covmat.sum())
        self.n2odiagsum = np.sqrt(np.sum(np.diag(self.n2o_covmat)))
        # Setup test raster file names and location.
        cwd = os.getcwd()
        self.invin = cwd + "/data/model_peat_examp_1.tiff"
        self.uncertin = cwd + "/data/uncert_peat_examp_1.tiff"

        # Setup test vector file names and location.
        self.invvector = cwd + "/data/n2o_eu_2010_inventory/" \
                               "n2o_eu_2010_inventory.shp"
        # Data source: http://ec.europa.eu/eurostat/statistics-explained/
        # index.php/Agri-environmental_indicator_-_greenhouse_gas_emissions

    def tearDown(self):
        pass

    def test_inventory_dict(self):
        i = inventory.Inventory("N2O-Agrar-2012", "Gg",
                                "German N2O inventory of agricultural source "
                                "categories for 2012",
                                creator="Tester")

        i.import_inventory(self.n2o_inv, self.n2o_uncert,
                           self.n2o_index)

        self.assertEqual(round(sum(i.inv_dict.values()), 1), 43703.4)
        self.assertEqual(round(sum(i.uncert_dict.values()), 1),
                         round(sum(self.n2o_uncert), 1))
        self.assertEqual(len(i.inv_dict), 18)

    def test_inventory_dict_relative(self):
        i = inventory.Inventory("N2O-Agrar-2012", "Gg",
                                "German N2O inventory of agricultural source "
                                "categories for 2012",
                                ("2012-01-01 00:00:00", "2013-01-01 00:00:00"),
                                creator="Tester")

        i.import_inventory(self.n2o_inv, self.n2o_percent,
                           self.n2o_index, True)

        self.assertEqual(round(sum(i.inv_dict.values()), 1), 43703.4)
        self.assertEqual(round(sum(i.uncert_dict.values()), 1),
                         round(sum(self.n2o_uncert), 1))
        self.assertEqual(len(i.inv_dict), 18)

    def test_timestamp(self):
        i = inventory.Inventory("N2O-Agrar-2012", "Gg",
                                "German N2O inventory of agricultural source "
                                "categories for 2012",
                                ("2012-01-01 00:00:00", None),
                                creator="Tester")
        start, end = i.timestamp
        self.assertEqual((start, end),
                         ("2012-01-01 00:00:00", 'None'))
        i.timestamp = (None, "2013-01-01 00:00:00")
        start, end = i.timestamp
        self.assertEqual((start, end),
                         ('None', "2013-01-01 00:00:00"))
        i.timestamp = ("2012-01-01 00:00:00", "2013-01-01 00:00:00")
        start, end = i.timestamp
        self.assertEqual((start, end),
                         ("2012-01-01 00:00:00", "2013-01-01 00:00:00"))

    def test_mtime_dict(self):
        i = inventory.Inventory("N2O-Agrar-2012", "Gg",
                                "German N2O inventory of agricultural source "
                                "categories for 2012",
                                ("2012-01-01 00:00:00", None),
                                creator="Tester")
        time.sleep(1)
        i.import_inventory(self.n2o_inv, self.n2o_percent,
                           self.n2o_index, True)
        self.assertNotEqual(i.ctime, i.mtime)

    def test_inventory_array(self):
        i = inventory.Inventory("N2O-Agrar-2012", "Gg",
                                "German N2O inventory of agricultural source "
                                "categories for 2012",
                                creator="Tester")
        n2o_inv_array = np.array(self.n2o_inv).reshape((3, 6))
        n2o_inv_index = np.array(self.n2o_index).reshape((3, 6))
        n2o_inv_uncert = np.array(self.n2o_percent).reshape((3, 6))

        i.import_inventory(n2o_inv_array, n2o_inv_uncert,
                           n2o_inv_index, True)

        self.assertEqual(np.sum(i.inv_array), 43703.4)
        self.assertEqual(round(np.sum(i.inv_uncert_array), 0),
                         round(sum(self.n2o_uncert), 0))
        self.assertEqual(i.inv_array.shape, (3, 6))

    def test_dict_accumulate(self):
        i = inventory.Inventory("N2O-Agrar-2012", "Gg",
                                "German N2O inventory of agricultural source "
                                "categories for 2012",
                                creator="Tester")

        i.import_inventory(self.n2o_inv, self.n2o_uncert,
                           self.n2o_index)
        i.accumulate()
        self.assertEqual(round(i.inv_sum, 1), round(np.sum(self.n2o_inv), 1))

    def test_array_accumulate(self):
        i = inventory.Inventory("N2O-Agrar-2012", "Gg",
                                "German N2O inventory of agricultural source "
                                "categories for 2012",
                                creator="Tester")

        n2o_inv_array = np.array(self.n2o_inv).reshape((3, 6))
        n2o_inv_index = np.array(self.n2o_index).reshape((3, 6))
        n2o_inv_uncert = np.array(self.n2o_percent).reshape((3, 6))

        i.import_inventory(n2o_inv_array, n2o_inv_uncert,
                           n2o_inv_index, True)
        i.accumulate()
        self.assertEqual(round(i.inv_sum, 1), round(np.sum(self.n2o_inv), 1))

    def test_dict_propagate(self):
        i = inventory.Inventory("N2O-Agrar-2012", "Gg",
                                "German N2O inventory of agricultural source "
                                "categories for 2012",
                                creator="Tester")

        i.import_inventory(self.n2o_inv, self.n2o_uncert,
                           self.n2o_index)
        i.propagate()
        self.assertEqual(round(i.inv_uncert, 1), round(self.n2o_inv_uncert, 1))

    def test_dict_propagate_cv(self):
        i = inventory.Inventory("N2O-Agrar-2012", "Gg",
                                "German N2O inventory of agricultural source "
                                "categories for 2012",
                                creator="Tester")

        i.import_inventory(self.n2o_inv, self.n2o_uncert,
                           self.n2o_index)
        i.inv_covmat = self.n2o_covmat
        i.propagate(cv=True)
        self.assertEqual(round(i.inv_uncert, 1), round(self.n2ocovsum, 1))

    def test_array_propagate(self):
        i = inventory.Inventory("N2O-Agrar-2012", "Gg",
                                "German N2O inventory of agricultural source "
                                "categories for 2012",
                                creator="Tester")

        n2o_inv_array = np.array(self.n2o_inv).reshape((3, 6))
        n2o_inv_index = np.array(self.n2o_index).reshape((3, 6))
        n2o_inv_uncert = np.array(self.n2o_percent).reshape((3, 6))

        i.import_inventory(n2o_inv_array, n2o_inv_uncert,
                           n2o_inv_index, True)
        i.propagate()
        self.assertEqual(round(i.inv_uncert, 1), round(self.n2o_inv_uncert, 1))

    def test_array_propagate_cv(self):
        i = inventory.Inventory("N2O-Agrar-2012", "Gg",
                                "German N2O inventory of agricultural source "
                                "categories for 2012",
                                creator="Tester")

        n2o_inv_array = np.array(self.n2o_inv).reshape((3, 6))
        n2o_inv_index = np.array(self.n2o_index).reshape((3, 6))
        n2o_inv_uncert = np.array(self.n2o_percent).reshape((3, 6))

        i.import_inventory(n2o_inv_array, n2o_inv_uncert,
                           n2o_inv_index, True)
        i.inv_covmat = self.n2o_covmat
        i.propagate(cv=True)
        self.assertEqual(round(i.inv_uncert, 1), round(self.n2ocovsum, 1))

    def test_dict_printsum(self):
        i = inventory.Inventory("N2O-Agrar-2012", "Gg",
                                "German N2O inventory of agricultural source "
                                "categories for 2012",
                                ("2012-01-01 00:00:00", "2013-01-01 00:00:00"),
                                creator="Tester")

        i.import_inventory(self.n2o_inv, self.n2o_uncert,
                           self.n2o_index)
        i.accumulate()
        i.propagate()
        i.printsum()
        self.assertEqual(round(i.inv_sum, 1), round(np.sum(self.n2o_inv), 1))
        self.assertEqual(round(i.inv_uncert, 1), round(self.n2o_inv_uncert, 1))

    def test_array_printsum(self):
        i = inventory.Inventory("N2O-Agrar-2012", "Gg",
                                "German N2O inventory of agricultural source "
                                "categories for 2012",
                                creator="Tester")

        n2o_inv_array = np.array(self.n2o_inv).reshape((3, 6))
        n2o_inv_index = np.array(self.n2o_index).reshape((3, 6))
        n2o_inv_uncert = np.array(self.n2o_percent).reshape((3, 6))

        i.import_inventory(n2o_inv_array, n2o_inv_uncert,
                           n2o_inv_index, True)
        i.inv_covmat = self.n2o_covmat
        i.accumulate()
        i.propagate(cv=True)
        i.printsum()
        self.assertEqual(round(i.inv_uncert, 1), round(self.n2ocovsum, 1))
        self.assertEqual(round(i.inv_sum, 1), round(np.sum(self.n2o_inv), 1))

    def test_inventory_raster_import(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_raster(self.invin, self.uncertin)
        self.assertEqual(si.inv_array.shape, (44, 55))
        self.assertEqual(si.inv_uncert_array.shape, (44, 55))
        self.assertEqual(round(np.nanmin(si.inv_array), 3), 0.071)
        self.assertEqual(round(np.nanmax(si.inv_array), 3), 1.728)
        self.assertEqual(round(np.nanmin(si.inv_uncert_array), 3), 0.269)
        self.assertEqual(round(np.nanmax(si.inv_uncert_array), 3), 1.387)

    def test_inventory_raster_accumulate(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_raster(self.invin, self.uncertin)
        self.assertEqual(round(si.accumulate(), 2), 1552.0)

    def test_inventory_raster_propagate(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_raster(self.invin, self.uncertin)
        si.propagate()
        self.assertEqual(round(si.inv_uncert, 2), 45.75)

    def test_inventory_raster_propagate_cv(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_raster(self.invin, self.uncertin)
        si.get_cov_matrix()
        si.propagate(cv=True)
        self.assertEqual(round(si.inv_uncert, 2), 932.65)

    def test_mtime_raster(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")
        time.sleep(1)
        si.import_inventory_as_raster(self.invin, self.uncertin)

        self.assertNotEqual(si.ctime, si.mtime)

    def test_inventory_vector_import(self):
        si = spatialinventory.VectorInventory("N2O-Agrar-EU-2010", "Gg",
                                              "N2O inventory for EU-27"
                                              "emissions from agriculture",
                                              ("2010-01-01 00:00:00",
                                               "2011-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_vector(self.invvector, 'n2o_Gg',
                                      uncert='uncert_Gg', index='NUTS_ID',
                                      relative=True)
        self.assertEqual(round(np.nanmin(si.inv_array), 3), 848.0)
        self.assertEqual(round(np.nanmax(si.inv_array), 3), 51690.0)
        self.assertEqual(round(np.nanmin(si.inv_uncert_array), 3), 1272.0)
        self.assertEqual(round(np.nanmax(si.inv_uncert_array), 3), 77535.0)

    def test_inventory_vector_import_nouncert(self):
        si = spatialinventory.VectorInventory("N2O-Agrar-EU-2010", "Gg",
                                              "N2O inventory for EU-27"
                                              "emissions from agriculture",
                                              ("2010-01-01 00:00:00",
                                               "2011-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_vector(self.invvector, 'n2o_Gg',
                                      index='NUTS_ID',
                                      relative=True)
        self.assertEqual(round(np.nanmin(si.inv_array), 3), 848.0)
        self.assertEqual(round(np.nanmax(si.inv_array), 3), 51690.0)
        self.assertEqual(str(np.nanmin(si.inv_uncert_array)), 'nan')
        self.assertEqual(str(np.nanmax(si.inv_uncert_array)), 'nan')

    def test_inventory_vector_accumulate(self):
        si = spatialinventory.VectorInventory("N2O-Agrar-EU-2010", "Gg",
                                              "N2O inventory for EU-27"
                                              "emissions from agriculture",
                                              ("2010-01-01 00:00:00",
                                               "2011-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_vector(self.invvector, 'n2o_Gg',
                                      uncert='uncert_Gg', index='NUTS_ID',
                                      relative=True)
        si.accumulate()
        self.assertEqual(round(si.inv_sum, 3), round(np.nansum(si.inv_array),
                                                     3))

    def test_inventory_vector_propagate(self):
        si = spatialinventory.VectorInventory("N2O-Agrar-EU-2010", "Gg",
                                              "N2O inventory for EU-27"
                                              "emissions from agriculture",
                                              ("2010-01-01 00:00:00",
                                               "2011-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_vector(self.invvector, 'n2o_Gg',
                                      uncert='uncert_Gg', index='NUTS_ID',
                                      relative=True)
        si.propagate()
        self.assertEqual(round(si.inv_uncert, 3),
                         round(np.sqrt(np.nansum(map(np.square,
                                                     si.inv_uncert_array))),
                               3))

    def test_inventory_vector_propagate_cv(self):
        si = spatialinventory.VectorInventory("N2O-Agrar-EU-2010", "Gg",
                                              "N2O inventory for EU-27"
                                              "emissions from agriculture",
                                              ("2010-01-01 00:00:00",
                                               "2011-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_vector(self.invvector, 'n2o_Gg',
                                      uncert='uncert_Gg', index='NUTS_ID',
                                      relative=True)
        si.get_cov_matrix()
        si.propagate(cv=True)
        self.assertEqual(round(si.inv_uncert, 3), 137877.173)
        self.assertGreater(si.inv_uncert,
                           round(np.sqrt(np.nansum(map(np.square,
                                                       si.inv_uncert_array))),
                                 3))

if __name__ == "__main__":
    unittest.main()
