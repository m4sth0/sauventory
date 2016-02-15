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

import itertools
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

        # part of our data set recording porosity
        #P = np.array( z[['x','y','por']] )
        # bandwidth, plus or minus 250 meters
        #bw = 500
        # lags in 500 meter increments from zero to 10,000
        #hs = np.arange(0,10500,bw)
        #sv = SV( P, hs, bw )
        #plot( sv[0], sv[1], '.-' )
        #xlabel('Lag [m]')
        #ylabel('Semivariance')
        #title('Sample Semivariogram') ;
        #savefig('sample_semivariogram.png',fmt='png',dpi=200)

    def test_basic_variogram(self):
        si = spatialinventory.RasterInventory("N2O-Agrar-2012", "g/m2",
                                              "Example N2O inventory of "
                                              "organic soils",
                                              ("2012-01-01 00:00:00",
                                               "2013-01-01 00:00:00"),
                                              creator="Tester")

        si.import_inventory_as_raster(self.invin, self.uncertin)
        v = variogram.Variogram()
        shp = si.inv_array.shape
        coords = itertools.product(range(shp[0]), range(shp[1]))
        acoords = np.array(map(np.asarray, coords))
        data = np.hstack((acoords, si.inv_array.reshape((si.inv_array.size, 1))))
        sv = v.semivvarh(data, 1, 1)
        print(sv)
        sv = v.semivvarh(data, 10, 5)
        print(sv)

if __name__ == "__main__":
    unittest.main()
