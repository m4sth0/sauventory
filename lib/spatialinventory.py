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
This module defines an spatial explicit inventory class that inherits the
basic functionalities from inventory class and adds additionally spatial
relevant functions to create, handle and calculate spatial explicit inventories
and corresponding spatial autocorrelated uncertainties.
"""


import numpy as np
import logging
import sys
from osgeo import gdal

from inventory import Inventory

# Configure logger.
logger = logging.getLogger('spatialinventory')


class SpatialInventory(Inventory):
    """Spatial explicit inventory class"""

    def __init__(self, *args, **kwargs):
        """
        Class instance constructor with additional arguments:

        For other parameters see: inventory.Inventory
        """
        super(SpatialInventory, self).__init__(*args, **kwargs)

    def check_moran(self):
        """ Get Moran's I statistic for georeferenced inventory"""

    def get_variogram(self):
        """Get variogram function for spatial inventory values"""

    def get_cov_matrix(self):
        """Create covariance matrix of spatial auto correlated inventory"""


class RasterInventory(SpatialInventory):
    """Spatial raster inventory class"""

    def __init__(self, *args, **kwargs):
        """
        Class instance constructor with additional arguments:

        For other parameters see: inventory.Inventory
        """
        super(RasterInventory, self).__init__(*args, **kwargs)

    def import_raster(self, infile, band=1):
        """ Import routine for rasterized geographical referenced data

        The import is handled by gdal package of osgeo.
        Keyword arguments:
            infile    Input file name and directory as string.
            band    Number of selected raster band.
        """
        # Open raster file.
        try:
            rfile = gdal.Open(infile)
        except ImportError:
            msg = 'Unable to open %s' % (infile)
            raise ImportError(msg)
        # Get selected raster band.
        try:
            rband = rfile.GetRasterBand(band)
        except ImportError, e:
            msg = 'No band %i found' % (band)
            raise ImportError(msg)

        except RuntimeError, e:
            # for example, try GetRasterBand(10)
            print 'Band ( %i ) not found' % band
            print e
            sys.exit(1)
        cols = rfile.RasterXSize
        rows = rfile.RasterYSize
        bands = rfile.RasterCount
        # Print imported raster summary.
        print "[ PROJECTION ] = ", rfile.GetProjectionRef()[7:].split(',')[0]
        print "[ ROWS ] = ", rows
        print "[ COLUMNS ] = ", cols
        print "[ BANDS ] = ", bands
        print "[ NO DATA VALUE ] = ", rband.GetNoDataValue()
        print "[ MIN ] = ", rband.GetMinimum()
        print "[ MAX ] = ", rband.GetMaximum()
        print "[ SCALE ] = ", rband.GetScale()
        print "[ UNIT TYPE ] = ", rband.GetUnitType()
        # Get raster band data values as numpy array.
        rdata = rband.ReadAsArray(0, 0, cols, rows).astype(np.float)

        return(rdata)

    def import_inventory_as_raster(self, values, uncert=None, index=None,
                                   relative=False, valband=1, uncertband=1):
        """ Import raster arrays that represents spatial explicit inventory
            values and uncertainties in form of georeferenced raster band data
            formats.

            Optionally additionally arrays with information of
            inventory indices and uncertainties can be attached,
            Which represents the raster indices for the pixel values and
            the corresponding uncertainties of the inventory raster cells.

            Keyword arguments:
                values  Input raster file representing inventory values,
                        that are stated in defined inventory unit.
                uncert  Input file representing uncertainty of
                        inventory values stated in defined inventory unit
                        (absolute values) as standard deviation (sigma).
                        Relative values are possible -- See <relative> argument
                index  Numpy array containing corresponding indices
                       Per default - increasing numbering is used.
                relative  Boolean to activate import of percentage values of
                          uncertainty. -> Provoke intern calculation of
                          absolute uncertainty values.
                valband  Raster band number containing inventory data.
                         Default is 1.
                uncertband  Raster band number containing uncertainty data.
                            Default is 1.
        """
        self.inv_array = self.import_raster(values)
        self.inv_index = index
        if uncert is not None:
            uncertarray = self.import_raster(uncert)
            if relative:
                uncert_rel = self.inv_array * uncertarray / 100
                self.inv_uncert_array = uncert_rel
            else:
                self.inv_uncert_array = uncertarray

        logger.info('Inventory <%s> with %s shape successful imported as '
                    'raster array' % (self.name, self.inv_array.shape))

        self._Inventory__modmtime()
