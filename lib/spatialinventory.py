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
import pysal
import sys
import ogr
import gdal

from inventory import Inventory

# Configure logger.
logger = logging.getLogger('spatialinventory')


class SpatialInventory(Inventory):
    """Spatial explicit inventory class"""

    def __init__(self, *args, **kwargs):
        """
        Class instance constructor with additional arguments:
        mi    Morans'I value
        invfile    Name of file containing spatial inventory data.
        sptype    Type of spatial data. Available options are vector and raster
                  only.

        For other parameters see: inventory.Inventory
        """
        self.mi = None
        self.invfile = None
        self.sptype = None
        super(SpatialInventory, self).__init__(*args, **kwargs)

    @property
    def mi(self):
        return self.__mi

    @mi.setter
    def mi(self, mi):
        self.__mi = mi

    @property
    def invfile(self):
        return self.__invfile

    @invfile.setter
    def invfile(self, invfile):
        self.__invfile = invfile

    @property
    def sptype(self):
        return self.__sptype

    @sptype.setter
    def sptype(self, sptype):
        typelist = ['raster', 'vector', None]
        if sptype in typelist:
            self.__sptype = sptype
        else:
            msg = 'Spatial type <%s> is not supported. '
            'Choose one of %s' % (sptype, typelist)
            raise NameError(msg)

    def get_weight_matrix(self, array, rook=False, shpfile=None):
        """Return the spatial weight matrix based on pysal functionalities

        Keyword arguments:
            array    Numpy array with inventory values.
            rook    Boolean to select spatial weights matrix as rook or
                    queen case.
            shpfile    Name of file used to setup weight matrix.
        """
        # Get grid dimension.
        dim = array.shape
        if self.sptype == 'vector':
            try:
                shpfile = self.invfile
                if rook:
                    w = pysal.rook_from_shapefile(shpfile)
                else:
                    w = pysal.queen_from_shapefile(shpfile)
            except:
                msg = "Couldn't build spatial weight matrix for vector "
                "inventory <%s>" % (self.name)
                raise RuntimeError(msg)
        elif self.sptype == 'raster':
            try:
                # Construct weight matrix in input grid size.
                w = pysal.lat2W(*dim, rook=rook)
            except:
                msg = "Couldn't build spatial weight matrix for raster "
                "inventory <%s>" % (self.name)
                raise RuntimeError(msg)

        return(w)

    def check_moran(self, rook=False, shpfile=None):
        """ Get Moran's I statistic for georeferenced inventory

        This method is utilizing pysal package functions for Moran's I
        statistics.
        In case of regular grid input arrays the weight matrix is
        constructed as queen's case by default.
        Each cell (c) as only direct neighbors (n) in each direction per
        default. Alternatively the rook type of neighbors can be chosen.
        Rook:   –––––––––––    Queen:    –––––––––––
                |- - - - -|              |- - - - -|
                |- - n - -|              |- n n n -|
                |- n c n -|              |- n c n -|
                |- - n - -|              |- n n n -|
                |- - - - -|              |- - - - -|
                –––––––––––              –––––––––––
        In case of vectorized input data an shape file has to be passed, which
        will be used as base for rook or queen weight matrix creation.
        Per default the file location is taken from class argument <infile>.

        Keyword arguments:
            rook    Boolean to select spatial weights matrix as rook or
                    queen case.
            shpfile    Name of file used to setup weight matrix.
        """
        # Mask nan values of input array.
        array = self.inv_array

        # Get grid dimension.
        dim = array.shape
        nanids = []
        # Construct weight matrix in input grid size.
        w = self.get_weight_matrix(array, rook=rook)
        try:
            # Reshape input array to N,1 dimension.
            array = array.reshape((w.n, 1))
            # Remove weights and neighbors for nan value ids.
            if np.any(np.isnan(array)):
                idlist = w.id_order
                # Get indices for nan values in array.
                nanids = [i for i in idlist if np.isnan(array[i])]
                # Remove entries from spatial weight keys for nan indices.
                for i in nanids:
                    del w.weights[i]
                    del w.neighbors[i]
                    del w.cardinalities[i]
                # Remove entries from spatial weight values for nan indices.
                for lid in idlist:
                    if lid not in nanids:
                        wlist = w.weights[lid]
                        nlist = w.neighbors[lid]
                        idnonan = [nlist.index(ele) for ele in nlist
                                   if ele not in nanids]
                        wnew = [wlist[i] for i in idnonan]
                        nnew = [nlist[i] for i in idnonan]
                        w.weights[lid] = wnew
                        w.neighbors[lid] = nnew
                # Adjust spatial weight parameters.
                w._id_order = [ele for ele in idlist if ele not in nanids]
                w._n = len(w.weights)
                # Remove nan valeus from array.
                array = np.delete(array, nanids, axis=0)

                logger.info("Found %d NAN values in input array -> "
                            "All will be removed prior to Moran's I statistic"
                            % (len(nanids)))
            # TODO: Use wsp - sparse weight matrix for large grids.
            # Calculate Morans's statistic.
            mi = pysal.Moran(array, w, two_tailed=False)

            logger.info("Moran's I successfully calculated")
            # Print out info box with statistcs.
            info = "\n" + \
                   "---------------------------------------------------\n" +\
                   "| ####     Global Moran's I statistics         ####\n" +\
                   "| Inventory name   : " + self.name + "\n" +\
                   "| -------------------------------------------------\n" +\
                   "| Moran's I              : " + "%.6f" % mi.I + "\n" +\
                   "| Expected value         : " + "%.6f" % mi.EI + "\n" +\
                   "| p-value                : " + "%.6f" % mi.p_norm + "\n" +\
                   "| -------------------------------------------------\n" +\
                   "| Number of non-NA cells : " + str(len(array)) + "\n" +\
                   "| Number of NA cells     : " + str(len(nanids)) + "\n" +\
                   "---------------------------------------------------\n"
            print(info)
            self.mi = mi.I
        except:
            msg = "Couldn't calculate Moran's I for inventory "
            "<%s>" % (self.name)
            raise RuntimeError(msg)

        return(self.mi)

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
        self.sptype = 'raster'

    def _import_raster(self, infile, band=1):
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
            msg = 'Unable to open raster file <%s>' % (infile)
            raise ImportError(msg)
        # Get selected raster band.
        try:
            rband = rfile.GetRasterBand(band)
        except ImportError:
            msg = 'No band %i found' % (band)
            raise ImportError(msg)

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
        self.inv_array = self._import_raster(values)

        # Set import file.
        self.invfile = values

        self.inv_index = index
        if uncert is not None:
            uncertarray = self._import_raster(uncert)
            if relative:
                uncert_rel = self.inv_array * uncertarray / 100
                self.inv_uncert_array = uncert_rel
            else:
                self.inv_uncert_array = uncertarray

        logger.info('Inventory <%s> with %s shape successful imported from '
                    'raster array' % (self.name, self.inv_array.shape))

        self._Inventory__modmtime()


class VectorInventory(SpatialInventory):
    """Spatial vector inventory class"""

    def __init__(self, *args, **kwargs):
        """
        Class instance constructor with additional arguments:

        For other parameters see: inventory.Inventory
        """
        super(VectorInventory, self).__init__(*args, **kwargs)
        self.sptype = 'vector'

    def _import_vector(self, infile, key, valcol, uncol=None, layer=0):
        """ Import routine for vectorized geographical referenced datasets

        The import is handled by ogr package of osgeo.
        Attention: The vector format is limited to ESRI Shapefile support only

        Keyword arguments:
            infile    Input file name and directory as string.
            key    Name of key column in vector file.
            valcol    Name of attribute column selected for inventory values.
            uncol    Name of attribute column selected for inventory
                     uncertainty.
            layer    Number of vector layer. Default is 0

        Returns:
            Arrays with inventory values, indices and uncertainties.
        """

        # Initialize ogr driver.
        driver = ogr.GetDriverByName('ESRI Shapefile')
        # open vector inpuf for read-only.
        vfile = driver.Open(infile, 0)
        if vfile is None:
            msg = 'Unable to open vector file <%s>' % (infile)
            raise ImportError(msg)
        # Get selected vector layer.
        try:
            vlayer = vfile.GetLayer(layer)
        except ImportError, e:
            msg = 'No Layer found'
            raise ImportError(msg)

        # Get layer informations
        n = vlayer.GetFeatureCount()
        extent = vlayer.GetExtent()
        spatialref = vlayer.GetSpatialRef()
        proj = spatialref.GetAttrValue('geogcs')

        # Create attribute table with selected columns.
        featuredict = {}
        indict = {}
        undict = {}
        feature = vlayer.GetNextFeature()
        while feature:
            value = feature.GetField(valcol)
            keyname = feature.GetField(key)

            if uncol is None:
                uncert = np.nan
            else:
                uncert = feature.GetField(uncol)
            featuredict[feature.GetField(key)] = [value, keyname, uncert]
            feature = vlayer.GetNextFeature()
        # Remove None key values from dictionary.
        try:
            del featuredict[None]
        except:
            pass
        # Get inventory data as numpy array.
        valarray = np.array([i[0] for i in featuredict.values()], dtype=float)
        inarray = np.array([i[1] for i in featuredict.values()],
                           dtype='string')
        unarray = np.array([i[2] for i in featuredict.values()], dtype=float)

        # Print imported vector summary.
        print "[ LAYER ] = ", layer
        print "[ EXTENT ] = ", extent
        print "[ PROJECTION ] = ", proj
        print "[ FEATURE COUNT ] = ", n
        print "[ KEY ATTRIBUTE COUNT ] = ", len(featuredict)
        print "[ MIN VALUE] = ", np.nanmin(valarray)
        print "[ MAX VALUE] = ", np.nanmax(valarray)
        print "[ MIN UNCERTAINTY] = ", np.nanmin(unarray)
        print "[ MAX UNCERTAINTY] = ", np.nanmax(unarray)

        return(valarray, inarray, unarray)

    def import_inventory_as_vector(self, infile, values, uncert=None,
                                   index='cat', relative=False, layer=0):
        """ Import vector map layer that stores spatial explicit inventory
            values and uncertainties in form of columns for georeferenced
            vector feature attributes.

            Optionally additionally attribute columns with information of
            inventory indices and uncertainties can be specified,
            which represents the vector indices for the inventory components
            and the corresponding uncertainties.

            Keyword arguments:
                infile  Input vector file name and directory as string.
                values  Name of attribute column containing inventory data.
                uncert  Name of attribute column containing uncertainty of
                        inventory values stated in defined inventory unit
                        (absolute values) as standard deviation (sigma).
                        Relative values are possible -- See <relative> argument
                index  Name of key column containing corresponding index values
                       for the inventory. Per default - cat column is used.
                relative  Boolean to activate import of percentage values of
                          uncertainty. -> Provoke intern calculation of
                          absolute uncertainty values.
                layer  Number of layer containing inventory data.
                       Default is 0.
        """
        valarray, inarray, unarray = self._import_vector(infile, key=index,
                                                         valcol=values,
                                                         uncol=uncert,
                                                         layer=layer)
        self.inv_array = valarray
        self.inv_index = inarray
        if relative:
            uncert_rel = self.inv_array * unarray / 100
            self.inv_uncert_array = uncert_rel
        else:
            self.inv_uncert_array = unarray

        # Set import file.
        self.invfile = infile

        logger.info('Inventory <%s> with %d categories successful imported '
                    'from vector feature layer' % (self.name,
                                                   len(self.inv_array)))

        self._Inventory__modmtime()
