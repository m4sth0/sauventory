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
This module defines an inventory class that provide functionalities to compute
inventories and corresponding uncertainties from given data.
"""


from datetime import datetime
import logging

import ranktransform
import numpy as np

# Configure logger.
logger = logging.getLogger('inventory')


class Inventory(object):

    # Define class constants.
    NODATAVAL = -9999

    def __init__(self, name, unit, desc=None, timestamp=(None, None),
                 creator=None):
        """ Class instance with following attributes:
        Public:
            name      String represent the name of the inventory.
            desc      String with inventory description.
            creator   String containing name of the creator.
        Privat:
            unit       String describing the SI base unit of the inventory.
            timestamp  Tuple with start and end time in %Y-%m-%d %H:%M:%S
                       format. Time instance if end time is empty or None
            ctime      Time stamp of inventory opject creation.
            mtime      Last modification time stamp.
            inv_array  Numpy array represent inventory values
            inv_index  Numpy array represent inventory indices
            inv_covmat    Covariance matrix if inventory as ndarray.
        """
        self.name = name
        self.desc = desc
        self.unit = unit
        self.creator = creator
        self.timestamp = timestamp
        self.__ctime = datetime.now()
        self.__mtime = datetime.now()
        self.inv_dict = {}
        self.uncert_dict = {}
        self.inv_uncert = None
        self.__inv_sum = None
        self.__inv_uncert = None
        self.inv_covmat = None
        logger.info('Inventory object: initialized')

    @property
    def unit(self):
        return self.__unit

    @unit.setter
    def unit(self, unit):
        unitlist = ["g/m2", "kg/ha", "Gg"]
        if unit in unitlist:
            self.__unit = unit
        else:
            raise NameError('%s Unit is not supported. Choose one of %s'
                            % (unit, unitlist))

    @property
    def timestamp(self):
        return(map(str, self.__timestamp))

    @timestamp.setter
    def timestamp(self, timestamp):
        if timestamp is not None:
            start, end = timestamp
            if start is not None:
                starttime = datetime.strptime(start,
                                              "%Y-%m-%d %H:%M:%S")
            else:
                starttime = None
            if end is not None:
                endtime = datetime.strptime(end,
                                            "%Y-%m-%d %H:%M:%S")
            else:
                endtime = None
            self.__timestamp = (starttime, endtime)
        logger.info('Set timestamp to start: %s -- end: %s' %
                    (timestamp[0], timestamp[1]))

    @property
    def inv_dict(self):
        return self.__inv_dict

    @inv_dict.setter
    def inv_dict(self, inv_dict):
        self.__inv_dict = inv_dict

    @property
    def uncert_dict(self):
        return self.__uncert_dict

    @uncert_dict.setter
    def uncert_dict(self, uncert_dict):
        self.__uncert_dict = uncert_dict

    @property
    def inv_array(self):
        return self.__inv_array

    @inv_array.setter
    def inv_array(self, inv_array):
        self.__inv_array = inv_array

    @property
    def inv_index(self):
        return self.__inv_index

    @inv_index.setter
    def inv_index(self, inv_index):
        self.__inv_index = inv_index

    @property
    def inv_uncert_array(self):
        return self.__inv_uncert_array

    @inv_uncert_array.setter
    def inv_uncert_array(self, inv_uncert_array):
        self.__inv_uncert_array = inv_uncert_array

    @property
    def inv_sum(self):
        return self.__inv_sum

    @inv_sum.setter
    def inv_sum(self, inv_sum):
        self.__inv_sum = inv_sum

    @property
    def inv_uncert(self):
        return self.__inv_uncert

    @inv_uncert.setter
    def inv_uncert(self, inv_uncert):
        self.__inv_uncert = inv_uncert

    @property
    def inv_covmat(self):
        return self.__inv_covmat

    @inv_covmat.setter
    def inv_covmat(self, inv_covmat):
        self.__inv_covmat = inv_covmat

    def __getctime(self):
        return self.__ctime.strftime("%Y-%m-%d %H:%M:%S")

    def __getmtime(self):
        return self.__mtime.strftime("%Y-%m-%d %H:%M:%S")

    def __setmtime(self, mtime):
        self.__mtime = mtime

    def __modmtime(self):
        """Update modification time to present time"""
        self.mtime = datetime.now()

    # Define property methods.
    ctime = property(__getctime)
    mtime = property(__getmtime, __setmtime)

    def printsum(self):
        """Print inventory summary information"""
        start, end = self.timestamp
        if self.inv_sum:
            acc = round(self.inv_sum, 2)
        else:
            acc = self.inv_sum
        if self.inv_uncert:
            uncert = round(self.inv_uncert, 2)
        else:
            uncert = self.inv_uncert

        info = '\n--------------------------------------------------------' + \
               '\nInventory overview:\n' + 'Name: ' + self.name + \
               '\nDescription: ' + str(self.desc) + \
               '\nCreator: ' + str(self.creator) + \
               '\nUnit: ' + self.unit + \
               '\n--------------------------------------------------------' + \
               '\nCreation time: ' + self.ctime + \
               '\nStart: ' + start + \
               ' - End: ' + end + \
               '\nLast modification: ' + self.mtime + \
               '\n--------------------------------------------------------' + \
               '\nInventory accumulation: ' + str(acc) + \
               ' ' + self.unit + \
               '\nInventory uncertainty:  ' + str(uncert) + \
               ' ' + self.unit + \
               '\n--------------------------------------------------------'
        logger.info(info)
        print(info)

    def import_inventory(self, values, uncert=None, index=None,
                         relative=False):
        """ Import list or array that represents inventory values of
            different source categories and create input type dependent
            dictionary or array with corresponding values.

            Optionally additionally lists or arrays with information of
            inventory indices and uncertainties can be attached,
            Which represents the names/indices and the corresponding
            uncertainties of the inventory values.

            Keyword arguments:
                values  list or numpy array representing inventory values,
                        that are stated in defined inventory unit..
                       Per default - increasing numbering is used.
                uncert  list or numpy array representing uncertainty of
                        inventory values stated in defined inventory unit
                        (absolute values) as standard deviation (sigma).
                        Relative values are possible -- See <relative> argument
                index  list or numpy array containing corresponding indices
                       in string format
                relative  Boolean to activate import of percentage values of
                          uncertainty. -> Provoke intern calculation of
                          absolute uncertainty values.
        """
        inlist = [values, index, uncert]

        # Check argument types.
        dictbool = all([isinstance(arg, list) for arg in inlist
                        if arg is not None])
        arraybool = all([isinstance(arg, np.ndarray) for arg in inlist
                         if arg is not None])
        if dictbool:
            self.inv_dict = dict(zip(index, values))
            if uncert is not None:
                if relative:
                    uncert_rel = [a*b/100 for a, b in zip(values, uncert)]
                    self.uncert_dict = dict(zip(index, uncert_rel))
                else:
                    self.uncert_dict = dict(zip(index, uncert))

            logger.info('Inventory <%s> with %d elements successful imported '
                        'as dictionary' % (self.name, len(self.inv_dict)))

        elif arraybool:
            self.inv_array = values
            self.inv_index = index
            if uncert is not None:
                if relative:
                    uncert_rel = values * uncert / 100
                    self.inv_uncert_array = uncert_rel
                else:
                    self.inv_uncert_array = uncert

            logger.info('Inventory <%s> with %s shape successful imported as '
                        'array' % (self.name, self.inv_array.shape))
        else:
            raise TypeError('Input types are not matching. Need uniform list'
                            'or array inputs')

        self.__modmtime()

    def ranktransform(self, mat, cormat):
        """ Function to perform rank transformation of matrices by given
            Spearman rank correlation matrix using ranktransform module.
        """
        mat_trans = ranktransform.transform_by_corrmat(mat, cormat)

        self.__modmtime()

        return(mat_trans)

    def accumulate(self):
        """ Calculate inventory from intern source category dictionary or
            array values.

            NaN values are ignored and reatead as zero.
        """
        try:
            if not self.inv_dict:
                result = np.nansum(self.inv_array)
            else:
                result = np.nansum(self.inv_dict.values())

            self.inv_sum = result

            logger.info('Inventory <%s>: %d %s successfully '
                        'computed' % (self.name, self.inv_sum, self.unit))

        except:
            logger.info('Inventory <%s> not computed'
                        % (self.name))

        self.__modmtime()

        return(self.inv_sum)

    def propagate(self, cv=False):
        """ Calculate the overall uncertainty for saved inventory values by
            Gaussian error propagation.

        NaN values are ignored and treatead as zero.

        Keyword arguments:
            cv    Boolean if covariances should be used to propagate
                  uncertainty.
        """
        # Primarily choose existing uncertainty dictionary.
        # Select inventory uncertainty source.
        if not self.uncert_dict:
            uncertobj = self.inv_uncert_array
        else:
            uncertobj = self.uncert_dict.values()

        # Optional propagation via covariances.
        if cv and self.inv_covmat is None:
            msg = ("Couldn't find covariance matrix for inventory <%s>" %
                   (self.name))
            raise ValueError(msg)
        elif cv:
            result = np.sqrt(self.inv_covmat.sum())
        else:
            result = np.sqrt(np.nansum(map(np.square, uncertobj)))
        self.inv_uncert = result
        try:
            float(self.inv_uncert)
            logger.info('Inventory <%s> uncertainty: <%d> <%s> successfully '
                        'computed' % (self.name, self.inv_uncert, self.unit))
        except:
            logger.info('Inventory <%s> uncertainty not computed'
                        % (self.name))

        self.__modmtime()

        return(self.inv_uncert)
