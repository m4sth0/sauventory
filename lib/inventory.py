'''
Created on 07.01.2016

@author: Thomas Leppelt
@contact: thomas.leppelt@dwd.de

###############################################################################
                This module is part of the SAUVENTORY package.
           -- Spatial Autocorrelated Uncertainty of Inventories --
###############################################################################

This module defines an inventory class that provide functionalities to compute
inventories and corresponding uncertainties from given data.
'''

from datetime import datetime
import logging

import ranktransform
import numpy as np

# Configure logger.
logger = logging.getLogger('inventory')


class Inventory(object):

    # Define class constants.
    NODATAVAL = np.NaN

    def __init__(self, name, unit, desc=None, timestamp=(None, None), creator=None):
        """ Class instance with following attributes:
        Public:
            name
            desc
            creator
        Privat:
            unit
            timestamp  Tuple with start and end time in %Y-%m-%d %H:%M:%S
                       format. Time instance if end time is empty or None
            ctime
            mtime
            inv_array
            inv_index
        """
        self.name = name
        self.desc = desc
        self.unit = unit
        self.creator = creator
        self.timestamp = timestamp
        self.__ctime = datetime.now()
        self.__mtime = self.__ctime
        self.inv_dict = {}
        self.uncert_dict = {}
        logger.info('Inventory object: initialized')

    @property
    def unit(self):
        return self.__unit

    @unit.setter
    def unit(self, unit):
        unitlist = ["g/m2", "kg/ha", "Gg/a"]
        if unit in unitlist:
            self.__unit = unit
        else:
            raise NameError('%s Unit is not supported. Choose one of %s'
                            % (unit, unitlist))

    @property
    def timestamp(self):
        return(self.__timestamp)

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

    def __getctime(self):
        return self.__ctime.strftime("%Y-%m-%d %H:%M:%S")

    def __getmtime(self):
        return self.__mtime.strftime("%Y-%m-%d %H:%M:%S")

    # Define property methods.
    ctime = property(__getctime)
    mtime = property(__getmtime)

    def printsum(self):
        info = 'Inventory overview:\n' + 'Name: ' + self.name + \
               '\nDescription: ' + self.desc + \
               '\nCreator: ' + self.creator + \
                '\nCreation time: ' + self.ctime + \
                '\nUnit: ' + self.unit + \
                '\nTimestamp: ' + self.timestamp + \
                '\nLast modification: ' + self.mtime
        logger.info(info)
        print(info)

    def import_inventory_as_dict(self, values, index=None, uncert=None,
                                 relative=False):
        """ Import an list that represents inventory values of different
            source categories and create an dictionary.
            Optionally a second list of index values can be attached,
            representing the names or indices of the inventory values.
            Args:
                values  list representing inventory values, that are
                       stated in defined inventory unit.
                index  list containg corresponding indices in string
                       format. Per default - increasing numbering is used.
                uncert  list representing uncertainty of inventory values
                        stated in defined inventory unit (absolute values)
                        Relative values are possible -- See <relative> argument
                relative  Boolean to activate improt of percentage values of
                          uncertainty.
            Returns:
                dictionary  containing inventory values with
                            corresponding indices as keys.
        """
        self.inv_dict = dict(zip(index, values))
        if uncert is not None:
            if relative:
                uncert_rel = [a*b/100 for a, b in zip(values, uncert)]
                self.uncert_dict = dict(zip(index, uncert_rel))
            else:
                self.uncert_dict = dict(zip(index, uncert))

        logger.info('Inventory %s with %d elements successful imported as '
                    'dictionary' % (self.name, len(self.inv_dict)))

    def import_inventory_as_array(self, array, index=None):
        """ Import an array that represents inventory values of different
            source categories.
            Optionally a second array of index values can be attached,
            representing the names or indices of the inventory values.
            Args:
                array  numpy array representing inventory values, that are
                       stated in defined inventory unit.
                index  numpy array containg corresponding indices in string
                       format. Per default - increasing numbering is used.
        """
        self.inv_array = array
        self.inv_index = index

    def ranktransform(self, mat, cormat):
        """ Function to perform rank transformation of matrices by given
            Spearman rank correlation matrix using ranktransform module.
        """
        mat_trans = ranktransform.transform_by_corrmat(mat, cormat)

        return(mat_trans)

if __name__ == '__main__':
    pass
