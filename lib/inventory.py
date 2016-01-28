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

    def __init__(self, name, unit, desc=None, timestamp=(None, None),
                 creator=None):
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
            inv_array  Numpy array represent inventory values
            inv_index  Numpy array represent inventory indices
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
        self.inv_uncert = None
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

    def import_inventory(self, values, index=None, uncert=None,
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
                        that are stated in defined inventory unit.
                index  list or numpy array containg corresponding indices
                       in string format.
                       Per default - increasing numbering is used.
                uncert  list or numpy array representing uncertainty of
                        inventory values stated in defined inventory unit
                        (absolute values).
                        Relative values are possible -- See <relative> argument
                relative  Boolean to activate import of percentage values of
                          uncertainty. -> Provoke ntern calculation of absolut
                          uncertainty values.
            Returns:
                dictionarys or numpy arrays containing inventory values or
                uncertainties with corresponding indices as keys.
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

    def ranktransform(self, mat, cormat):
        """ Function to perform rank transformation of matrices by given
            Spearman rank correlation matrix using ranktransform module.
        """
        mat_trans = ranktransform.transform_by_corrmat(mat, cormat)

        return(mat_trans)

    def get_inventory(self):
        """ Calculate inventory from intern source category dictionary or
            array.
        """
        if not self.inv_dict:
            result = np.sum(self.inv_array)
        else:
            result = np.sum(self.inv_dict.values())

        self.inv_sum = result

        logger.info('Inventory <%s>: %d %s successfully '
                    'computed' % (self.name, self.inv_sum, self.unit))

        return(result)

    def propagate(self):
        """ Calculate the overall uncertainty for saved inventory values by
            Gaussian error propagation.
        """
        # Primarily choose existing uncertainty dictionary.
        if not self.uncert_dict:
            uncertobj = self.inv_uncert_array
        else:
            uncertobj = self.uncert_dict.values()

        result = np.sqrt(np.sum(map(np.square, uncertobj)))
        self.inv_uncert = result

        logger.info('Inventory <%s> uncertainty: %d %s successfully '
                    'computed' % (self.name, self.inv_uncert, self.unit))

if __name__ == '__main__':
    pass
