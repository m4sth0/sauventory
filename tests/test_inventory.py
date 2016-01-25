'''
Created on 14.01.2016

@author: Thomas Leppelt
@contact: thomas.leppelt@dwd.de

###############################################################################
                This module is part of the SAUVENTORY package.
           -- Spatial Autocorrelated Uncertainty of Inventories --
###############################################################################

This module perform unittests for the inentory module.

Testing the module with example the agricultural N2O inventory for 2012:

https://www.umweltbundesamt.de/sites/default/files/medien/378/publikationen/
climate-change_24_2014_nationaler_inventarbericht_0.pdf
'''
import unittest

import inventory


class Test(unittest.TestCase):

    def setUp(self):

        # N2O inventory of agricultural source categories for 2012.
        self.n2o_inv = [958.4, 1092.7, 497.8, 42.2, 7.3, 135.9, 1.7, 0.8,
                        51.1, 9539.9, 4693.5, 472.6, 6171.3, 4750.5, 1315,
                        2213.5, 11596.4, 162.8]

        # Corresponding list of source category describtions.
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

        # Convert to absolute values in Gg/a.
        self.n2o_uncert = [a*b/100 for a, b in zip(self.n2o_inv,
                                                   self.n2o_percent)]

    def tearDown(self):
        pass

    def test_inventar(self):
        i = inventory.Inventory("N2O-Agrar-2012", "Gg/a",
                                "German N2O inventory of agricultural source "
                                "categories for 2012",
                                creator="Leppelt")

        i.import_inventory_as_dict(self.n2o_inv, self.n2o_index,
                                   self.n2o_uncert)

        self.assertEqual(round(sum(i.inv_dict.values(), 1)), 43704.0)
        self.assertEqual(round(sum(i.uncert_dict.values()), 1),
                         round(sum(self.n2o_uncert), 1))
        self.assertEqual(len(i.inv_dict), 18)

    def test_inventar_relative(self):
        i = inventory.Inventory("N2O-Agrar-2012", "Gg/a",
                                "German N2O inventory of agricultural source "
                                "categories for 2012",
                                ("2012-01-01 00:00:00", "2013-01-01 00:00:00"),
                                creator="Leppelt")

        i.import_inventory_as_dict(self.n2o_inv, self.n2o_index,
                                   self.n2o_percent, True)

        self.assertEqual(round(sum(i.inv_dict.values(), 1)), 43704.0)
        self.assertEqual(round(sum(i.uncert_dict.values()), 1),
                         round(sum(self.n2o_uncert), 1))
        self.assertEqual(len(i.inv_dict), 18)

    def test_timestamp(self):
        i = inventory.Inventory("N2O-Agrar-2012", "Gg/a",
                                "German N2O inventory of agricultural source "
                                "categories for 2012",
                                ("2012-01-01 00:00:00", None),
                                creator="Leppelt")
        start, end = i.timestamp
        self.assertEqual((start.strftime("%Y-%m-%d %H:%M:%S"), end),
                         ("2012-01-01 00:00:00", None))
        i.timestamp = (None, "2013-01-01 00:00:00")
        start, end = i.timestamp
        self.assertEqual((start, end.strftime("%Y-%m-%d %H:%M:%S")),
                         (None, "2013-01-01 00:00:00"))
        i.timestamp = ("2012-01-01 00:00:00", "2013-01-01 00:00:00")
        start, end = i.timestamp
        self.assertEqual((start.strftime("%Y-%m-%d %H:%M:%S"),
                          end.strftime("%Y-%m-%d %H:%M:%S")),
                         ("2012-01-01 00:00:00", "2013-01-01 00:00:00"))

if __name__ == "__main__":
    unittest.main()
