import test_functions
import test_inventory
import test_ranktransform
import test_variogram
import unittest
import doctest


def suite():
    suite = unittest.TestSuite()
    # suite.addTests(doctest.DocTestSuite(inventory))
    suite.addTests(test_inventory.suite())
    suite.addTests(test_ranktransform.suite())
    suite.addTests(test_functions.suite())
    suite.addTests(test_variogram.suite())
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
