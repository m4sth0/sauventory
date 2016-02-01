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
This module perform unittests for supplementary functions within the
sauventory package.
"""

import numpy as np
import unittest

import stepfunction as sf


class Test(unittest.TestCase):

    def setUp(self):
        self.x = np.arange(20)
        self.y = np.arange(20)

        self.a = np.random.normal(1, 0.3, 10000)

    def test_stepfunction(self):
        f = sf.StepFunction(self.x, self.y)
        f2 = sf.StepFunction(self.x, self.y, side='right')

        self.assertEqual(f(5), 4)
        self.assertEqual(f2(5), 5)

    def test_ecdf(self):
        ecdf = sf.ECDF(self.a)
        self.assertAlmostEqual(ecdf(1), 0.5, 1)


if __name__ == "__main__":
    unittest.main()
