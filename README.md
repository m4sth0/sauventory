# SAUVENTORY Python Package
_Spatial autocorrelated uncertainty for inventories_

This package provide the functionalities for calculating inventories and corresponding uncertainties of independent and spatial related datasets in Python.
**Inventories** represent the accounting of an certain measurable _variable_ for all relevant _source categories_ in a defined _spatial_ and _temporal_ scope. For example [emission inventories](https://en.wikipedia.org/wiki/Emission_inventory) provide information about the amount of pollutans in the atmosphere, species population inventories are used to monitor endangered animals or invasive [alien species](http://www.europe-aliens.org/) and climatological inventories, e. g. for precipitation, are needed to monitor regional natural resources, managing drought risks or modeling climate changes.

Hence the usage of inventories is ubiquitous. The accuracy of an inventory is crucial. Especially if policital or economical decisions has to be made based on theses accountings. The uncertainty calculation of inventories can be performed in different ways. This Python package provide methods to compute inventories uncertainties by the *Gaussian error propagation*. The Tylor series expansion for a sum of source categories can be computed by including covariances or simply utilizing the sum of all variances. Therefore the availability of uncertainties for *all* source categories is mandatory to propagate the inventory uncertainty.

As said before, inventories have a spatial reference. Thus the underlaying source categories can have spatial different sub references, e. g. belonging to different regions with unique uncertainties which are summarized for a national inventory. In this case the *First Law of Geography* according to Waldo Tobler holds:
 
> Everything is related to everything else, but near things are more related than distant things.

and the uncertainty calculation of a spatial related inventory is affected. Which means that you have to take the spatial autocorrelation structure of the inventory into account. This package provide a toolset to do exactely that: Propagate uncertainties for spatial related inventories with consideration of spatial autocorrelations.

---

## Installation:
```pip install sauventory
```
 
---

## Description:
The *sauventory* package primarily consists of two main Python modules. The *inventory.py* and *spatialinventory.py*. In these modules four main classes are defined. The *Inventory* class is the main basis class with methods to import datasets and calculate inventories and uncertainties. The *SpatialInventory* class is derived from the *SpatialInventory* class and features extended methods to handle spatial datasets, test for spatial autocorrelation and calculate semivariograms as well as covariance matrices. The *RasterInventory* and *VectorInventory* classes inherite all attributes and methods from the *SpatialIventory* class and add specific support functionalities for raster and vector datasets, respectively.

						*Inventory*
							 |
							 |
					 *SpatialInventory*
						|		|
						|		|
			*RasterInventory* *VectorInventory*

---

## Examples:
```python
import sauventory as sau
i = sau.inventory.Inventory(name="Test-inventory", desc="For testing purpose", creator="me")
```
---

## Links:

---

## License:

sauventory is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

sauventory is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
