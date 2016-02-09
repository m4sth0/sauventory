from setuptools import setup

setup(name='sauventory',
      version='0.1',
      description='Spatial Autocorrelated Uncertainty of Inventories',
      url='https://github.com/m4sth0/sauventory',
      author='Thomas Leppelt',
      author_email='thomas.leppelt@gmail.com',
      license='GPL',
      packages=['sauventory'],
      install_requires=[
          'gdal',
          'numpy',
          'ogr',
          'pysal',
      ],
      zip_safe=False)
