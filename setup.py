from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

setup(name='sauventory',
      version='1.0.0',
      description='Spatial Autocorrelated Uncertainty of Inventories',
      long_description=readme(),
      url='https://github.com/m4sth0/sauventory',
      author='Thomas Leppelt',
      author_email='thomas.leppelt@gmail.com',
      license='GPL',
      test_suite="tests",
      packages=['sauventory'],
      install_requires=[
          'gdal',
          'numpy',
          'pysal',
      ],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 2.7',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Topic :: Scientific/Engineering :: GIS',
      ],
      keywords='spatial autocorrelation inventory uncertainty',
      zip_safe=False)
