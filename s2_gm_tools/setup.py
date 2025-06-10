#!/usr/bin/env python3

import io
import os
from setuptools import find_packages, setup

# Where are we?
IS_DEA_SANDBOX = ('sandbox' in os.getenv('JUPYTER_IMAGE', default=''))

# What packages are required for this module to be executed?
REQUIRED = [
    "xarray",
    "numba",
    "numpy",
    "datacube",
    "pyproj",
    "fsspec",
    "odc-stats",
    "odc-stac",
    "odc-io",
    "odc-algo",
    "odc-dscache",
    "pandas",
    "typing",
    "geopandas",
    "gdal",
    "scipy",
    "dask",
    "dask-ml",
    "geopy",
    "dea_tools",
]

# Package meta-data.
NAME = "s2-gm-tools"
DESCRIPTION = "Tools for running DEA Sentinel-2 geomedians"
URL = "https://github.com/GeoscienceAustralia/dea-notebooks"
EMAIL = "chad.burton@ga.gov.au"
AUTHOR = "Digital Earth Australia"
REQUIRES_PYTHON = ">=3.9.0"

# Import the README and use it as the long-description.
here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup_kwargs = {
    "name": NAME,
    "version": "1.0.0",
    "description": DESCRIPTION,
    "long_description": long_description,
    "author": AUTHOR,
    "author_email": EMAIL,
    "python_requires": REQUIRES_PYTHON,
    "url": URL,
    "install_requires": REQUIRED if not IS_DEA_SANDBOX else [],
    "packages": find_packages(),
    "include_package_data": True,
    "license": "Apache License 2.0",
    "entry_points": {
        "console_scripts": [
            "cm-task = cm_tools.geojson_defined_tasks:main",
        ]
    },
}

setup(**setup_kwargs)