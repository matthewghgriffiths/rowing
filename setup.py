#!/usr/bin/env python
from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='rowing',
    version='0.3.1',
    description='Library for loading and presenting data from the worldrowing.com',
    author='Matthew Griffiths',
    author_email='matthewghgriffiths@gmail.com',
    url='https://github.com/matthewghgriffiths/rowing',
    packages=['rowing'],
    entry_points={
        'console_scripts': [
            'worldrowing = rowing.worldrowing.app:main [WORLDROWING]',
            'garmin = rowing.analysis.garmin:main [GARMIN]',
            'gpx = rowing.analysis.files:main [GPX]',
        ]
    },
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'pandas>=1.3.0',
        'matplotlib>=3.5.1',
        'tqdm>=4.0.0',
    ],
    extras_require={
        'WORLDROWING': ['streamlit>=1.22.0', 'plotly>=5.14.1'],
        # Requests is not required if using pyodide
        'REQ': ['requests>=2.27.1'],
        'GARMIN': [
            'garminconnect>=0.1.45',
            'fitparse>=1.2.0',
            "openpyxl>=3.0.0",
            "pyarrow>=7.0.0"
        ],
        "GPX": ['gpxpy>=1.5.0', 'fitparse>=1.2.0'],
    },
    python_requires=">=3.8",
    package_data={
        'rowing': [
            'data/*.csv.gz',
            'data/iso_country.json',
            'data/flags/*.png',
            'data/*.tsv'
        ],
    }
)
