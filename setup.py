#!/usr/bin/env python
from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='rowing',
    version='0.2.2',
    description='Library for loading and presenting data from the worldrowing.com',
    author='Matthew Griffiths',
    author_email='matthewghgriffiths@gmail.com',
    url='https://github.com/matthewghgriffiths/rowing',
    packages=['rowing.world_rowing', 'rowing.analysis'],
    entry_points={
        'console_scripts': [
            'world_rowing = rowing.world_rowing.cli:run [CLI]', 
            'garmin = rowing.analysis.garmin:main [GARMIN]',
            'gpx = rowing.analysis.files:main [GPX]',
        ]
    },
    license='MIT', 
    long_description=long_description, 
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.8.0',
        'pandas>=1.4.0',
        'matplotlib>=3.0.0',
        'tqdm>=4.0.0',
    ],
    extras_require={
        'CLI': ['cmd2>=2.0.0'],
        'REQ': ['requests>=2.0.0'], # Requests is not required if using pyodide
        'GARMIN': ['garminconnect>=0.1.45', 'fitparse>=1.2.0'],
        "GPX": ['gpxpy>=1.5.0', 'fitparse>=1.2.0'],
    },
    python_requires=">=3.8",
    package_data={
        'rowing': [
            'world_rowing/data/*.csv.gz', 
            'world_rowing/data/iso_country.json', 
            'world_rowing/data/flags/*.png', 
            'analysis/data/*.tsv'
        ],
    }
)
