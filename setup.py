#!/usr/bin/env python
from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='rowing',
    version='0.3.1',
    description='Library for loading and presenting data from the worldrowing.com'
    ' and analysing rowing data',
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
        'numpy==1.25.0',
        'scipy==1.11.4',
        'pandas==2.0.2',
        'matplotlib==3.7.1',
        'tqdm==4.65.0',
    ],
    extras_require={
        'WORLDROWING': ['streamlit==1.29.0', 'plotly==5.14.1'],
        # Requests is not required if using pyodide
        'REQ': ['requests==2.31.0'],
        'GARMIN': [
            'garminconnect==0.1.45',
            'fitparse==1.2.0',
            "openpyxl==3.1.0",
            "pyarrow==12.0.0"
        ],
        "GPX": ['gpxpy==1.5.0', 'fitparse==1.2.0'],
    },
    python_requires=">=3.9",
    package_data={
        'rowing': [
            'data/*.csv.gz',
            'data/iso_country.json',
            'data/flags/*.png',
            'data/*.tsv'
            'data/*'
        ],
    }
)
