#!/usr/bin/env python
from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='rowing',
    version='0.4.0',
    description='Library for loading and presenting data from the worldrowing.com'
    ' and analysing rowing data',
    author='Matthew Griffiths',
    author_email='matthewghgriffiths@gmail.com',
    url='https://github.com/matthewghgriffiths/rowing',
    packages=['rowing'],
    entry_points={
        'console_scripts': []
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
    python_requires=">=3.10",
    package_data={
        'rowing': [
            'data/iso_country.json',
            'data/flags/*.png',
            'data/*'
        ],
    }
)
