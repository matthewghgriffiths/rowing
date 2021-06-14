#!/usr/bin/env python
from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='world_rowing',
    version='0.1.6',
    description='Library for loading and presenting data from the worldrowing.com',
    author='Matthew Grifiths',
    author_email='matthewghgriffiths@gmail.com',
    url='https://github.com/matthewghgriffiths/rowing',
    packages=['world_rowing'],
    entry_points={
        'console_scripts': [
            'world_rowing = world_rowing.cli:run'
        ]
    },
    license='MIT', 
    long_description=long_description, 
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'requests',
        'cmd2>=2.0.0'
    ],
    python_requires=">=3.8",
    package_data={
        'world_rowing': ['data/*.csv.gz', 'data/flags/*.png'],
    }
)
