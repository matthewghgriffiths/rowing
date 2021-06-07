#!/usr/bin/env python
from setuptools import setup


setup(
    name='rowing-pkg-matthewghgriffiths',
    version='0.1.0',
    description='Python Distribution Utilities',
    author='Matthew Grifiths',
    author_email='matthewghgriffiths@gmail.com',
    url='https://github.com/matthewghgriffiths/rowing',
    packages=['world_rowing'],
    install_requires=[
        'numpy',
        'scipy', 
        'pandas', 
        'matplotlib',
        'requests',
        'cmd2', 
    ],
    python_requires=">=3.8",
)