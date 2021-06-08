#!/usr/bin/env python
from setuptools import setup


setup(
    name='rowing-pkg-matthewghgriffiths',
    version='0.0.3',
    description='Python Distribution Utilities',
    author='Matthew Grifiths',
    author_email='matthewghgriffiths@gmail.com',
    url='https://github.com/matthewghgriffiths/rowing',
    packages=['world_rowing'],
    entry_points={
        'console_scripts': [
            'world_rowing = world_rowing.cli.run'
        ]
    },
    install_requires=[
        'numpy',
        'scipy', 
        'pandas', 
        'matplotlib',
        'requests',
        'cmd2'
    ],
    python_requires=">=3.8",
)