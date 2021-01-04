#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="porto_taxis",
    version="0.0.1",
    install_requires=[
        "gym >= 0.2.3",
        "numpy",
        "scipy",
        "numba",
        "pandas >= 1.0.1",
        "matplotlib",
        "seaborn",
        "tqdm",
        "sacred",
        "pymongo",
        "mdp_extras @ git+https://github.com/aaronsnoswell/mdp-extras.git",
        "unimodal_irl @ git+https://github.com/aaronsnoswell/unimodal-irl.git",
        "multimodal_irl @ git+https://github.com/aaronsnoswell/multimodal-irl.git",
    ],
    packages=find_packages(),
)
