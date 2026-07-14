#!/usr/bin/env python

from setuptools import find_packages, setup

print(f"Installing {find_packages()}")
setup(
    name="egomax2d",
    version="0.1.0",
    description="EgoMax2D: 2D keypoint estimation for head-mounted stereo (ViT heatmap)",
    packages=find_packages(),
)
