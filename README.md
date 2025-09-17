Generator for Asteroid detection in Machine-learning Enhanced Surveys

This script generates synthetic asteroid streaks for astronomical image analysis, specifically for simulating the appearance of asteroids in ZTF-like survey images. The user can specify a coarse magnitude-length distribution to generate the streaks. For a given magnitude, it calculates the appropriate brightness per pixel and injects synthetic streaks into the images, making them realistic by convolving them with the PSF of each individual image. The orientation of the streaks is random. The resultant images can be used for testing detection algorithms or training machine-learning models.

**What it does:**

* Generates synthetic asteroid streaks with specified parameters
* Simulates realistic noise and background
* Keeps track of the metadata for a later analysis
* Useful for monitoring detection efficiency

**What it doesn't do:**

* Simulate flux variations (induced by rotation, phase angle, etc.)
* Use real population models to generate the streak properties

**How to use:**

* asteroidGenerator.py injects the asteroids into the FITS images. 
* FITS2PNG.py converts the FITS to PNG images. This is useful for many machine-learning applications. 

Created at IPAC (Infrared Processing and Analysis Center), Caltech.
Author: Belén Yu Irureta-Goyena
Date: September 2023

This script was originally developed for ZTF images, under the purview of the NEOZTF project, and VST images. For more details, see https://iopscience.iop.org/article/10.1088/1538-3873/add379 and https://www.aanda.org/articles/aa/abs/2025/02/aa52756-24/aa52756-24.html. If used, it should be cited accordingly. 

This project has received funding from the European Union’s Horizon 2020 and Horizon Europe research and innovation programmes under the Marie Skłodowska-Curie grant agreement Nos. 945363 and 101105725, and funding from the Swiss National Science Foundation and the Swiss Innovation Agency (Innosuisse) via the BRIDGE Discovery grant 40B2-0 194729.
