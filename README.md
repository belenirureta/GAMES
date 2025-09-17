This script generates synthetic asteroid streaks for astronomical image analysis, specifically for simulating the appearance of asteroids in ZTF-like survey images. The user can specify a coarse magnitude-length distribution to generate the streaks. For a given magnitude, it calculates the appropriate brightness per pixel and injects synthetic streaks into the images, making them realistic by convolving them with the PSF of each individual image. The orientation of the streaks is random. The resultant images can be used for testing detection algorithms or training machine-learning models.

**What it does:**

* Generates synthetic asteroid streaks with specified parameters
* Simulates realistic noise and background
* Keeps track of the metadata for a later analysis
* Useful for monitoring detection efficiency

**What it doesn't do:**

* Simulate flux variations (induced by rotation, phase angle, etc.)
* Use real population models to generate the streak properties

Created at IPAC (Infrared Processing and Analysis Center), Caltech.
Author: Belén Yu Irureta-Goyena
Date: September 2023
