# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# SNRasteroidGenerator.py
#
# This script generates synthetic asteroid streaks for astronomical image analysis,
# specifically for simulating the appearance of asteroids in VST-like survey images.
# The user can specify an SNR-length distribution to generate the streaks.
# For a given SNR, it calculates the appropriate brightness per pixel and injects 
# synthetic streaks into the images, making them realistic by convolving them with 
# the PSF of each individual image. The orientation of the streaks is random.
# The resultant images can be used for testing detection algorithms or training 
# machine-learning models.
#
# What it does:
#   - Generates synthetic asteroid streaks with specified parameters
#   - Simulates realistic noise and background
#   - Keeps track of the metadata for a later analysis
#   - Useful for monitoring detection efficiency
#
# What it doesn't do:
#   - Simulate flux variations (induced by rotation, phase angle, etc.)
#   - Use real population models to generate the streak properties
#
# Created at École Polytechnique Fédérale de Lausanne (EPFL).
# Author: Belén Yu Irureta-Goyena
# Date: September 2024
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import astropy 
from astropy.io import fits
import random
import os
import csv
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
import lacosmic
import astropy.io.fits as fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
import os.path

def clip(data, nsigma):
    """
    estimates the background level and its standard deviation

    Parameters
    ----------
    data : numpy array
        image.
    nsigma : int
        nsigma from the background to be included.

    Returns
    -------
    tuple
        background, noise.

    """
    lennewdata = 0
    lenolddata = data.size
    while lenolddata>lennewdata:
        #print('len', lenolddata)
        lenolddata = data.size
        data       = data[np.where((data<np.nanmedian(data)+nsigma*np.nanstd(data)) & \
                                   (data>np.nanmedian(data)-nsigma*np.nanstd(data)))]
        lennewdata = data.size
    return np.nanstd(data) # np.nanmedian(data) 

def asteroidGenerator(image, SNR, length, width):
    """
    Paint a synthetic asteroid streak onto an astronomical image.

    This function randomly selects a starting position and orientation for the streak,
    computes the corresponding end position based on the desired length and ensures the streak
    remains within image boundaries. The brightness of the streak is set according to the input
    SNR and local noise, and the trail is injected into the image as a straight line.

    Args:
        image (2D np.ndarray): The astronomical image to modify (will be updated in-place).
        SNR (float): Desired signal-to-noise ratio of the asteroid streak.
        length (float): Desired length of the streak in pixels.
        width (float): Width of the streak in pixels (used for flux calculation).

    Returns:
        tuple or None: (traildata, start_x, start_y, end_x, end_y, slope, total_flux, brightness_per_pixel, length, noise_per_pixel)
        if the streak is successfully painted, else None.
    """
    
    image_shape = image.shape

    traildata = np.zeros(image_shape)

    # randomly generate a slope and starting position
    slope = np.random.uniform(-1, 1)
    start_y = np.random.randint(0, image_shape[0])
    start_x = np.random.randint(0, image_shape[1])
    
    print('Trail start position: (' + str(start_x) + ', ' + str(start_y) + ')')
    print('Trail slope: ' + str(slope))
    
    # calculate the ending position based on the desired length and slope
    end_x = int(start_x + length / np.sqrt(1 + slope**2))
    end_y = int(start_y + slope * (end_x - start_x))
    
    # make sure the endpoints are within the image boundaries
   # end_x = np.clip(end_x, 0, image_shape[1]-1)
   # end_y = np.clip(end_y, 0, image_shape[0]-1)

    if end_x < 0:
        start_x = start_x - end_x
        end_x = 0

    elif end_x >= image_shape[1]:
        start_x = start_x - (end_x - image_shape[1] + 1)
        end_x = image_shape[1] - 1

    if end_y < 0:
        start_y = start_y - end_y
        end_y = 0

    elif end_y >= image_shape[0]:
        start_y = start_y - (end_y - image_shape[0] + 1)
        end_y = image_shape[0] - 1

    print('Trail end position: (' + str(end_x) + ', ' + str(end_y) + ')')

    # Calculate the size of the square for clipping (e.g., 200x200)
    square_size = 200

    # Determine the bounds for the square clipping region
    square_left = start_x - square_size // 2
    square_right = start_x + square_size // 2
    square_top = start_y - square_size // 2
    square_bottom = start_y + square_size // 2

    # Ensure the square stays within the image boundaries
    if square_left < 0:
        square_left = 0
        square_right = square_size
    elif square_right >= image.shape[1]:
        square_right = image.shape[1] - 1
        square_left = image.shape[1] - square_size - 1

    if square_top < 0:
        square_top = 0
        square_bottom = square_size
    elif square_bottom >= image.shape[0]:
        square_bottom = image.shape[0] - 1
        square_top = image.shape[0] - square_size - 1

    # Extract the square region from the image for clipping
    square = image[square_top:square_bottom, square_left:square_right]

    # Calculate noise_per_pixel using the clip function on the square region
    noise_per_pixel = clip(square, 3)  
    total_flux = SNR**2 * noise_per_pixel * length**0.5 * width**0.5
    brightness_per_pixel = total_flux / length

    xs, ys = [], []
    for x in range(start_x,end_x):
        yaffected = start_y + slope * (x - start_x)
        xs.append(x)
        ys.append(int(yaffected))

    # Randomly flip x/y axes 50 % of the time; otherwise you miss half of the orientations
    if np.random.randint(0, 2) > 0:
        xs, ys = ys, xs
        start_x, start_y = start_y, start_x
        end_x, end_y = end_y, end_x
        print('Streak was flipped')
    else:
        print('Streak was not flipped')

    # Paint the trail, skip if out of bounds
    unsuccessful_painting = 0
    for x, y in zip(xs, ys):
        try:
            traildata[y, x] = brightness_per_pixel
        except IndexError:
            unsuccessful_painting += 1
    if unsuccessful_painting == 0:
        return traildata, start_x, start_y, end_x, end_y, slope, total_flux, brightness_per_pixel, length, noise_per_pixel
    else:
        print('Painting unsuccessful, retrying...')
        return None

def cameronSeeing(data, header):
    """
    Estimate the seeing (FWHM of stars) in an astronomical image using SExtractor.
    The seeing is needed to set up the PSF kernel for streak convolution.

    This function performs the following steps:
      1. Subtracts the background from the image using sigma clipping.
      2. Removes cosmic rays using lacosmic.
      3. Saves the cleaned image to disk.
      4. Runs SExtractor to detect sources and measure their FWHM.
      5. Selects star-like objects and computes the median FWHM (seeing) in arcseconds.

    Args:
        data (2D np.ndarray): Image data array.
        header (astropy.io.fits.Header): FITS header with GAIN keyword.

    Returns:
        float: Median seeing in arcseconds (FWHM of stars).

    Written by Cameron Lemon
    """
    # Create a background-subtracted image to give to SExtractor
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # Error map from Poisson noise of objects and readnoise (approximate)
    errormap = (((data-median)/header['GAIN']) + std**2.)**0.5


    # Remove cosmic rays with lacosmic
    cleaned_image, _ = lacosmic.lacosmic(data-median, contrast=2., cr_threshold=5., neighbor_threshold=5., error=errormap)

    # Use a temporary filename for the cleaned image
    clean_filename = 'temp_cleaned.fits'
    from astropy.io import fits as pyfits
    hdu = pyfits.PrimaryHDU(cleaned_image, header=header)
    hdu.writeto(clean_filename, overwrite=True)

    catalog_name = clean_filename.split('.fits')[0]+'.cat'

    # Write the SExtractor parameter names to a file
    params = ['NUMBER', 'FLUX_AUTO', 'X_IMAGE', 'Y_IMAGE', 'CLASS_STAR', 'FWHM_WORLD', 'FWHM_IMAGE', 'FLUX_MAX', 'FLAGS']
    with open('sexparams.param', 'w') as file:
        file.write('\n'.join(params))

    os.system('sex '+clean_filename+' -CATALOG_NAME '+catalog_name+' -DETECT_MINAREA 5 -PARAMETERS_NAME sexparams.param')

    # Load the SExtractor catalogue
    catdata = np.loadtxt(catalog_name, unpack=True)

    # Parse data into an astropy table
    t = Table(names=params, data=catdata.T)
    t = t[t['FLAGS'] == 0.]

    # Find the star-like objects and measure FWHM
    starlike = (t['CLASS_STAR']>0.7)
    print('median FWHM of stars is', np.median(t[starlike]['FWHM_WORLD'])*3600, 'arcseconds')
    print('median FWHM in pixels is', np.median(t[starlike]['FWHM_IMAGE']), 'pixels')

    seeing = np.median(t[starlike]['FWHM_WORLD'])*3600
    return seeing


def process_asteroid_images(input_dir, output_dir, pixel_scale=0.21, exposure_time=320):
    """
    Process all FITS images in input_dir, inject synthetic asteroid streaks, and save results to output_dir and a CSV file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    exposure_time_h = exposure_time / 3600
    csv_filepath = os.path.join(output_dir, "asteroid_info.csv")
    with open(csv_filepath, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Filename', 'Start_X', 'Start_Y', 'End_X', 'End_Y', 'Slope', 'Total_Flux', 'Brightness_per_pixel', 'SNR', 'Length of track', 'Noise per pixel', 'Seeing'])
        for filename in os.listdir(input_dir):
            if filename.endswith(".fits"):
                print(f"Processing {filename}")
                base_filename = os.path.splitext(filename)[0]
                with fits.open(os.path.join(input_dir, filename)) as hdu:
                    image = hdu[0].data
                    header = hdu[0].header
                    seeing = cameronSeeing(image, header)
                    seeinginpixel = seeing / pixel_scale
                    sigma_Gaussian = seeinginpixel / (2 * (2 * np.log(2))**0.5)
                    width = 4 * seeinginpixel
                    kernel = Gaussian2DKernel(sigma_Gaussian)
                    asteroid_info = []
                    # Population 1: Brighter asteroids
                    for _ in range(random.randint(5, 10)):
                        SNR = np.random.uniform(3, 10)
                        length = np.random.uniform(3, 120)
                        max_attempts = 10
                        for _ in range(max_attempts):
                            asteroid_result = asteroidGenerator(image, SNR, length, width)
                            if asteroid_result is not None:
                                asteroid_data, start_x, start_y, end_x, end_y, slope, total_flux, brightness_per_pixel, length, noise_per_pixel = asteroid_result
                                break
                        else:
                            print(f"Failed to generate asteroid data after {max_attempts} attempts.")
                            continue
                        image = image + convolve(asteroid_data, kernel)
                        asteroid_info.append([base_filename, start_x, start_y, end_x, end_y, slope, total_flux, brightness_per_pixel, SNR, length, noise_per_pixel, seeing])
                    # Population 2: Other asteroids
                    for _ in range(random.randint(5, 10)):
                        SNR = np.random.uniform(3, 20)
                        length = np.random.uniform(3, 120)
                        max_attempts = 10
                        for _ in range(max_attempts):
                            asteroid_result = asteroidGenerator(image, SNR, length, width)
                            if asteroid_result is not None:
                                asteroid_data, start_x, start_y, end_x, end_y, slope, total_flux, brightness_per_pixel, length, noise_per_pixel = asteroid_result
                                break
                        else:
                            print(f"Failed to generate asteroid data after {max_attempts} attempts.")
                            continue
                        image = image + convolve(asteroid_data, kernel)
                        asteroid_info.append([base_filename, start_x, start_y, end_x, end_y, slope, total_flux, brightness_per_pixel, SNR, length, noise_per_pixel, seeing])
                        print(f"base_filename: {base_filename}, total_flux: {total_flux}, brightness_per_pixel: {brightness_per_pixel}, SNR: {SNR}, length: {length}, noise: {noise_per_pixel}, seeing: {seeing}")
                    csv_writer.writerows(asteroid_info)
                    asteroid_info.clear()
                new_filepath = os.path.join(output_dir, base_filename + ".fits")
                hdu = fits.PrimaryHDU(image)
                hdu.writeto(new_filepath, overwrite=True)

# Replace with your actual input and output directories
input_dir = '/Volumes/VST/'
output_dir = '/Volumes/VST/painted'
process_asteroid_images(input_dir, output_dir)
