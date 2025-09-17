# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# MAGasteroidGenerator.py
#
# This script generates synthetic asteroid streaks for astronomical image analysis,
# specifically for simulating the appearance of asteroids in ZTF-like survey images.
# The user can specify a coarse magnitude-length distribution to generate the streaks.
# For a given magnitude, it calculates the appropriate brightness per pixel and injects 
# synthetic streaks into the images, making them realistic by convolving them with the
# PSF of each image. The orientation of the streaks is random. The resultant images can
# be used for testing detection algorithms or training machine-learning models.
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
# Created at IPAC (Infrared Processing and Analysis Center), Caltech.
# Author: Belén Yu Irureta-Goyena
# Date: September 2023
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
import random
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from astropy.coordinates import SkyCoord
import astropy.units as u
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


def generate_random_point_within_pentagon(vertices, num_points, density_params):
    """
    Generate random (magnitude, length) points within an irregular pentagon, with enhanced density in specified rectangular subregions.

    Args:
        vertices (list of tuple): Vertices of the pentagon [(x1, y1), ...].
        num_points (int): Number of random points to generate.
        density_params (list of float): Relative densities for each rectangle.

    Returns:
        list of tuple: List of (x, y) points representing (magnitude, length).
    """
    # Create a shapely Polygon from the pentagon vertices
    polygon = Polygon(vertices)

    # Define three rectangular subregions inside the pentagon, each with a density weight
    rectangles = [
        {'vertices': [(17, 55), (17, 40), (17.8, 40), (17.8, 55)], 'density': density_params[0]},
        {'vertices': [(18, 10), (18, 30), (19.5, 30), (19.5, 10)], 'density': density_params[1]},
        {'vertices': [(17.5, 25), (17.5, 38), (18.8, 25), (18.8, 38)], 'density': density_params[2]}
    ]

    random_points = []

    while len(random_points) < num_points:
        # Decide whether to sample from the pentagon background or from one of the rectangles
        # The probability of choosing a rectangle is proportional to the sum of density_params
        region_choice = random.choices(['pentagon', 'rectangle'], weights=[1, sum(density_params)])[0]

        if region_choice == 'pentagon':
            # Uniformly sample a point within the bounding box of the pentagon
            random_x = random.uniform(min(p[0] for p in vertices), max(p[0] for p in vertices))
            random_y = random.uniform(min(p[1] for p in vertices), max(p[1] for p in vertices))
        else:
            # Choose one rectangle, weighted by its density, and sample uniformly within its bounding box
            rectangle = random.choices(rectangles, weights=[rect['density'] for rect in rectangles])[0]['vertices']
            random_x = random.uniform(min(p[0] for p in rectangle), max(p[0] for p in rectangle))
            random_y = random.uniform(min(p[1] for p in rectangle), max(p[1] for p in rectangle))

        point = Point(random_x, random_y)

        # Only accept the point if it falls inside the pentagon
        if polygon.contains(point):
            random_points.append((random_x, random_y))

    return random_points

def positionAngle(ra1, dec1, ra2, dec2):
    """
    Calculate the position angle of a segment defined by its endpoints in RA, Dec.

    Parameters:
    - ra1, dec1: RA and Dec of the first endpoint in degrees.
    - ra2, dec2: RA and Dec of the second endpoint in degrees.

    Returns:
    - Position angle in degrees.
    """
    # Convert RA and Dec to SkyCoord objects
    coord1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree, frame='icrs')
    coord2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree, frame='icrs')

    # Calculate differences in RA and Dec between the two endpoints
    delta_ra = coord2.ra - coord1.ra
    delta_dec = coord2.dec - coord1.dec

    # Calculate the position angle in radians using arctangent
    pa_rad = np.arctan2(delta_ra.to(u.rad).value, delta_dec.to(u.rad).value)

    # Convert the position angle to degrees and ensure it's in the range [0, 360)
    pa_deg = np.degrees(pa_rad) % 360

    return pa_deg


def wcs_pix2world(x, y, cd, crpix, crval):
    """
    Convert pixel coordinates to celestial coordinates using WCS transformation.

    Parameters:
    - x, y: Pixel coordinates.
    - cd: CD matrix coefficients.
    - crpix: Reference pixel coordinates.
    - crval: Reference celestial coordinates.

    Returns:
    - Tuple of (RA, Dec) in degrees.
    """
    x, y = np.asarray(x), np.asarray(y)
    cd = np.asarray(cd).reshape((2, 2))
    crpix = np.asarray(crpix)
    crval = np.asarray(crval)

    # Compute celestial coordinates (RA, Dec) using the WCS transformation
    dra = x - crpix[0]
    ddec = y - crpix[1]
    ra = crval[0] + (cd[0, 0] * dra + cd[0, 1] * ddec)
    dec = crval[1] + (cd[1, 0] * dra + cd[1, 1] * ddec)

    return ra, dec

# Function that takes an image and paints asteroids on it
def asteroidGenerator(image, zero_point, magnitude, length):
    """
    Paint a synthetic asteroid streak onto an astronomical image.

    The function randomly selects a starting position and orientation (slope) for the streak,
    computes the corresponding end position based on the desired length, and ensures the trail
    remains within image boundaries. The brightness of the streak is set according to the input
    magnitude and zero point, and the trail is injected into the image as a straight line.

    Args:
        image (2D np.ndarray): The astronomical image to modify.
        zero_point (float): Photometric zero point for flux calculation.
        magnitude (float): Desired apparent magnitude of the asteroid.
        length (float): Desired length of the streak in pixels.

    Returns:
        tuple or None: (traildata, start_x, start_y, end_x, end_y, slope, total_flux, brightness_per_pixel, length)
        if the streak is successfully painted, else None.
    """
    # Get the shape of the image
    image_shape = image.shape

    # Initialize an empty array to hold the trail (same shape as image)
    traildata = np.zeros(image_shape)

    # Randomly generate a slope (orientation) for the trail, between -1 and 1
    slope = np.random.uniform(-1, 1)

    # Randomly select a starting position (pixel coordinates) within the image
    start_y = np.random.randint(0, image_shape[0])
    start_x = np.random.randint(0, image_shape[1])

    print(f'Trail start position: ({start_x}, {start_y})')
    print(f'Trail slope: {slope}')

    # Calculate the end position based on the desired length and slope
    # The trail is a straight line: length = sqrt((dx)^2 + (dy)^2)
    end_x = int(start_x + length / np.sqrt(1 + slope**2))
    end_y = int(start_y + slope * (end_x - start_x))

    # Adjust endpoints to ensure the trail stays within image boundaries
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

    # Calculate the actual length of the trail after boundary adjustment
    actual_length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

    print(f'Trail end position: ({end_x}, {end_y})')
    print(f'Theoretical length: {length}')
    print(f'Real length: {actual_length}')

    # Calculate the total flux for the asteroid using the photometric equation
    total_flux = 10**((zero_point - magnitude) / 2.5)

    # Distribute the flux evenly along the trail
    brightness_per_pixel = total_flux / length

    # Generate the pixel coordinates along the trail
    xs, ys = [], []
    for x in range(start_x, end_x):
        yaffected = start_y + slope * (x - start_x)
        xs.append(x)
        ys.append(int(yaffected))

    # Check for NaN values in the image (to avoid painting over bad pixels)
    nan_mask = np.isnan(image)

    streaks_painted = False

    # Flip x/y axes for the trail to cover the other half of possible orientations
    xs_flipper = xs
    ys_flipper = ys
    start_x_flipper = start_x
    start_y_flipper = start_y
    end_x_flipper = end_x
    end_y_flipper = end_y

    control_parameter = np.random.randint(0, 2)
    if control_parameter > 0:
        # Flip x and y for the trail in 50 % of the cases
        xs = ys_flipper
        ys = xs_flipper
        start_x = start_y_flipper
        start_y = start_x_flipper
        end_x = end_y_flipper
        end_y = end_x_flipper
        print('flipped')
    else:
        print('not flipped')

    # Paint the trail only if all affected pixels are not NaN
    if not any(nan_mask[y, x] for x, y in zip(xs, ys)):
        for x, y in zip(xs, ys):
            traildata[y, x] = brightness_per_pixel
            streaks_painted = True

    # Convolve the trail with the PSF kernel and add to the image (if streak was painted)
    if streaks_painted:
        image[traildata != 0] += convolve(traildata, kernel)[traildata != 0]

    # Return all relevant metadata if successful, else None
    return (
        traildata,
        start_x,
        start_y,
        end_x,
        end_y,
        slope,
        total_flux,
        brightness_per_pixel,
        length,
    ) if streaks_painted else None


input_dir = '/Volumes/Extreme Pro/ZTF'
output_dir = '/Volumes/Extreme Pro/ZTF/painted/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


exposure_time_h = 30 / 3600

csv_filepath = os.path.join(output_dir, "ztf_painted.csv")

# Vertices of the irregular pentagon
vertices = [(15, 60), (17, 5), (20, 5), (17, 120), (16, 120)]

# Density parameters for the three rectangles
density_params = [0.25, 0.45, 0.3]

# Number of random points to generate
num_points = 1

with open(csv_filepath, "w", newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Filename', 'Start_X', 'Start_Y', 'End_X', 'End_Y', 'Slope', 'Total_Flux', 'Brightness_per_pixel', 'Magnitude', 'Length of track', 'Zero Point', 'Seeing', 'Position angle'])

    # Loop over the FITS files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".fits"):
            print(f"Processing {filename}")
            base_filename = os.path.splitext(filename)[0]

            # Read the FITS file
            with fits.open(os.path.join(input_dir, filename)) as hdul:
                header = hdul[0].header
                image = hdul[0].data

                # Extract zero_point and seeinginpixel from the header
                zero_point = header.get('MAGZP')  
                seeinginpixel = header.get('SEEING')  
                cd = header['CD1_1'], header['CD1_2'], header['CD2_1'], header['CD2_2']
                crpix = header['CRPIX1'], header['CRPIX2']
                crval = header['CRVAL1'], header['CRVAL2']



                # Update width using seeinginpixel
                width = 2 * seeinginpixel / (2 * (2 * np.log(2))**0.5)
                sigma_Gaussian = seeinginpixel / (2 * (2 * np.log(2))**0.5)
                kernel = Gaussian2DKernel(sigma_Gaussian)

                # Create lists to store asteroid information for this image
                asteroid_info = []


                # Iterate over all asteroid populations
                iterations_all_asteroids = random.randint(5, 15)
                for i in range(iterations_all_asteroids):
                    # Generate random points within the pentagon
                    random_magnitude_length = generate_random_point_within_pentagon(vertices, num_points, density_params)
                    #print(random_magnitude_length)
                    magnitude = random_magnitude_length[0][0]
                    length = random_magnitude_length[0][1]

                    try: 
                        asteroid_data, start_x, start_y, end_x, end_y, slope, total_flux, brightness_per_pixel, length = asteroidGenerator(image, zero_point, magnitude, length)
                        image = image + convolve(asteroid_data, kernel)

                        ra1, dec1 = wcs_pix2world(start_x, start_y, cd, crpix, crval)
                        ra2, dec2 = wcs_pix2world(end_x, end_y, cd, crpix, crval)

                        position_angle = positionAngle(ra1, dec1, ra2, dec2)

                        # Append to the list of asteroid information for this image
                        asteroid_info.append([base_filename, start_x, start_y, end_x, end_y, slope, total_flux, brightness_per_pixel, magnitude, length, zero_point, seeinginpixel, position_angle])
                        print(f"base_filename: {base_filename}, total_flux: {total_flux}, brightness_per_pixel: {brightness_per_pixel}, magnitude: {magnitude}, length: {length}, zero_point: {zero_point}, seeing: {seeinginpixel}, position_angle: {position_angle}")
                    
                    except:
                        print('Track in NaN')
                # Write asteroid information for this image to the CSV file
                csv_writer.writerows(asteroid_info)

                # Clear the asteroid_info list so it's ready for the next file
                asteroid_info.clear()

            # Save the FITS file
            new_filepath = os.path.join(output_dir, base_filename + ".fits")
            hdu = fits.PrimaryHDU(image)
            hdu.writeto(new_filepath, overwrite=True)
