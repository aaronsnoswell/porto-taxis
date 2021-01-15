import os
import copy
import difflib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PORTO_IMG_FILE_GREY = "porto.full.grey.png"
PORTO_IMG_FILE_COLOR = "porto.full.color.png"

# The left, right, bottom, top data coordinates of the Porto image
PORTO_LRBT = [-8.724, -8.488, 41.0370, 41.2706]

# The porto image aspect: height / width
PORTO_IMG_ASPECT = 7371 / 5614


def load_porto_img(
    prefix=os.path.join(os.path.dirname(os.path.realpath(__file__)), "img"), grey=True
):
    """Load the porto image object for plotting
    
    Args:
        prefix (str): Path to 'img' folder containing porto images
        grey (bool): If true, render image as greyscale
    
    Returns:
        (matplotlib.image): Loaded image object, ready for ploting with imshow
    """

    # The porto image file
    if grey:
        PORTO_IMG_FILE = os.path.join(prefix, PORTO_IMG_FILE_GREY)
    else:
        PORTO_IMG_FILE = os.path.join(prefix, PORTO_IMG_FILE_COLOR)

    return mpimg.imread(PORTO_IMG_FILE)


def plot_porto_img(img, **kwargs):
    """Plot a rasterized map of the city of Porto into the current axes
    
    This plots the image with GPS data units so GPS paths can be directly plotted
    (using a flat-earth assumption) on top.
    
    Args:
        img (numpy array): Pre-loaded raster map image from load_porto_img
    """
    plt.imshow(img, extent=PORTO_LRBT, aspect=PORTO_IMG_ASPECT, **kwargs)

    plt.ylabel("Latitude")
    plt.xlabel("Longitude")


def zoom_to_lines(ax=None, pad=0.1):
    """Adjust axis limits to zoom in on some plotted lines on a matplotlib axis
    
    This is a helper method as plot_porto_img() makes an imshow call which will
    mess with the pyplot auto data limits.
    
    Args:
        ax (matplotlib.Axes): Axis object where paths are plotted, or current axis
        pad (float): Padding amount to leave around the lines
    """

    if ax is None:
        ax = plt.gca()
    lines = ax.lines

    minx = min([min(l.get_xdata()) for l in lines])
    maxx = max([max(l.get_xdata()) for l in lines])
    xrng = maxx - minx
    xmn = (maxx + minx) / 2

    miny = min([min(l.get_ydata()) for l in lines])
    maxy = max([max(l.get_ydata()) for l in lines])
    yrng = maxy - miny
    ymn = (maxy + miny) / 2

    xpad = xrng * (0.5 + pad)
    ypad = yrng * (0.5 + pad)
    plt.xlim(xmn - xpad, xmn + xpad)
    plt.ylim(ymn - ypad, ymn + ypad)


def geoid_dist(lat1, lon1, lat2, lon2, *, r=6378.1):
    """Compute geoid (great-circle) distance between GPS points in km
    From http://www.ridgesolutions.ie/index.php/2013/11/14/algorithm-to-calculate-speed-from-two-gps-latitude-and-longitude-points-and-time-difference/
    Args:
        lat1 (numpy array): Latitude of start points
        lon1 (numpy array): Longitude of start points
        lat2 (numpy array): Latitude of end points
        lon2 (numpy array): Longitude of end points
        r (float): Geoid radius in km, defaults to 6378.1 (radius of earth)
    Returns:
        (numpy array): Distance between points in meters
    """

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = np.deg2rad([lat1, lon1, lat2, lon2])

    # Compute P
    rho1 = r * np.cos(lat1)
    z1 = r * np.sin(lat1)
    x1 = rho1 * np.cos(lon1)
    y1 = rho1 * np.sin(lon1)

    # Compute Q
    rho2 = r * np.cos(lat2)
    z2 = r * np.sin(lat2)
    x2 = rho2 * np.cos(lon2)
    y2 = rho2 * np.sin(lon2)

    # Dot product
    dot = x1 * x2 + y1 * y2 + z1 * z2
    cos_theta = dot / (r * r)

    # Prevent float rounding errors with arccos
    cos_theta = np.clip(cos_theta, -1, 1)

    # Compute angle
    theta = np.arccos(cos_theta)

    # Distance in KM
    return r * theta
