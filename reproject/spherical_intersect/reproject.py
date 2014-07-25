# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS

from ..wcs_utils import wcs_to_celestial_frame, convert_world_coordinates

from ._overlap import _compute_overlap

__all__ = ['reproject_celestial']


def reproject_celestial(array, wcs_in, wcs_out, shape_out):
    """
    Reproject celestial slices from an n-d array from one WCS to another using
    flux-conserving spherical polygon intersection.

    Parameters
    ----------
    array : `~numpy.ndarray`
        The array to reproject
    wcs_in : `~astropy.wcs.WCS`
        The input WCS
    wcs_out : `~astropy.wcs.WCS`
        The output WCS
    shape_out : tuple
        The shape of the output array

    Returns
    -------
    array_new : `~numpy.ndarray`
        The reprojected array
    """

    # TODO: make this work for n-dimensional arrays
    if wcs_in.naxis != 2:
        raise NotImplementedError("Only 2-dimensional arrays can be reprojected at this time")

    # TODO: at the moment, we compute the coordinates of all of the corners,
    # but we might want to do it in steps for large images.

    # Start off by finding the world position of all the corners of the input
    # image in world coordinates

    ny_in, nx_in = array.shape

    x = np.arange(nx_in + 1.) - 0.5
    y = np.arange(ny_in + 1.) - 0.5

    xp_in, yp_in = np.meshgrid(x, y)

    xw_in, yw_in = wcs_in.wcs_pix2world(xp_in, yp_in, 0)

    # Now compute the world positions of all the corners in the output header

    ny_out, nx_out = shape_out

    x = np.arange(nx_out + 1.) - 0.5
    y = np.arange(ny_out + 1.) - 0.5

    xp_out, yp_out = np.meshgrid(x, y)

    xw_out, yw_out = wcs_out.wcs_pix2world(xp_out, yp_out, 0)

    # Convert the input world coordinates to the frame of the output world
    # coordinates.

    xw_in, yw_in = convert_world_coordinates(xw_in, yw_in, wcs_in, wcs_out)

    # Finally, compute the pixel positions in the *output* image of the pixels
    # from the *input* image.

    xp_inout, yp_inout = wcs_out.wcs_world2pix(xw_in, yw_in, 0)

    # Create output image

    array_new = np.zeros(shape_out)
    weights = np.zeros(shape_out)
    print(nx_in, ny_in)
    i = np.arange(nx_in)
    j = np.arange(ny_in)

    # For every input pixel we find the position in the output image in
    # pixel coordinates, then use the full range of overlapping output
    # pixels with the exact overlap function.
    
    xmin_a = np.nan_to_num(np.minimum(xp_inout[:-1, 1:], xp_inout[:-1, 1:]))
    xmin_b = np.nan_to_num(np.minimum(xp_inout[1:, 1:], xp_inout[1:, :-1]))
    xmin = np.array(np.minimum(xmin_a, xmin_b), dtype=int)
    xmax_a = np.nan_to_num(np.maximum(xp_inout[:-1, 1:], xp_inout[:-1, 1:]))
    xmax_b = np.nan_to_num(np.maximum(xp_inout[1:, 1:], xp_inout[1:, :-1]))
    xmax = np.array(np.maximum(xmax_a, xmax_b), dtype=int)
    ymin_a = np.nan_to_num(np.minimum(yp_inout[:-1, 1:], yp_inout[:-1, 1:]))
    ymin_b = np.nan_to_num(np.minimum(yp_inout[1:, 1:], yp_inout[1:, :-1]))
    ymin = np.array(np.minimum(ymin_a, ymin_b), dtype=int)
    ymax_a = np.nan_to_num(np.maximum(yp_inout[:-1, 1:], yp_inout[:-1, 1:]))
    ymax_b = np.nan_to_num(np.maximum(yp_inout[1:, 1:], yp_inout[1:, :-1]))
    ymax = np.array(np.maximum(ymax_a, ymax_b), dtype=int)
    
    ilon = [[xw_in[:-1, 1:], xw_in[:-1, 1:], xw_in[1:, 1:], xw_in[1:, :-1]][::-1]]
    ilat = [[yw_in[:-1, 1:], yw_in[:-1, 1:], yw_in[1:, 1:], yw_in[1:, :-1]][::-1]]
    ilon = np.radians(np.array(ilon))
    ilat = np.radians(np.array(ilat))
 
    olon = [[xw_out[:-1, 1:], xw_out[:-1, 1:], xw_out[1:, 1:], xw_out[1:, :-1]][::-1]]
    olat = [[yw_out[:-1, 1:], yw_out[:-1, 1:], yw_out[1:, 1:], yw_out[1:, :-1]][::-1]]
    olon = np.radians(np.array(olon))
    olat = np.radians(np.array(olat))

    
    # Figure out the fraction of the input pixel that makes it
    # to the output pixel at this position.
    
    
    import IPython; IPython.embed()
    # Loop over here - how does my input compare to the original? Check by switching branch
    overlap, _ = _compute_overlap(ilon[:][0][0][0], ilat[:][0][0][0], olon[:][0][0][0], olat[:][0][0][0])
    original, _ = _compute_overlap(ilon, ilat, ilon, ilat)
    array_new += array * overlap / original
    weights += overlap / original
    array_new /= weights

    return np.nan_to_num(array_new)
