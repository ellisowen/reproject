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
    for i in range(nx_in):
        j = np.arange(ny_in)

        # For every input pixel we find the position in the output image in
        # pixel coordinates, then use the full range of overlapping output
        # pixels with the exact overlap function.
        xmin_a = np.minimum(xp_inout[j[:-1]+1, i], xp_inout[j[1:], i+1])
        xmin_b = np.minimum(xp_inout[j[:-1]+1, i+1], xp_inout[j[:-1]+1, i])
        xmin = np.array(np.minimum(xmin_a, xmin_b), dtype=int)
        xmax_a = np.maximum(xp_inout[j[1:], i], xp_inout[j[1:], i+1])
        xmax_b = np.maximum(xp_inout[j[:-1]+1, i+1], xp_inout[j[:-1]+1, i])
        xmax = np.array(np.maximum(xmax_a, xmax_b), dtype=int)
        ymin_a = np.minimum(yp_inout[j[1:], i], yp_inout[j[1:], i+1])
        ymin_b = np.minimum(yp_inout[j[:-1]+1, i+1], yp_inout[j[:-1]+1, i])
        ymin = np.array(np.minimum(ymin_a, ymin_b), dtype=int)
        ymax_a = np.maximum(yp_inout[j[1:], i], yp_inout[j[1:], i+1])
        ymax_b = np.maximum(yp_inout[j[:-1]+1, i+1], yp_inout[j[:-1]+1, i])
        ymax = np.array(np.maximum(ymax_a, ymax_b), dtype=int)

        ilon = [[xw_in[j[1:], i][0], xw_in[j[1:], i+1][0], xw_in[j[:-1]+1, i+1][0], xw_in[j[:-1]+1, i][0]][::-1]]
        ilat = [[yw_in[j[1:], i][0], yw_in[j[1:], i+1][0], yw_in[j[:-1]+1, i+1][0], yw_in[j[:-1]+1, i][0]][::-1]]
        # possible bug here with ilat
        ilon = np.radians(np.array(ilon))
        ilat = np.radians(np.array(ilat))

        xmin_array = np.maximum(np.zeros_like(xmin), xmin)
        xmax_array = np.minimum(np.ones_like(xmax) * (nx_out-1), xmax)
        ymin_array = np.maximum(np.zeros_like(ymin), ymin)
        ymax_array = np.minimum(np.ones_like(ymax) * (ny_out-1), ymax)
        indices = np.arange(len(xmin))
        for index in indices:
            xmin = xmin_array[index]
            xmax = xmax_array[index]
            ymin = ymin_array[index]
            ymax = ymax_array[index]
            for ii in range(xmin, xmax+1):
                for jj in range(ymin, ymax+1): 
                    olon = [[xw_out[jj, ii], xw_out[jj, ii+1], xw_out[jj+1, ii+1], xw_out[jj+1, ii]][::-1]]
                    olat = [[yw_out[jj, ii], yw_out[jj, ii+1], yw_out[jj+1, ii+1], yw_out[jj+1, ii]][::-1]]
                    olon = np.radians(np.array(olon))
                    olat = np.radians(np.array(olat))

                    # Figure out the fraction of the input pixel that makes it
                    # to the output pixel at this position.
                    import IPython; IPython.embed()
                    _, overlap = _compute_overlap(ilon, ilat, olon, olat)
                    _, original = _compute_overlap(ilon, ilat, ilon, ilat)
                    array_new[jj, ii] += array[j, i] * overlap / original
                    weights[jj, ii] += overlap / original
    array_new /= weights

    return np.nan_to_num(array_new)
