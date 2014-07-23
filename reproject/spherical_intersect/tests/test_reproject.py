# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from gammapy.image.utils import cube_to_image
from reproject.spherical_intersect import reproject_celestial

# TODO: add reference comparisons


def test_reproject_celestial_slices_2d():

    #filename = get_pkg_data_filename('/home/eowen/software/python/reproject/reproject/tests/data/gc_ga.hdr')
    #filename = '/home/eowen/software/python/reproject/reproject/tests/data/gc_ga.hdr'
    filename = '/home/eowen/software/python/gammapy/gammapy/datasets/data/fermi/gll_iem_v02_cutout.fits'
    #header_in = fits.Header.fromtextfile(filename)
    fermi_image = cube_to_image(fits.open(filename)[0])
    header_in = fermi_image.header
    print(header_in)
    #filename = '/home/eowen/software/python/reproject/reproject/tests/data/gc_eq.hdr'
    filename = '/home/eowen/software/python/gammapy/gammapy/datasets/data/fermi/fermi_exposure.fits.gz'
    exposure_im = cube_to_image(fits.open(filename)[0])
    header_out = exposure_im.header
    print(header_out)
    #header_out = fits.Header.fromtextfile(filename)

    #array_in = np.ones((100, 100))
    array_in = fermi_image.data
    exposure_array = exposure_im.data
    #fits.ImageHDU(data = array_in, header = header_in)
    fits.writeto('in.fits', array_in, header_in, clobber=True)
    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)
    shape_out = exposure_array.shape#(1000, 1000)

    array_out = reproject_celestial(array_in, wcs_in, wcs_out, shape_out)
    
    #fits.ImageHDU(data = array_out, header = header_out)
    fits.writeto('out.fits', array_out, header_out, clobber=True)
    assert array_out.shape == shape_out


if __name__ == '__main__':
    test_reproject_celestial_slices_2d()