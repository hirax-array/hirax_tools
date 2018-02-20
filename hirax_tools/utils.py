from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from itertools import combinations_with_replacement, combinations
from astropy import units
from astropy import coordinates as coords
from astropy.stats import gaussian_fwhm_to_sigma
from scipy import signal
import numpy as np

# From: http://www.hartrao.ac.za/HartRAO_Coordinates.html, using VLBI pos for now
# will be off by 10s of meters
HARTRAO_COORD = coords.EarthLocation.from_geodetic(
    lon=27.6853931*units.deg,
    lat=-25.8897515*units.deg,
    height=1415.710*units.m,
    ellipsoid='WGS84')

def baseline_cross_list(channel_indices, auto=True):
    if auto:
        combs = combinations_with_replacement(channel_indices, 2)
    else:
        combs = combinations(channel_indices, 2)
    return list(combs)

def gaussian_beam(theta, primary_beam_fwhm):
    theta = units.Quantity(theta, unit=units.deg)
    fwhm = units.Quantity(primary_beam_fwhm, unit=units.deg)
    x = (-(theta)**2/2/(gaussian_fwhm_to_sigma*fwhm)**2).to('').value
    return np.exp(x)

def pointing(times, altitude, azimuth, location=HARTRAO_COORD, frame='ICRS'):
    altitude = coords.Angle(altitude, unit=units.deg)
    azimuth = coords.Angle(azimuth, unit=units.deg)
    altazs = coords.SkyCoord(
        frame='altaz',
        alt=altitude*np.ones(len(times)),
        az=azimuth*np.ones(len(times)),
        location=location, obstime=times)
    new_frame = coords.frame_transform_graph.lookup_name(frame.lower())
    return altazs.transform_to(new_frame)

def lmn_coordinates(pointing, target, location=HARTRAO_COORD):
    offset_coordinates = target.transform_to(pointing.skyoffset_frame())
    cart = offset_coordinates.represent_as(coords.representation.CartesianRepresentation)
    l, m, n = cart.y.value, cart.z.value, cart.x.value - 1
    return l, m, n

def butterworth_filter(data, filter_type, char_freqs, order=4, sample_rate=1,
                       axis=None):
    if axis is None:
        axis_size = len(data)
        axis = -1
    else:
        axis_size = data.shape[axis]

    # FT data on specified axis
    vis_fft = np.fft.fft(data, axis=axis)

    # scipy signal processing implementation details...
    b, a = signal.butter(N=order, Wn=char_freqs, btype=filter_type, analog=True)
    freqs = np.fft.fftfreq(axis_size, d=sample_rate)
    _, h = signal.freqs(b, a, freqs)

    # This ensures things broadcast correctly 
    # depending on the specified axis.
    broadcast_axes = [None for i in range(data.ndim)]
    broadcast_axes[axis] = slice(None, None, -1)

    # filter data on axis specified
    filtered_vis_fft = vis_fft*(np.abs(h)[broadcast_axes]**2)

    return np.fft.ifft(filtered_vis_fft, axis=axis)
