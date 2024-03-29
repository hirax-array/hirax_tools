
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from itertools import combinations_with_replacement, combinations
from astropy import units
from astropy import coordinates as coords
from astropy.stats import gaussian_fwhm_to_sigma
from scipy import signal
import numpy as np
from colorsys import hls_to_rgb

# From: http://www.hartrao.ac.za/HartRAO_Coordinates.html, using VLBI pos for now
# will be off by 10s of meters
HARTRAO_COORD = coords.EarthLocation.from_geodetic(
    lon=27.6853931*units.deg,
    lat=-25.8897515*units.deg,
    height=1415.710*units.m,
    ellipsoid='WGS84')

BLEIEN_COORD = coords.EarthLocation.from_geodetic(
    lon=8.11*units.deg,
    lat=47.34*units.deg,
    height=469*units.m)

def colorize(z, s=1.0, zmin=None, zmax=None, zero='black'):
    # Adapted from https://stackoverflow.com/questions/17044052/mathplotlib-imshow-complex-2d-array
    h = (np.angle(z) + np.pi)  / (2 * np.pi) + 0.5
    if zmin is None:
        zmin = np.abs(z).min()
    if zmax is None:
        zmax= np.abs(z).max()
    magz = (np.abs(z) - zmin)/zmax # Range from 0 to 1
    magz[np.abs(z) <= zmin] = 0.
    magz[np.abs(z) >= zmax] = 1.
    if zero == 'white':
        l = 1 - 0.5*magz
    elif zero == 'black':
        l = 0.5*magz
    return np.array(np.vectorize(hls_to_rgb)(h,l,s)).swapaxes(0, 2).swapaxes(0, 1)

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

def sun_sep(pointing, times, location=HARTRAO_COORD, return_alt=False,
            mask_when_set=True):
    sun = coords.get_sun(times)
    sun_seps = pointing.separation(sun)
    sun_alts = sun.transform_to(coords.AltAz(location=location)).alt
    if mask_when_set:
        sun_seps[sun_alts < -1*units.degree] = np.NaN
    if not return_alt:
        return sun_seps
    else:
        return sun_seps, sun_alts

def moon_sep(pointing, times, location=HARTRAO_COORD, return_alt=False,
             mask_when_set=True):
    moon = coords.get_moon(times)
    moon_seps = pointing.separation(moon)
    moon_alts = moon.transform_to(coords.AltAz(location=location)).alt
    if mask_when_set:
        moon_seps[moon_alts < -1*units.degree] = np.NaN
    if not return_alt:
        return moon_seps
    else:
        return moon_seps, moon_alts

def lmn_coordinates(pointing, target):
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
    filtered_vis_fft = vis_fft*(np.abs(h)[tuple(broadcast_axes)]**2)

    return np.fft.ifft(filtered_vis_fft, axis=axis)
