from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from itertools import combinations_with_replacement

import numpy as np
import tempfile
import shutil
import h5py
import matplotlib.pyplot as plt
from matplotlib import dates
from astropy.time import Time
from astropy.visualization import PercentileInterval, ImageNormalize

from .utils import butterworth_filter

class RawData(object):
    """
    Class for representing and manipulating raw HIRAX data files
    """
    def __init__(self, filename, time_subsample=1,
                 freq_subsample=1, transpose_on_init=False,
                 overwrite_original=False):
        """
        Creates a RawData object given the filename of a HIRAX 
        formatted hdf5 file

        Parameters
        ----------
        filename : str
            Location of the HIRAX datafile to use
        time_subsample : int (optional)
            Stride to use in the time dimension if subsampling is
            desired. Useful for quick, low resolution analyses.
            Default: 1
        freq_subsample : int (optional)
            Stride to use in the frequency dimension if subsampling is
            desired. Useful for quick, low resolution analyses.
            Default: 1
        transpose_on_init : bool (optional)
            Whether to transpose the datafiles from there raw structure
            to a transposed structure where the products dimension is
            the slowest moving. This will make most data reading tasks 
            quicker but will create a large temporary file that is not 
            automatically cleaned up.
            Default: False
        overwrite_original : bool (optional)
            Instead of creating a temporary file in when transposing,
            overwrite the input file. Use with caution.
            Default: False
        """

        if transpose_on_init and not overwrite_original:
            print('Creating temporary file for transpose.')
            tmp_file = tempfile.NamedTemporaryFile(delete=False)
            self.is_temp_file = True
            shutil.copyfile(filename, tmp_file.name)
            self.filename = tmp_file.name
            with h5py.File(self.filename, 'a') as raw_hdf:
                if 'vis_transposed' not in raw_hdf.keys():
                    print('Transposing...')
                    vis = raw_hdf['vis']
                    nshape = (vis.shape[2], vis.shape[1], vis.shape[0])
                    raw_hdf.create_dataset(
                        'vis_transposed', chunks=True,
                        shape=nshape, dtype=vis.dtype,
                        data=np.swapaxes(vis.value, 2, 0))
        else:
            self.filename = filename
            self.is_temp_file = False

        with h5py.File(self.filename, 'r') as raw_hdf:
            self.metadata = dict(raw_hdf.attrs)
            self.baseline_list = list(combinations_with_replacement(
                range(np.asscalar(self.metadata['n_input'])), 2))
            self.time_slice = slice(
                0, np.asscalar(self.metadata['acq.frames_per_file']),
                time_subsample)
            self.freq_slice = slice(
                0, np.asscalar(self.metadata['n_freq']),
                freq_subsample)
            self.has_transposed = ('vis_transposed' in raw_hdf.keys())

    @property
    def ctimes(self):
        with h5py.File(self.filename, 'r') as raw_hdf:
            out = raw_hdf['index_map/time']['ctime'][
                self.time_slice]
        return out

    @property
    def times(self):
        return Time(self.ctimes, format='unix')

    @property
    def bands(self):
        with h5py.File(self.filename, 'r') as raw_hdf:
            out = raw_hdf['index_map/freq']['centre'][
                self.freq_slice]
        return out

    def raw_query(self, query):
        """
        Read an abitrary dataset from the wrapped HDF5 file.

        Parameters
        ----------
        query : str
            The location of the dataset to query.

        Returns
        -------
        `~numpy.array`
            numpy array of requested dataset
        """
        with h5py.File(self.filename, 'r') as raw_hdf:
            out = raw_hdf[query].value
        return out

    def baseline_from_prod(self, prod):
        """
        Returns the index of a baseline product from the tuple 
        represention.

        Parameters
        ----------
        prod : array_like (2,)
            The pair of input indices to convert to product index

        Returns
        -------
        int
            index of baseline pair in product dimension

        """
        try:
            return self.baseline_list.index(prod)
        except ValueError:
            return self.baseline_list.index(prod[::-1])

    def visibilities(self, baseline, which='complex'):
        """
        Return the visbility data from the raw data file

        Parameters
        ----------
        baseline : int or array_like (2,)
            Product index or list/tuple of baseline pari indices
            for which the visiobility will be returned
        which : str (optional)
            Format for return visibilities, can be 'complex', 
            ['mag', 'magnitude', 'amp', 'amplitude'], 'phase', ['real', 'r']
            or ['imaginary', 'i'].
            Default: 'complex'

        Returns
        -------
        `~numpy.array`
            Array of the visibility data for the requested baseline in
            the desired format.
        """
        if isinstance(baseline, (tuple, list)):
            baseline = self.baseline_from_prod(baseline)

        with h5py.File(self.filename, 'r') as raw_hdf:
            if self.has_transposed:
                index = (baseline, self.freq_slice, self.time_slice)
                dset = 'vis_transposed'
                real = raw_hdf[dset][index]['r'].T
                imag = raw_hdf[dset][index]['i'].T
            else:
                index = (self.time_slice, self.freq_slice, baseline)
                dset = 'vis'
                real = raw_hdf[dset][index]['r']
                imag = raw_hdf[dset][index]['i']

        out = real + 1j * imag
        if which.lower() == 'complex':
            return out
        elif which.lower() in ['mag', 'magnitude', 'amp', 'amplitude']:
            return np.abs(out)
        elif which.lower() == 'phase':
            return np.angle(out)
        elif which.lower() in ['real', 'r']:
            return out.real
        elif which.lower() in ['imag', 'i', 'imaginary']:
            return out.imag

    def time_axis(self, which, axis=None):
        """
        Convenience function for generating matplotlib axes
        with the correct time formatting extracted from metadata.

        Parameters
        ----------
        which : str
            Which axis should the time axis be constructed on. 
            Can be 'x' or 'y'.
        axis : `~matplotlib.axis.Axis`
            An existing mapltolib axis to modify. If None, create
            a new axis.
            Default: None

        Returns
        -------
        tuple (`~matplotlib.axis.Axis`, `~numpy.array`)
            Tuple of 
                the axis object with time information added,
                array of numdates to plot data against that are
                consistent with this axis.
        TODO: Could use an example usage...
        """
        if axis is None:
            fig, axis = plt.subplots(1, 1)

        if which.lower() == 'y':
            axis.yaxis_date()
            axis.set_ylabel('UTC')
            axis.yaxis.set_major_locator(
                dates.MinuteLocator(byminute=[0, 30]))
            axis.yaxis.set_major_formatter(
                dates.DateFormatter('%H:%M:%S'))
                    # Annotate with date of obs start
            axis.text(-0.015, 1.02, self.times[0].iso.split(' ')[0],
                      ha='right', va='bottom',
                      transform=axis.transAxes)
            return axis, dates.date2num(self.times.datetime)

        elif which.lower() == 'x':
            axis.xaxis_date()
            axis.set_xlabel('UTC')
            axis.xaxis.set_major_locator(
                dates.MinuteLocator(byminute=[0, 30]))
            axis.xaxis.set_major_formatter(
                dates.DateFormatter('%H:%M:%S'))
            axis.text(-0.015, 1.02, self.times[0].iso.split(' ')[0],
                      ha='right', va='bottom',
                      transform=axis.transAxes)
            return axis, dates.date2num(self.times.datetime)
        else:
            raise ValueError("'which' must be x or y.")
        

    def waterfall_plot(self, baseline,
                       threshold_percentile=100,
                       mask_percentile=100,
                       remove_median=False, axis=None,
                       baseline_title=False, which='mag',
                       filt_kwargs=None, mask=None,
                       other_data=None):

        if type(baseline) is tuple:
            baseline = self.baseline_from_prod(baseline)

        if axis is None:
            fig, axis = plt.subplots(1, 1)
        else:
            fig = axis.figure

        if filt_kwargs is not None:
            image = self.filtered_on_time(
                baseline, which=which, other_data=other_data, **filt_kwargs)
        elif other_data is None:
            image = self.visibilities(baseline, which=which)
        else:
            image = other_data

        if remove_median:
            image -= np.median(image, axis=0)

        if mask is None:
            mask = np.abs(image) > np.percentile(np.abs(image), q=mask_percentile)
        image[mask] = np.NaN

        norm = ImageNormalize(
            image,
            interval=PercentileInterval(threshold_percentile))

        extent = (self.bands[0], self.bands[-1],
                  dates.date2num(self.times.datetime[-1]),
                  dates.date2num(self.times.datetime[0]))

        aspect = np.abs(
            (extent[1] - extent[0]) / (extent[3] - extent[2]))

        _ = axis.imshow(image, norm=norm, cmap='RdBu_r',
                        aspect=aspect, extent=extent)

        axis, _ = self.time_axis(which='y', axis=axis)

        axis.set_xlabel('Frequency [MHz]')
        if baseline_title:
            axis.set_title('Baseline: ({:d}, {:d})'.format(
                *self.baseline_list[baseline]))

        return fig

    def filtered_on_time(self, baseline, filter_type='band',
                         char_freqs=1, which='complex', other_data=None):
        dtime = self.ctimes[1] - self.ctimes[0]
        if other_data is None:
            to_filt = self.visibilities(baseline, which=which)
        else:
            to_filt = other_data
        filtered = butterworth_filter(
            to_filt, filter_type=filter_type, char_freqs=char_freqs,
            axis=0, sample_rate=dtime)
        if which.lower() != 'complex':
            return filtered.real
        else:
            return filtered

    def filtered_on_band(self, baseline, filter_type,
                         char_freqs, which='complex', other_data=None):
        dfreq = self.bands[1] - self.bands[0]
        if other_data is None:
            to_filt = self.visibilities(baseline, which=which)
        else:
            to_filt = other_data
        filtered = butterworth_filter(
            to_filt, filter_type=filter_type, char_freqs=char_freqs,
            axis=1, sample_rate=dfreq)
        if which.lower() != 'complex':
            return filtered.real
        else:
            return filtered