from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from astropy import coordinates as coords
import numpy as np
import pandas as pd

from .utils import pointing, sun_sep, moon_sep

# Could be OO
def index_file(rd, altitude=90, azimuth=180):
    out_dict = {}
    utc_time = rd.times.to_datetime()
    pntg = pointing(rd.times, altitude, azimuth)
    ss, sa = sun_sep(pntg, times=rd.times, return_alt=True,
                     mask_when_set=False)
    out_dict['sun_sep'] = ss
    out_dict['sun_alt'] = sa
    ms, ma = moon_sep(pntg, times=rd.times, return_alt=True,
                      mask_when_set=False)
    out_dict['moon_sep'] = ms
    out_dict['moon_alt'] = ma
    out_dict['ra'] = pntg.ra.degree
    out_dict['dec'] = pntg.dec.degree
    galactic = pntg.transform_to(coords.Galactic)
    out_dict['l'] = galactic.l.degree
    out_dict['b'] = galactic.b.degree
    out_dict['data_index'] = np.arange(len(rd.times))
    out_df = pd.DataFrame(out_dict, index=pd.DatetimeIndex(utc_time))
    return out_df
