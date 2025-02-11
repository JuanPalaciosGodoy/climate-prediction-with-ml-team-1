### time selection functions

#imports
import numpy as np
import pandas as pd
import xarray as xr

def to_datetime64(ds):
    """
    Takes a ds with time as a string and returns a ds with coordinate iso_time
    that has type datetime64
    """
    
    iso_time_flat = ds['iso_time'].values.flatten()
    iso_time_flat_str = iso_time_flat.astype(str)
    iso_time_flat_dt = pd.to_datetime(iso_time_flat_str, errors='coerce').to_numpy().reshape(ds['iso_time'].shape)

    ds['iso_time'] = (('storm', 'date_time'), iso_time_flat_dt)
    ds = ds.set_coords('iso_time')

    return ds

def add_origin_year(ds):
    """
    Takes a ds with coordinate of time and selects the first year in the time
    array. This new coordinate is called origin_year
    """
    
    origin_years = []
    for storm in range(0,len(ds.storm)):
        origin_year = int(ds.iso_time.dt.year[storm][0])
        origin_years.append(origin_year)

    ds['origin_year'] = ('storm', origin_years)
    ds = ds.set_coords('origin_year')

    return ds

def select_years(ds, start_year, end_year):
    """
    Takes a ds and selects only the data between the start_year and end_year inputs
    based on the origin_year coordinate.
    """
    
    mask = (ds.origin_year >= start_year) & (ds.origin_year <= end_year)
    ds_filtered = ds.where(mask, drop=True)

    return ds_filtered