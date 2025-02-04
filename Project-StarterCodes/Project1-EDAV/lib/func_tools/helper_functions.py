### helper functions

# Python Imports
import numpy as np

def get_storm_ids(storm):
    storm_ids = storm.sid.values
    lon_lst = storm.lon.values
    return lon_lst[~np.isnan(lon_lst)]

def filter_storms(storms, years:list=[], storm_ids: list=[], clusters: list=[]):

    """
    select multiple storms by sid or by year

    Parameters
    ----------
        storms (xarray): storms in xarray format
    
        storm_ids (list): list of storm's sid that are fetched

        years (list): list of years that are fetched

    Returns
    -------
        (xarray) filtered storms
    
    """
    
    # update years if empty to avoid filtering if nothing is passed
    years = storms.season.values if len(years) == 0 else years
    
    # update storm ids if empty to avoid filtering if nothing is passed
    storm_ids = storms.sid.values if len(storm_ids) == 0 else storm_ids
    
    # select the hurricanes that happened in selected years
    filtered_storms = storms.where(storms.season.isin(years), drop=True)
    filtered_storms = filtered_storms.where(filtered_storms.sid.isin(storm_ids), drop=True)

    # filter by cluster if they exist
    try:
        clusters = storms.cluster.values if len(clusters) == 0 else clusters
        filtered_storms = filtered_storms.where(filtered_storms.cluster.isin(clusters), drop=True)
    except:
        return filtered_storms
    
    return filtered_storms

def get_lon_lat(storm):
    lon_lst = storm.lon.values
    lat_lst = storm.lat.values
    return lon_lst[~np.isnan(lon_lst)], lat_lst[~np.isnan(lat_lst)]

    # !!! Note that even though it's a convention to place latitude before longitude,
    # to work with cartopy projection, longitude MUST be placed first. !!!

def get_moments(storm):
    # A function to calculate the track moments given a storm
    # OUTPUT:
    # X-centroid, Y-centroid, X_var, Y_var, XY_var
    
    # Note that:
    # In this case, no weights are set. In other words, all weights are 1.
    # A weight variable would need to be added in order to explore other weights
    
    lon_lst, lat_lst = get_lon_lat(storm)
    # If the track only has one point, there is no point in calculating the moments
    if len(lon_lst)<= 1: return None
          
    # M1 (first moment = mean). 
    # No weights applied
    lon_weighted, lat_weighted = np.mean(lon_lst), np.mean(lat_lst)
        
    # M2 (second moment = variance of lat and of lon / covariance of lat to lon
    # No weights applied
    cv = np.ma.cov([lon_lst, lat_lst])
        
    return [lon_weighted, lat_weighted, cv[0, 0], cv[1, 1], cv[0, 1]]

def assign_cluster(storms, clusters):
    return storms.assign(cluster = clusters)

def standardize(variables: np.array):
    mu = np.mean(variables, axis=0)
    std = np.std(variables, axis=0)
    return (variables - mu)/std

def normalize(variables: np.array):
    mx = np.max(variables, axis=0)
    mn = np.min(variables, axis=0)
    return (variables - mn) / (mx - mn)