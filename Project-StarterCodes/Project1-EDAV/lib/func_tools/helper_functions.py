### helper functions

# Python Imports
import numpy as np
import pandas as pd
import xarray
from sklearn.cluster import k_means # to perform k-means
from sklearn.cluster import DBSCAN # to perform dbscan

# Library Imports
from func_tools.plotting_functions import plot_kmeans_inertia

def get_storms_summary_data(storms:xarray.Dataset) -> pd.DataFrame:
    """
        storms summary data as pandas DataFrame. This includes max wind speed, cluster, 
    
        Parameters
        ----------
            storms (xarray.Dataset): storms in xarray format
    
        Returns
        -------
            (pd.DataFrame): pandas DataFrame with storm summary data (cluster, max wind speed)
    
    """
    clusters = np.array(storms.cluster.values)
    max_wind_speed = np.nanmax(storms.wmo_wind.values, axis=1)
    lon_std = np.array(storms.lon_std.values)
    lat_std = np.array(storms.lat_std.values)
    year = np.array(storms.avg_month.values).astype(int)
    month = np.array(storms.avg_year.values).astype(int)
    day = np.array(storms.avg_day.values).astype(int)

    
    return pd.DataFrame(np.array([clusters,
                                  max_wind_speed,
                                  lon_std,
                                  lat_std,
                                  year,
                                  month,
                                  day]).T, 
                        columns=["cluster", "max_wind_speed", "lon_std", "lat_std", "avg_year", "avg_month", "avg_day"])

def storms_kmeans(storms:xarray.Dataset, k_clusters:int, get_variables) -> xarray.Dataset:
    """
        run k-means over given variables. The variables are calculated by using the passed function get_variables() which takes as an input the parameter storms
    
        Parameters
        ----------
            storms (xarray.Dataset): storms in xarray format

            k_clusters (int): number of clusters

            get_variables (function): function to fetch the variables from the dataset storms.
    
        Returns
        -------
            (xarray.DataArray): xarray of storms with assigned cluster as kmeans_cluster
    
    """
    # run kmeans
    variables = get_variables(storms=storms)
    km = k_means(variables, n_clusters=k_clusters)
    cluster_labels = km[1]

    # get xarray of clusters
    cluster_xarray = _assign_cluster(storms=storms, cluster_labels=cluster_labels, name="kmeans_cluster", data_type="int")

    # assign clusters
    return storms.assign(cluster=(cluster_xarray))

def kmeans_inertia(storms:xarray.Dataset, get_variables, max_clusters:int=15):
    """
        run k-means from 1 cluster to max_clusters and plot kmeans inertia.
    
        Parameters
        ----------
            storms (xarray.Dataset): storms in xarray format

            max_clusters (int): maximum number of clusters

            get_variables (function): function to fetch the variables from the dataset storms.
    
        Returns
        -------
            (None)
    
    """
    
    variables = get_variables(storms=storms)
    sum_of_squares = []
    for k in range(1, max_clusters):
        # run kmeans
        km = k_means(variables, n_clusters=k)
        sum_of_squares.append(km[2])

    plot_kmeans_inertia(sum_of_squares=sum_of_squares, max_clusters=range(1, max_clusters))
    

def storms_dbscan(storms:xarray.Dataset, eps:float, min_samples:int, get_variables) -> xarray.Dataset:
    """
        run DBSCAN over given variables. The variables are calculated by using the passed function get_variables() which takes as an input the parameter storms
    
        Parameters
        ----------
            storms (xarray.Dataset): storms in xarray format

            eps (int): maximum distance between two samples for one to be considered as in the neighborhood of the other. 

            min_samples (int): number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

            get_variables (function): function to fetch the variables from the dataset storms.
    
        Returns
        -------
            (xarray.DataArray): xarray of storms with assigned cluster as kmeans_cluster
    
    """
    # run dbscan
    variables = get_variables(storms=storms)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(variables)
    cluster_labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(db.labels_)) - (1 if -1 in cluster_labels else 0)
    print(f"DBSCAN: found {n_clusters_} clusters")

    # get xarray of clusters
    cluster_xarray = _assign_cluster(storms=storms, cluster_labels=cluster_labels, name="dbscan_cluster", data_type="int")

    # assign clusters
    return storms.assign(cluster=(cluster_xarray))

def standardize(variables: np.array) -> np.array:
    """
        standardize input variables across axis=1. That is, standardized = (x - mean)/ standard deviation
    
        Parameters
        ----------
            variables (np.array): variables where each column corresponds to a timeseries
            
        Returns
        -------
            (np.array) standardized variables
    
    """
    mu = np.mean(variables.astype(float), axis=0)
    std = np.std(variables.astype(float), axis=0)
    return (variables - mu)/std

def normalize(variables: np.array) -> np.array:
    """
        normalize input variables across axis=1. That is, standardized = (x - min)/ (max - min)
    
        Parameters
        ----------
            variables (np.array): variables where each column corresponds to a timeseries
            
        Returns
        -------
            (np.array) normalized variables
    
    """
    mx = np.max(variables.astype(float), axis=0)
    mn = np.min(variables.astype(float), axis=0)
    return (variables - mn) / (mx - mn)

def get_storms_moments(storms:xarray.Dataset) -> np.array:
    """
        get track moments for the given storms
    
        Parameters
        ----------
            storms (xarray.Dataset): storms in xarray format with pre-calculated moments
    
        Returns
        -------
            (numpy.array): numpy array of moments
    
    """
    
    try:
        mask = not_nan_mask(storms=storms)
        return np.array([storms.lon_mean.values[mask],
                           storms.lat_mean.values[mask],
                           storms.lon_var.values[mask],
                           storms.lat_var.values[mask],
                           storms.lon_lat_cov.values[mask]]).T
    except:
        raise ValueError("need to calculate moments first by using the function get_storms_with_moments()")
    

def filter_storms(storms:xarray.Dataset, years:list=[], storm_ids: list=[], kmeans_clusters: list=[], dbscan_clusters:list=[]) -> xarray.Dataset:
    """
        select multiple storms by sid or by year
    
        Parameters
        ----------
            storms (xarray.Dataset): storms in xarray format
        
            storm_ids (list): list of storm's sid that are fetched
    
            years (list): list of years that are fetched
    
        Returns
        -------
            (xarray.Dataset) filtered storms
    
    """
    
    # select storm years
    if len(years) != 0:
        storms = storms.where(storms.season.isin(years), drop=True)
    
    # select storm ids
    if len(storm_ids) != 0:
        storms = storms.where(storms.sid.isin(storm_ids), drop=True)
    
    # select storm kmeans clusters
    if len(kmeans_clusters) != 0:
        storms = storms.where(storms.kmeans_cluster.isin(kmeans_clusters), drop=True)

    # select storm dbscan clusters
    if len(dbscan_clusters) != 0:
        storms = storms.where(storms.dbscan_cluster.isin(dbscan_clusters), drop=True)
    
    return storms

def get_storms_with_moments(storms:xarray.Dataset, weights:list=[]) -> xarray.Dataset:
    """
        calculate the track moments for the given storms
    
        Parameters
        ----------
            storms (xarray.Dataset): storms in xarray format
    
            weights (list): optional. weights applied to the mean and covariance estimation
    
        Returns
        -------
            (xarray.Dataset): assign X-centroid, Y-centroid, X_var, Y_var, and XY_var to each storm
    
    """ 
    
    moments_list = np.array([_get_storm_moments(storms.sel(storm=i), weights=weights) 
                            for i in range(storms.dims['storm'])
                            if _get_storm_moments(storms.sel(storm=i), weights=weights)]).T

    # get xarray of clusters
    lon_mean = moments_list[0]
    lat_mean = moments_list[1]
    lon_var = moments_list[2]
    lat_var = moments_list[3]
    lon_lat_cov = moments_list[4]
    lon_std = lon_var ** 1/2
    lat_std = lat_var ** 1/2
    lon_delta = moments_list[5]
    lat_delta = moments_list[6]
    avg_year = moments_list[7]
    avg_month = moments_list[8]
    avg_day = moments_list[9]
    lon_lat_corr = lon_lat_cov / (lon_std * lat_std)

    # moments
    lon_mean_xarray = _assign_cluster(storms=storms, cluster_labels=lon_mean, name="lon_mean", data_type="float")
    lat_mean_xarray = _assign_cluster(storms=storms, cluster_labels=lat_mean, name="lat_mean", data_type="float")
    lon_var_xarray = _assign_cluster(storms=storms, cluster_labels=lon_var, name="lat_var", data_type="float")
    lat_var_xarray = _assign_cluster(storms=storms, cluster_labels=lat_var, name="lat_var", data_type="float")
    lon_lat_cov_xarray = _assign_cluster(storms=storms, cluster_labels=lon_lat_cov, name="lon_lat_cov", data_type="float")
    lon_std_xarray = _assign_cluster(storms=storms, cluster_labels=lon_std, name="lon_std", data_type="float")
    lat_std_xarray = _assign_cluster(storms=storms, cluster_labels=lat_std, name="lat_std", data_type="float")
    lon_lat_corr_xarray = _assign_cluster(storms=storms, cluster_labels=lon_lat_corr, name="lon_lat_corr", data_type="float")

    # deltas
    lon_delta_xarray = _assign_cluster(storms=storms, cluster_labels=lon_delta, name="lon_delta", data_type="float")
    lat_delta_xarray = _assign_cluster(storms=storms, cluster_labels=lat_delta, name="lat_delta", data_type="float")

    # dates
    month_xarray = _assign_cluster(storms=storms, cluster_labels=avg_month, name="avg_month", data_type="int")
    year_xarray = _assign_cluster(storms=storms, cluster_labels=avg_year, name="avg_year", data_type="int")
    day_xarray = _assign_cluster(storms=storms, cluster_labels=avg_day, name="avg_day", data_type="int")
        
    # assign moments
    storms = storms.assign(lon_mean=(lon_mean_xarray))
    storms = storms.assign(lat_mean=(lat_mean_xarray))
    storms = storms.assign(lon_var=(lon_var_xarray))
    storms = storms.assign(lat_var=(lat_var_xarray))
    storms = storms.assign(lon_lat_cov=(lon_lat_cov_xarray))
    storms = storms.assign(lon_std=(lon_std_xarray))
    storms = storms.assign(lat_std=(lat_std_xarray))
    storms = storms.assign(lon_lat_corr=(lon_lat_corr_xarray))
    storms = storms.assign(lon_delta=(lon_delta_xarray))
    storms = storms.assign(lat_delta=(lat_delta_xarray))
    storms = storms.assign(avg_year=(month_xarray))
    storms = storms.assign(avg_month=(year_xarray))
    storms = storms.assign(avg_day=(day_xarray))

    return storms

def _get_storm_moments(storm:xarray.Dataset, weights:list=[]) -> tuple[list]:
    """
        calculate the track moments for the one storm
    
        Parameters
        ----------
            storm (xarray.Dataset): storm in xarray format
    
            weights (list): optional. weights applied to the mean and covariance estimation
    
        Returns
        -------
            (tuple[list]): tuple with 5 lists. Each least corresponds to X-centroid, Y-centroid, X_var, Y_var, XY_var respectively
    
    """

    # get longitude and latitude
    lon_lst, lat_lst = _get_lon_lat(storms=storm)

    # adjust weights
    weights = [1] * len(lon_lst) if len(weights) == 0 else weights
    
    # If the track only has one point, there is no point in calculating the moments
    if len(lon_lst)<= 1: return None

    # verify weights length
    if len(weights) != len(lon_lst):
        raise ValueError(f"length of weights <{len(weights)}> must be the same size as the length of the lon and lat of the storm <{len(lon_lst)}>")
    else:
        # apply weights
        lon_weighted, lat_weighted = lon_lst * weights, lat_lst * weights

    
    # M1 (first moment = mean). 
    lon_mean, lat_mean = np.mean(lon_weighted), np.mean(lat_weighted)
        
    # M2 (second moment = variance of lat and of lon / covariance of lat to lon
    cv = np.ma.cov([lon_weighted, lat_weighted])

    # lon and lat delta
    lon_delta = np.ptp(lon_lst)
    lat_delta = np.ptp(lat_lst)

    # median month and year
    timestamps = storm.iso_time.values  # Adjust the variable name if different
    
    # Convert byte strings to pandas datetime, handling missing values
    timestamps = pd.to_datetime([t.decode("utf-8") if isinstance(t, bytes) else t for t in timestamps], errors="coerce")
    
    # Drop NaN values
    timestamps = timestamps.dropna()
    
    # Extract month and day
    years = timestamps.year
    months = timestamps.month
    days = timestamps.day
    
    # Compute the average month and average day
    avg_year = round(np.nanmean(years))
    avg_month = round(np.nanmean(months))
    avg_day = round(np.nanmean(days))
        
    return [lon_mean, lat_mean, cv[0, 0], cv[1, 1], cv[0, 1], lon_delta, lat_delta, avg_year, avg_month, avg_day]

def _get_lon_lat(storms:xarray.Dataset) -> tuple[list]:
    """
        get longitude and latitude of storms
    
        Parameters
        ----------
            storms (xarray.Dataset): storms in xarray format
    
        Returns
        -------
            (tuple[list]): tuple with two lists. The first list corresponds to the longitude and the second list to the latitude
    
    """
    
    lon_lst = storms.lon.values
    lat_lst = storms.lat.values
    return lon_lst[~np.isnan(lon_lst)], lat_lst[~np.isnan(lat_lst)]

def _complete_cluster_list(storms:xarray.Dataset, cluster_labels:list) -> list:
    """
        create complete list of variables (including nans)
    
        Parameters
        ----------
            storms (xarray.Dataset): storm in xarray format
    
            cluster_labels (list): cluster labels
    
        Returns
        -------
            (list): list of clusters. If a storm has only one lon, then the cluster is None
    
    """
    

    # initialize clusters with size equal to the number of storms
    complete_clusters = np.array([None] * len(storms.sid))
    mask = not_nan_mask(storms=storms)

    # assign clusters based on mask
    complete_clusters[mask] = cluster_labels
    return complete_clusters

def not_nan_mask(storms:xarray.Dataset) -> list:
    """
        create mask of not-nan values
    
        Parameters
        ----------
            storms (xarray.Dataset): storms in xarray format
    
        Returns
        -------
            (list): mask where True indicates not nan values
    
    """
    mask = [True] * len(storms.sid)

    # create mask to find which storms where part of the clustering
    for i in range(storms.dims['storm']):
        lon_lst = storms.sel(storm=i).lon.values

        # not a valid storm, then set flag to False
        if len(lon_lst[~np.isnan(lon_lst)]) <= 1:
            mask[i] = False

    return mask

def _assign_cluster(storms:xarray.Dataset, cluster_labels:list, name: str, data_type:str) -> xarray.DataArray:
    """
        assign given cluster to each storm
    
        Parameters
        ----------
            storms (xarray): storms in xarray format

            cluster_labels (list): list of clusters where each row corresponds to each storm's cluster
    
        Returns
        -------
            (xarray.DataArray): xarray of clusters
    
    """

    complete_clusters = _complete_cluster_list(storms=storms, cluster_labels=cluster_labels)

    # define dataarray of clusters by taking as a blueprint the sid
    cluster_xarray = xarray.DataArray(
        data=complete_clusters,
        dims=storms.sid.dims,
        coords=storms.sid.coords,
        attrs=dict(
            description=name,
            units=data_type,
        ),
    )

    return cluster_xarray