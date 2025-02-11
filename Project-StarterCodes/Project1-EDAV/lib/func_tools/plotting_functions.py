# Plotting Functions

# Python Imports
import matplotlib.pyplot as plt  # matplotlib
import cartopy.feature as cfeature  # used for map projection
import cartopy.crs as ccrs  # used for map projection
import numpy as np
import pandas as pd
from numpy import linalg as LA  # to plot the moments (by calculating the eigenvalues)
import xarray
import plotly.figure_factory as ff
import plotly.express as px


def plot_clusters_box_whiskers(variables: pd.DataFrame, y_name: str) -> None:
    """
    Create histogram with density curve.

    Parameters
    ----------

    variables (pandas.DataFrame): pandas dataframe with columns "cluster" and y_name

    y_name (str): name of the plotted variable in the y axis

    Returns
    -------

    (None)

    """

    fig = px.box(variables, x="cluster", y=y_name, color="cluster", points="all")
    fig.show()


def plot_histogram(variables: np.array, group_labels: list, nbins: int = 1000) -> None:
    """
    Create histogram with density curve.

    Parameters
    ----------

    variables (numpy.array): variables to plot

    nbins (int): number of bins.

    Returns
    -------

    (None)

    """

    fig = ff.create_distplot(
        variables.astype(float), group_labels, bin_size=[nbins] * len(group_labels)
    )
    fig.show()


def map_background(label: bool = False, extent: list = [-100, 0, 0, 60]):
    """
    A helper function for creating the map background.

    Parameters
    ----------

    extent (list): corresponds to the location information of the showed map. this controls where the projection appears, i.e., which part of earth map should appear on the projection. Its four entries can be viewd as (starting_longitude, ending_longitude, starting_latitude, ending_latitude).

    label (boolean): to short label or not on the output map background

    Returns
    -------

    () matplotlib axes object

    """

    plt.figure(figsize=(20, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_extent(extent)
    ax.gridlines(draw_labels=label)  # show labels or not
    __LAND__ = cfeature.NaturalEarthFeature(
        "physical",
        "land",
        "10m",
        edgecolor="face",
        facecolor=cfeature.COLORS["land"],
        linewidth=0.1,
    )
    __OCEAN__ = cfeature.NaturalEarthFeature(
        "physical",
        "ocean",
        "10m",
        edgecolor="face",
        facecolor=cfeature.COLORS["water"],
        linewidth=0.1,
    )
    ax.add_feature(__LAND__, zorder=0)
    ax.add_feature(__OCEAN__)
    return ax


def plot_kmeans_inertia(sum_of_squares: list, max_clusters: int):
    plt.plot(max_clusters, sum_of_squares, "bx-")
    plt.xlabel("k")
    plt.ylabel("Sum_of_squared_distances")
    plt.title("Inertia Values against the Number of Cluster k")


def plot_tracks(
    storms: xarray.DataArray,
    extent: list = [-100, 0, 0, 60],
    color: str = "orange",
    all_track: bool = True,
    marker_color: str = "black",
    label: bool = True,
):
    # INPUT:
    # ax: Matplotlib axes object
    # storm: a Xarray DATASET object (this can be adjusted if desired)
    # all_track: plot the whole track or just the origin

    # OUTPUT:
    # None

    __STORM__ = "storm"
    storm_num = storms.dims[__STORM__] if __STORM__ in storms.dims else 1

    print(f"There are {storm_num} storms.\n")
    ax = map_background(extent=extent, label=label)

    for ind in range(storm_num):
        storm = storms.sel(storm=ind) if storm_num > 1 else storms
        ax = plot_one_track(
            ax=ax,
            storm=storm,
            color=color,
            all_track=all_track,
            marker_color=marker_color,
        )

    return ax


def plot_one_track(
    ax,
    storm: xarray.DataArray,
    color: str = "orange",
    all_track: bool = True,
    marker_color="black",
):
    # INPUT:
    # ax: Matplotlib axes object
    # storm: a Xarray DATASET object (this can be adjusted if desired)
    # all_track: plot the whole track or just the origin

    # OUTPUT:
    # None

    lon_lst = storm.lon.values
    lat_lst = storm.lat.values

    if all_track:
        ax.plot(
            lon_lst, lat_lst, "-o", color=color, linewidth=2, markersize=3
        )  # marker='.'
        ax.plot(lon_lst[-1], lat_lst[-1], color=marker_color, marker="x", markersize=10)
    ax.plot(lon_lst[0], lat_lst[0], color=marker_color, marker="*", markersize=10)
    ax.text(
        lon_lst[0],
        lat_lst[0] - 2,
        str(storm.name.values)[2:-1],
        horizontalalignment="center",
    )

    return ax


def plot_track_moments(
    storms: xarray.DataArray, extent: list = [-100, 0, 0, 60], label: bool = True
):
    """
    Plot track moments as an ellipse

    Parameters
    ----------

    storms (xarray): storms in xarray format

    weights (list): to short label or not on the output map background

    Returns
    -------

    () matplotlib axes object

    """

    __STORM__ = "storm"
    storm_num = storms.dims[__STORM__] if __STORM__ in storms.dims else 1

    for ind in range(storm_num):
        # filter storm
        storm = storms.sel(storm=ind) if storm_num > 1 else storms

        # plot tracks
        ax = plot_tracks(storms=storm, extent=extent, label=label)

        # plot ellipse
        N, R_1, R_2, circle = _ellipse_params(
            lon_mean=storm.lon_mean.values**1,
            lat_mean=storm.lat_mean.values**1,
            lon_var=storm.lon_var.values**1,
            lat_var=storm.lat_var.values**1,
            lon_lat_cov=storm.lon_lat_cov.values**1,
        )

        ax.plot(R_1, R_2, "-", color="black", linewidth=1)
        ax.plot(
            [circle[0, 0], circle[0, int(N / 2)]] + storm.lon_mean.values,
            [circle[1, 0], circle[1, int(N / 2)]] + storm.lat_mean.values,
            "-gh",
        )
        ax.plot(
            [circle[0, int(N / 4)], circle[0, int(N * 3 / 4)]] + storm.lon_mean.values,
            [circle[1, int(N / 4)], circle[1, int(N * 3 / 4)]] + storm.lat_mean.values,
            "-gh",
        )

    return ax


def _ellipse_params(
    lon_mean: list, lat_mean: list, lon_var: list, lat_var: list, lon_lat_cov: list
):
    # Rotate the circle and calculate points on the circle
    # Set N larger to make the oval more precise and to consume more electricity
    N = 1000
    t = np.linspace(0, 2 * np.pi, N)

    lon_std = lon_var ** (1 / 2)
    lat_std = lat_var ** (1 / 2)

    circle = [lon_std * np.cos(t), lat_std * np.sin(t)]
    _, R_rot = LA.eig(np.array([[lon_var, lon_lat_cov], [lon_lat_cov, lat_var]]))
    circle = np.dot(R_rot, circle)
    R_1, R_2 = circle[0, :] + lon_mean, circle[1, :] + lat_mean

    return N, R_1, R_2, circle


def plot_centroids(storms: xarray.DataArray, title_name: str, cmap_name: str = "Set1"):
    """
    Plot track centroids

    Parameters
    ----------

    storms (xarray.Dataset): storms in xarray format with pre-calculated moments

    title_name (str): title of the plot

    cmap_name (str): matplotlib cmap name

    Returns
    -------

    (None)

    """

    colors = plt.cm.get_cmap(cmap_name)
    clusters = storms.cluster.values
    ax = map_background()
    for i in range(len(storms.lon_mean.values)):
        cluster = clusters[i]
        if cluster is not None:
            ax.plot(
                storms.lon_mean.values[i],
                storms.lat_mean.values[i],
                c=colors(cluster),
                marker="*",
            )

    plt.title(title_name)
