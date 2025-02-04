# Plotting Functions

# Python Imports
import matplotlib.pyplot as plt # matplotlib
import cartopy.feature as cfeature # used for map projection
import cartopy.crs as ccrs # used for map projection
import numpy as np
from numpy import linalg as LA # to plot the moments (by calculating the eigenvalues)

# Project Imports
from func_tools.helper_functions import get_lon_lat, get_moments

def map_background(label=False, extent=[-100, 0, 0, 60]):
  # A helpder function for creating the map background.
  # INPUT:
  # "extent": corresponds to the location information of the showed map.
  # "label": boolean

  # OUTPUT:
  # Matplotlib AXES object

  plt.figure(figsize = (20, 10))
  ax = plt.axes(projection=ccrs.PlateCarree())
  ax.coastlines()
  ax.set_extent(extent)
  ax.gridlines(draw_labels=label) # show labels or not
  __LAND__ = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                      edgecolor='face',
                                      facecolor=cfeature.COLORS['land'],
                                          linewidth=.1)
  __OCEAN__ = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
                                       edgecolor='face',
                                       facecolor=cfeature.COLORS['water'], linewidth=.1)
  ax.add_feature(__LAND__, zorder=0)
  ax.add_feature(__OCEAN__)
  return ax

def plot_tracks(storms, color='blue', all_track=True, marker_color='black'):
    # INPUT:
    # ax: Matplotlib axes object
    # storm: a Xarray DATASET object (this can be adjusted if desired)
    # all_track: plot the whole track or just the origin
    
    # OUTPUT:
    # None

    print(f"There are {storms.dims['storm']} storms.\n")
    storm_num = storms.dims['storm']
    ax = map_background(extent=[-100, 0, 0, 60], label=True)
    
    for ind in range(storm_num):
      storm = storms.sel(storm=ind)
      plot_one_track(ax, storm, color='orange')
    # plt.legend()
    
def plot_one_track(ax, storm, color='blue', all_track=True, marker_color='black'):
    # INPUT:
    # ax: Matplotlib axes object
    # storm: a Xarray DATASET object (this can be adjusted if desired)
    # all_track: plot the whole track or just the origin
    
    # OUTPUT:
    # None
    
    lon_lst, lat_lst = get_lon_lat(storm)
    if all_track:
        ax.plot(lon_lst, lat_lst, '-o', color=color, linewidth=2, markersize=3) # marker='.'
        ax.plot(lon_lst[-1], lat_lst[-1], color=marker_color, marker='x', markersize=10)
    ax.plot(lon_lst[0], lat_lst[0], color=marker_color, marker='*', markersize=10)
    ax.text(lon_lst[0], lat_lst[0]-2, str(storm.name.values)[2:-1], horizontalalignment='center')

def plot_track_moments(storm, lon_weighted, lat_weighted, lon_var, lat_var, xy_var):

    # Rotate the circle and calculate points on the circle
    # Set N larger to make the oval more precise and to consume more electricity
    N=1000
    t = np.linspace(0, 2 * np.pi, N)
    circle = [np.sqrt(lon_var) * np.cos(t), np.sqrt(lat_var) * np.sin(t)]
    _, R_rot = LA.eig(np.array([[lon_var, xy_var], [xy_var, lat_var]]))
    circle = np.dot(R_rot, circle)
    R_1, R_2 = circle[0, :] + lon_weighted, circle[1, :] + lat_weighted
    
    # Plot
    ax = map_background(extent=[-100, 20, 0, 80], label=True)
    plot_one_track(ax, storm)
    ax.plot(R_1, R_2, '-', color='black', linewidth=1)
    ax.plot([circle[0,0], circle[0,int(N/2)]]+lon_weighted,
            [circle[1,0], circle[1,int(N/2)]]+lat_weighted, '-gh')
    ax.plot([circle[0,int(N/4)], circle[0,int(N*3/4)]]+lon_weighted,
            [circle[1,int(N/4)], circle[1,int(N*3/4)]]+lat_weighted, '-gh')

def plot_centroids(moment_list:list, labels:list, title_name:str):
    colors = ['green', 'blue', 'yellow', 'green', 'red', 'magenta','orange','gray','white','brown','purple','cyan']
    ax = map_background()
    for k in range(len(moment_list)):
        ax.plot(moment_list[k][0], moment_list[k][1], c=colors[labels[k]], marker='*')

    plt.title(title_name)