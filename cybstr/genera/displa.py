import os
import json
import time
import requests
import datetime
import warnings
import osmnx as ox
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from copy import copy
from pyproj import CRS
import contextily as ctx
import movingpandas as mpd
import matplotlib.pyplot as plt
from urllib.request import urlopen
from datetime import datetime, timedelta
from shapely.geometry import Point, LineString

#================================================================================================================================

def display_shp_in_contextily_v1(path, crs, dim):
    """ 
    Displays a shapefile in contextily.

    Parameters
    ----------
    path: string
        The path to the shapefile
    crs: int
        The CRS projection of the map
    dim: int
        Map dimensions (square)

    Returns
    -------
    gdf: gpd.GeoDataFrame
        Shapefile converted to geodataframe.
    """
    # Read shapefile
    shp = gpd.read_file(path)
    
    # Set the Coordinate Reference System (CRS) if not already set (WGS84 - EPSG:4326 for lat/long)
    shp.to_crs(epsg=crs, inplace=True)
    
    # Plot the geometries on a basemap
    fig, ax = plt.subplots(figsize=(dim, dim))
    
    # Plot the roads (your data)
    shp.plot(ax=ax, color='blue', linewidth=1, alpha=0.3)
    
    # Use OpenStreetMap for the basemap
    ctx.add_basemap(ax, crs=shp.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
    
    # Show the plot
    plt.show()

    return 0

#================================================================================================================================

def display_loc_in_contextily_v1(gdf, title):
    """ 
    Displays locations from a geodataframe in contextily.

    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        Gdf containing the latitude and longitude.
    title: string
        The title of the map

    Returns
    -------
    0
    """
    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(gdf['longitude'], gdf['latitude'])]
    gdf = gpd.GeoDataFrame(gdf, geometry=geometry, crs='EPSG:4326')

    # Reproject to Web Mercator (EPSG:3857) for plotting with contextily
    gdf = gdf.to_crs(epsg=3857)

    # Plot the points
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color='blue', markersize=5, label=title)

    # Add a basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Customize the plot
    ax.set_title(title)
    plt.show()

    return 0

#================================================================================================================================