
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
import matplotlib.cm as cm
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

import folium
from keplergl import KeplerGl
import matplotlib.colors as mcolors
from folium.plugins import MarkerCluster


#================================================================================================================================

def plot_gdf_in_folium(gdf, gps_col_name, date_col_name):
    """ 
    Plot a geodataframe in folium.

    Parameters
    ----------
    gdf: GeoDataFrame
        Dataframe containing 'latitude' and 'longitude' column

    Returns
    -------
    folmap: folium.Map
        Folium map with points plotted.
    """
    # Initialize a Folium map
    center_lat, center_lon = gdf['latitude'].mean(), gdf['longitude'].mean()  # Map centered on average lat/lon
    folmap = folium.Map(location=[center_lat, center_lon], zoom_start=15)
    
    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(folmap)
    
    # Add all points from the DataFrame to the map
    for _, row in gdf.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"ID: {row[gps_col_name]}<br>DateTime: {row[date_col_name]}",
            tooltip="Click for details",
        ).add_to(marker_cluster)

    return folmap

#================================================================================================================================

def visualize_trips_in_folium(trips_gdf, trip_id_col="trip_id"):
    """
    Visualize multiple trips on a Folium map.

    Parameters
    ----------
    trips_gdf : GeoDataFrame
        The GeoDataFrame containing map-matched trips with LineStrings.
    trip_id_col : str
        Column name for trip IDs in the GeoDataFrame.

    Returns
    -------
    folium.Map
        An interactive map displaying the trajectories for all trips.
    """
    # Initialize the map centered on the average coordinates
    avg_lat = trips_gdf.geometry.centroid.y.mean()
    avg_lon = trips_gdf.geometry.centroid.x.mean()
    folium_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)

    # Assign a unique color to each trip
    unique_trips = trips_gdf[trip_id_col].unique()
    colormap = cm.get_cmap("tab20", len(unique_trips))
    trip_colors = {trip_id: colors.to_hex(colormap(i)) for i, trip_id in enumerate(unique_trips)}

    # Add each trip to the map
    for _, row in trips_gdf.iterrows():
        trip_id = row[trip_id_col]
        color = trip_colors[trip_id]
        line = row.geometry

        if isinstance(line, LineString):  # Ensure geometry is a LineString
            coords = [(lat, lon) for lon, lat in line.coords]
            folium.PolyLine(
                coords,
                color=color,
                weight=5,
                opacity=0.8,
                tooltip=f"Trip ID: {trip_id}"
            ).add_to(folium_map)
        else:
            print(f"Skipping trip {trip_id}: Geometry is not a LineString.")

    return folium_map

#================================================================================================================================

def visualize_trips_in_kepler(trips_gdf, trip_id_col="trip_id"):
    """
    Visualize map-matched trips in Kepler.gl, with each trip in a different color.

    Parameters
    ----------
    trips_gdf : GeoDataFrame
        The GeoDataFrame containing map-matched trips with LineStrings.
    trip_id_col : str
        Column name for trip IDs in the GeoDataFrame.

    Returns
    -------
    KeplerGl
        Kepler.gl map displaying the trips.
    """
    # Ensure all columns are JSON-serializable
    trips_gdf = trips_gdf.copy()
    if "day" in trips_gdf.columns:
        trips_gdf["day"] = trips_gdf["day"].astype(str)  # Convert datetime.date to string

    # Assign a color to each trip using a colormap
    unique_trips = trips_gdf[trip_id_col].unique()
    colormap = cm.get_cmap("tab20", len(unique_trips))  # Use a colormap with enough distinct colors
    trip_colors = {trip_id: mcolors.to_hex(colormap(i)) for i, trip_id in enumerate(unique_trips)}

    # Add a color column to the GeoDataFrame
    trips_gdf["color"] = trips_gdf[trip_id_col].map(trip_colors)

    # Create a KeplerGL map
    kepler_map = KeplerGl(height=600)

    # Add the GeoDataFrame to the map
    kepler_map.add_data(data=trips_gdf, name="Map-Matched Trips")

    return kepler_map

#================================================================================================================================

def plot_trajectory_on_folium(gps_df, trajs_df, registrationID, trip_no):
    """
    Plots GPS points from gps_df and overlays them with trajectory lines from trajs_df on a Folium map for a particular ID and trip.
    Saves the map as an HTML file named '<registrationID>_<trip_no>_map.html'.

    Parameters:
    -----------
    gps_df : GeoDataFrame
        The dataframe containing GPS point data.
    trajs_df : GeoDataFrame
        The dataframe containing trajectory LineStrings.
    registrationID : str
        The registration ID to filter data.
    trip_no : int
        The trip number to filter data.

    Returns:
    --------
    str
        Filename of the saved HTML map.
    """
    
    # Filter the first dataframe for matching registrationID and trip_no
    gps_filtered = gps_df[
        (gps_df["registrationID"] == registrationID) &
        (gps_df["trip_id"] == trip_no)
    ]

    # Filter the second dataframe for matching registrationID and trip_no
    traj_filtered = trajs_df[
        (trajs_df["gps_id"] == registrationID) &
        (trajs_df["trip_id"] == trip_no)
    ]

    # Get the initial center for the map (first GPS point if available)
    if not gps_filtered.empty:
        first_point = gps_filtered.iloc[0].geometry
        map_center = [first_point.latitude, first_point.longitude]
    elif not traj_filtered.empty:
        first_line = traj_filtered.iloc[0].geometry
        map_center = [first_line.coords[0][1], first_line.coords[0][0]]
    else:
        print("No data found for the given registrationID and trip_no.")
        return None

    # Create Folium map
    folium_map = folium.Map(location=map_center, zoom_start=13)

    # Plot GPS points (red markers)
    for _, row in gps_filtered.iterrows():
        folium.Marker(
            location=[row.geometry.latitude, row.geometry.longitude],
            popup=f"Time: {row['DateTime']}",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(folium_map)

    # Plot trajectory LineStrings (blue polylines)
    for _, row in traj_filtered.iterrows():
        if row.geometry and not row.geometry.is_empty:
            coords = [(lat, lon) for lon, lat in row.geometry.coords]
            folium.PolyLine(
                locations=coords,
                color="blue",
                weight=3,
                opacity=0.7,
                popup=f"Trip ID: {row['trip_id']}"
            ).add_to(folium_map)

    # Save map to HTML file
    map_filename = f"{registrationID}_{trip_no}_map.html"
    folium_map.save(map_filename)
    print(f"Map saved as {map_filename}")

    return map_filename
