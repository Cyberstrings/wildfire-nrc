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
from shapely.wkt import loads
from shapely.ops import nearest_points
from shapely.geometry import Point, LineString

from scipy.stats import norm
from sklearn.neighbors import KNeighborsRegressor
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import LabelEncoder


#================================================================================================================================

def interpolate_along_trajectory_v1(matched_trips_gdf, trip_id, interp_dist=100):
    """
    Interpolates points along a trajectory for a given trip and fixed distance intervals.
    
    Parameters:
    - matched_trips_gdf (GeoDataFrame): Contains trajectory geometries (LINESTRING).
    - trip_id (int or str): Trip ID to interpolate.
    - interp_dist (float): Distance between interpolated points in meters.

    Returns:
    - GeoDataFrame: Interpolated points with columns ['latitude', 'longitude', 'geometry'].
    """
    # Extract the trajectory for the given trip
    trip = matched_trips_gdf[matched_trips_gdf["trip_id"] == trip_id]

    if trip.empty:
        raise ValueError(f"Trip ID {trip_id} not found in matched_trips_gdf.")

    trajectory = trip.iloc[0].geometry  # Extract LINESTRING

    if not isinstance(trajectory, LineString):
        raise ValueError("Geometry must be a LINESTRING.")

    # Convert CRS to a projected coordinate system for accurate distance-based interpolation
    projected_gdf = matched_trips_gdf.to_crs(epsg=3395)  # World Mercator (meters)
    projected_traj = projected_gdf[projected_gdf["trip_id"] == trip_id].iloc[0].geometry

    # Compute total length and interpolation distances
    total_length = projected_traj.length
    distances = np.arange(0, total_length, interp_dist).tolist() + [total_length]

    # Generate interpolated points in projected CRS
    interpolated_points_proj = [projected_traj.interpolate(d) for d in distances]

    # Convert back to original CRS (WGS84)
    interpolated_points = gpd.GeoSeries(interpolated_points_proj, crs=3395).to_crs(epsg=4326)

    # Create DataFrame with latitude, longitude, and geometry
    interpolated_gdf = gpd.GeoDataFrame(
        {
            "latitude": interpolated_points.y,
            "longitude": interpolated_points.x,
            "geometry": interpolated_points,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    return interpolated_gdf

#================================================================================================================================

def interpolate_along_trajectory_v2(matched_trips_gdf, roads_gdf, trip_id, interp_dist=100):
    """
    Interpolates points along a trajectory for a given trip and fixed distance intervals.
    New: Adds dist_from_traj_start_m, road_name columns in the returned dataframe.
    
    Parameters:
    - matched_trips_gdf (GeoDataFrame): Contains trajectory geometries (LINESTRING).
    - roads_gdf (GeoDataFrame): Must contain 'STR_ID', 'FULLNAME', 'geometry' of roads.
    - trip_id (int or str): Trip ID to interpolate.
    - interp_dist (float): Distance between interpolated points in meters.

    Returns:
    - GeoDataFrame: Interpolated points with columns ['latitude', 'longitude', 'geometry'].
    """
    # Extract the trajectory for the given trip
    trip = matched_trips_gdf[matched_trips_gdf["trip_id"] == trip_id]

    if trip.empty:
        raise ValueError(f"Trip ID {trip_id} not found in matched_trips_gdf.")

    trajectory = trip.iloc[0].geometry  # Extract LINESTRING

    if not isinstance(trajectory, LineString):
        raise ValueError("Geometry must be a LINESTRING.")

    # Convert CRS to a projected coordinate system for accurate distance-based interpolation
    projected_gdf = matched_trips_gdf.to_crs(epsg=3395)  # World Mercator (meters)
    projected_traj = projected_gdf[projected_gdf["trip_id"] == trip_id].iloc[0].geometry

    # Compute total length and interpolation distances
    total_length = projected_traj.length
    distances = np.arange(0, total_length, interp_dist).tolist() + [total_length]  # Include last point

    # Generate interpolated points in projected CRS
    interpolated_points_proj = [projected_traj.interpolate(d) for d in distances]

    # Convert back to original CRS (WGS84)
    interpolated_points = gpd.GeoSeries(interpolated_points_proj, crs=3395).to_crs(epsg=4326)

    # Create DataFrame with latitude, longitude, geometry, and distance from start
    interpolated_gdf = gpd.GeoDataFrame(
        {
            "trip_id": trip_id,
            "latitude": interpolated_points.y,
            "longitude": interpolated_points.x,
            "geometry": interpolated_points,
            "dist_from_traj_start_m": distances,  # Store distance from start
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    
    # Add road details
    interpolated_gdf = interpolated_gdf.to_crs(roads_gdf.crs)
    interpolated_gdf = gpd.sjoin_nearest(interpolated_gdf, roads_gdf[['STR_ID', 'FULLNAME', 'geometry']], how='left', distance_col="dist")
    interpolated_gdf.rename(columns={'FULLNAME': 'road_name'}, inplace=True)
    
    # Convert back to EPSG:4326
    interpolated_gdf.set_crs('EPSG:26911', inplace=True)
    interpolated_gdf = interpolated_gdf.to_crs(epsg=4326)

    return interpolated_gdf
    
#================================================================================================================================

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString

def interpolate_along_trajectory_v3(matched_trips_gdf, roads_gdf, interp_dist=100):
    """
    Interpolates points along trajectories for multiple trips at fixed distance intervals.
    Adds dist_from_traj_start_m and road_name columns in the returned dataframe. 
    Can handle dataframe with multiple trip_ids; no need to specify only one.

    Parameters:
    - matched_trips_gdf (GeoDataFrame): Contains trajectory geometries (LINESTRING) with trip_id.
    - roads_gdf (GeoDataFrame): Must contain 'STR_ID', 'FULLNAME', and 'geometry' of roads.
    - interp_dist (float): Distance between interpolated points in meters.

    Returns:
    - GeoDataFrame: Interpolated points with ['trip_id', 'latitude', 'longitude', 'geometry', 'dist_from_traj_start_m', 'STR_ID', 'road_name'].
    """
    
    interpolated_list = []
    
    for trip_id, trip in matched_trips_gdf.groupby("trip_id"):
        trajectory = trip.iloc[0].geometry  # Extract LINESTRING

        if not isinstance(trajectory, LineString):
            continue  # Skip invalid geometries

        # Convert CRS to a projected system for accurate distance-based interpolation
        projected_gdf = trip.to_crs(epsg=3395)  # World Mercator (meters)
        projected_traj = projected_gdf.iloc[0].geometry

        # Compute total length and interpolation distances
        total_length = projected_traj.length
        distances = np.arange(0, total_length, interp_dist).tolist() + [total_length]  # Include last point

        # Generate interpolated points in projected CRS
        interpolated_points_proj = [projected_traj.interpolate(d) for d in distances]

        # Convert back to original CRS (WGS84)
        interpolated_points = gpd.GeoSeries(interpolated_points_proj, crs=3395).to_crs(epsg=4326)

        # Create DataFrame with trip_id, latitude, longitude, and distance from start
        interpolated_gdf = gpd.GeoDataFrame(
            {
                "trip_id": trip_id,
                "latitude": interpolated_points.y,
                "longitude": interpolated_points.x,
                "geometry": interpolated_points,
                "dist_from_traj_start_m": distances,
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        interpolated_list.append(interpolated_gdf)

    # Combine all trip data
    all_interpolated = pd.concat(interpolated_list, ignore_index=True)

    # Add road details using nearest spatial join
    all_interpolated = all_interpolated.to_crs(roads_gdf.crs)
    all_interpolated = gpd.sjoin_nearest(
        all_interpolated, roads_gdf[['STR_ID', 'FULLNAME', 'geometry']], how='left', distance_col="dist"
    )
    all_interpolated.rename(columns={'FULLNAME': 'road_name'}, inplace=True)

    # Convert back to EPSG:4326
    all_interpolated.set_crs('EPSG:26911', inplace=True)
    all_interpolated = all_interpolated.to_crs(epsg=4326)

    return all_interpolated


#================================================================================================================================

def estimate_speeds_with_bayesian_fewshot_v1(final_df):
    """
    Uses Bayesian hierarchical model and few shot learning to estimate missing speed values in final_df
    
    Parameters:
    - final_df (DataFrame): Contains -
                            dist_from_traj_start_m: Distance of GPS point from start of trajectory
                            STR_ID: Street ID to uniquely identify a street
                            time_from_traj_start_sec: Time it took to reach the GPS point from start of trajectory. Some values missing.
                            segment_speed_kmh: Speed during travelling to that point from earlier point. Some values missing.
                            STREETTYPE: Type of street. E.g. highways are type 20, roads are 10, smaller roads are 30 etc.

    Returns:
    - df: Dataframe with filled speed values, and calculated time values from the speed values.
    """
    df = final_df.copy()
    
    # Step 1: Bayesian Prior Speed Estimation Based on STREETTYPE
    streettype_speed_map = {
        20: (80, 15),  # Highways: Mean  80 km/h, Std Dev 15
        10: (40, 10),  # Local roads: Mean 40 km/h, Std Dev 10
        30: (20, 5),   # Smaller streets: Mean 20 km/h, Std Dev 5
        40: (5, 2)     # Non-vehicular roads: Mean 5 km/h, Std Dev 2
    }
    
    for i in range(len(df)):
        if pd.isna(df.loc[i, 'segment_speed_kmh']):
            streettype = df.loc[i, 'STREETTYPE']
            mean_speed, std_dev = streettype_speed_map.get(streettype, (30, 10))
            
            # Bayesian Updating: Adjust Mean Based on Observed STR_ID Speeds
            observed_speeds = df[(df['STREETTYPE'] == df.loc[i, 'STREETTYPE']) & ~df['segment_speed_kmh'].isna()]['segment_speed_kmh']
            if len(observed_speeds) > 0:
                mean_speed = (np.mean(observed_speeds) + mean_speed) / 2  # Bayesian update
            
            df.loc[i, 'segment_speed_kmh'] = max(0, norm.rvs(loc=mean_speed, scale=std_dev))  # Ensure non-negative speeds
    
    # Step 2: Few-Shot Learning for Road-Specific Fine-Tuning
    knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
    known_speeds = df.dropna(subset=['segment_speed_kmh'])[['STR_ID', 'segment_speed_kmh']]
    
    if len(known_speeds) > 3:
        knn.fit(known_speeds[['STR_ID']], known_speeds['segment_speed_kmh'])
        
        for i in range(len(df)):
            if pd.isna(df.loc[i, 'segment_speed_kmh']):
                df.loc[i, 'segment_speed_kmh'] = max(0, knn.predict([[df.loc[i, 'STR_ID']]])[0])
    
    # Step 3: Compute Time from Speed (Ensuring Monotonicity)
    for i in range(1, len(df)):
        if pd.isna(df.loc[i, 'time_from_traj_start_sec']):
            prev_time = df.loc[i-1, 'time_from_traj_start_sec']
            dist_diff = df.loc[i, 'dist_from_traj_start_m'] - df.loc[i-1, 'dist_from_traj_start_m']
            speed = df.loc[i, 'segment_speed_kmh']
            
            if speed > 0:
                estimated_time = prev_time + (dist_diff / (speed / 3.6))  # Convert km/h to m/s
                df.loc[i, 'time_from_traj_start_sec'] = max(prev_time + 1, estimated_time)  # Ensure monotonicity

    # Step 4: Apply Gaussian Smoothing to Speed Values
    df['segment_speed_kmh'] = gaussian_filter1d(df['segment_speed_kmh'], sigma=1)

    return df


#================================================================================================================================

def estimate_speeds_with_bayesian_fewshot_v2(final_df):
    """
    Uses Bayesian hierarchical model and few shot learning to estimate missing speed values in final_df
    Change: Does not change existing values of speed. Might result in erratic graph.
    
    Parameters:
    - final_df (DataFrame): Contains -
                            dist_from_traj_start_m: Distance of GPS point from start of trajectory
                            STR_ID: Street ID to uniquely identify a street
                            time_from_traj_start_sec: Time it took to reach the GPS point from start of trajectory. Some values missing.
                            segment_speed_kmh: Speed during travelling to that point from earlier point. Some values missing.
                            STREETTYPE: Type of street. E.g. highways are type 20, roads are 10, smaller roads are 30 etc.

    Returns:
    - df: Dataframe with filled speed values, and calculated time values from the speed values.
    """
    df = final_df.copy()
    
    # Step 1: Bayesian Prior Speed Estimation Based on STREETTYPE
    streettype_speed_map = {
        20: (90, 15),  # Highways: Mean 80 km/h, Std Dev 15
        10: (40, 10),  # Local roads: Mean 40 km/h, Std Dev 10
        30: (20, 5),   # Smaller streets: Mean 20 km/h, Std Dev 5
        40: (5, 2)     # Non-vehicular roads: Mean 5 km/h, Std Dev 2
    }
    
    for i in range(len(df)):
        if pd.isna(df.loc[i, 'segment_speed_kmh']):
            streettype = df.loc[i, 'STREETTYPE']
            mean_speed, std_dev = streettype_speed_map.get(streettype, (30, 10))
            
            # Bayesian Updating: Adjust Mean Based on Observed STR_ID Speeds
            observed_speeds = df[(df['STR_ID'] == df.loc[i, 'STR_ID']) & ~df['segment_speed_kmh'].isna()]['segment_speed_kmh']
            if len(observed_speeds) > 0:
                mean_speed = (np.mean(observed_speeds) + mean_speed) / 2  # Bayesian update
            
            df.loc[i, 'segment_speed_kmh'] = max(0, norm.rvs(loc=mean_speed, scale=std_dev))  # Ensure non-negative speeds
    
    # Step 2: Weighted Speed Interpolation (Give More Importance to Known Speeds)
    known_speeds = df.dropna(subset=['segment_speed_kmh'])[['dist_from_traj_start_m', 'segment_speed_kmh']]
    
    if len(known_speeds) > 3:
        knn = KNeighborsRegressor(n_neighbors=5, weights='distance')  # Use more neighbors for smoother transition
        knn.fit(known_speeds[['dist_from_traj_start_m']], known_speeds['segment_speed_kmh'])
        
        for i in range(len(df)):
            if pd.isna(df.loc[i, 'segment_speed_kmh']):
                df.loc[i, 'segment_speed_kmh'] = max(0, knn.predict([[df.loc[i, 'dist_from_traj_start_m']]])[0])
    
    # Step 3: Compute Time from Speed (Ensuring Monotonicity and Preserving Existing Times)
    for i in range(1, len(df)):
        if pd.isna(df.loc[i, 'time_from_traj_start_sec']):
            prev_time = df.loc[i-1, 'time_from_traj_start_sec']
            dist_diff = df.loc[i, 'dist_from_traj_start_m'] - df.loc[i-1, 'dist_from_traj_start_m']
            speed = df.loc[i, 'segment_speed_kmh']
            
            if speed > 0:
                estimated_time = prev_time + (dist_diff / (speed / 3.6))  # Convert km/h to m/s
                df.loc[i, 'time_from_traj_start_sec'] = max(prev_time + 1, estimated_time)  # Ensure monotonicity
    
    # Step 4: Ensure Strict Monotonicity by Correcting Any Decreasing Time Values
    for i in range(1, len(df)):
        if df.loc[i, 'time_from_traj_start_sec'] < df.loc[i-1, 'time_from_traj_start_sec']:
            df.loc[i, 'time_from_traj_start_sec'] = df.loc[i-1, 'time_from_traj_start_sec'] + 1
    
    # Step 5: Apply Gaussian Smoothing to Speed Values (Preserve Known Values)
    original_speeds = df['segment_speed_kmh'].copy()
    smoothed_speeds = gaussian_filter1d(df['segment_speed_kmh'], sigma=1)
    
    # Preserve known values while applying smoothing
    df['segment_speed_kmh'] = np.where(df['segment_speed_kmh'].isna(), smoothed_speeds, original_speeds)
    
    return df


#================================================================================================================================

def estimate_speeds_with_bayesian_fewshot_v3(final_df):
    """
    Uses Bayesian hierarchical model and few-shot learning to estimate missing speed values 
    separately for each trip in final_df and join back together; same as v1 but for multiple trips.
    Also instead of raw STR_ID, it is encoded.

    Parameters:
    - final_df (DataFrame): Contains -
                            trip_id: Unique ID for each trip
                            dist_from_traj_start_m: Distance of GPS point from start of trajectory
                            STR_ID: Street ID to uniquely identify a street
                            time_from_traj_start_sec: Time it took to reach the GPS point from start of trajectory. Some values missing.
                            segment_speed_kmh: Speed during travelling to that point from earlier point. Some values missing.
                            STREETTYPE: Type of street. E.g. highways are type 20, roads are 10, smaller roads are 30 etc.

    Returns:
    - combined_df: Processed DataFrame with estimated speed values and computed time values.
    """
    df = final_df.copy()
    
    # Encode STR_ID to numeric values
    df['STR_ID'] = df['STR_ID'].astype(str)  
    df['STR_ID_encoded'] = LabelEncoder().fit_transform(df['STR_ID'])

    # Bayesian Prior Speed Estimation Based on STREETTYPE
    streettype_speed_map = {
        20: (80, 15),  # Highways: Mean 80 km/h, Std Dev 15
        10: (40, 10),  # Local roads: Mean 40 km/h, Std Dev 10
        30: (20, 5),   # Smaller streets: Mean 20 km/h, Std Dev 5
        40: (5, 2)     # Non-vehicular roads: Mean 5 km/h, Std Dev 2
    }

    processed_trips = []  # Store processed trip DataFrames

    for trip_id, trip_df in df.groupby("trip_id"):
        trip_df = trip_df.copy().reset_index(drop=True)  # Work on a copy

        # Step 1: Bayesian Prior Speed Estimation
        for i in range(len(trip_df)):
            if pd.isna(trip_df.loc[i, 'segment_speed_kmh']):
                streettype = trip_df.loc[i, 'STREETTYPE']
                mean_speed, std_dev = streettype_speed_map.get(streettype, (30, 10))

                # Bayesian Updating: Adjust Mean Based on Observed STR_ID Speeds
                observed_speeds = trip_df[
                    (trip_df['STREETTYPE'] == streettype) & ~trip_df['segment_speed_kmh'].isna()
                ]['segment_speed_kmh']

                if len(observed_speeds) > 0:
                    mean_speed = (np.mean(observed_speeds) + mean_speed) / 2  # Bayesian update
                
                trip_df.loc[i, 'segment_speed_kmh'] = max(0, norm.rvs(loc=mean_speed, scale=std_dev))  # Ensure non-negative speeds

        # Step 2: Few-Shot Learning for Road-Specific Fine-Tuning     
        knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
        known_speeds = trip_df.dropna(subset=['segment_speed_kmh'])[['STR_ID_encoded', 'segment_speed_kmh']]
        
        if len(known_speeds) > 3:
            knn.fit(known_speeds[['STR_ID_encoded']], known_speeds['segment_speed_kmh'])
            
            for i in range(len(trip_df)):
                if pd.isna(trip_df.loc[i, 'segment_speed_kmh']):
                    trip_df.loc[i, 'segment_speed_kmh'] = max(0, knn.predict([[trip_df.loc[i, 'STR_ID_encoded']]])[0])

        # Step 3: Compute Time from Speed (Ensuring Monotonicity)
        for i in range(1, len(trip_df)):
            if pd.isna(trip_df.loc[i, 'time_from_traj_start_sec']):
                prev_time = trip_df.loc[i-1, 'time_from_traj_start_sec']
                dist_diff = trip_df.loc[i, 'dist_from_traj_start_m'] - trip_df.loc[i-1, 'dist_from_traj_start_m']
                speed = trip_df.loc[i, 'segment_speed_kmh']

                if speed > 0:
                    estimated_time = prev_time + (dist_diff / (speed / 3.6))  # Convert km/h to m/s
                    trip_df.loc[i, 'time_from_traj_start_sec'] = max(prev_time + 1, estimated_time)  # Ensure monotonicity

        # Step 4: Apply Gaussian Smoothing to Speed Values
        trip_df['segment_speed_kmh'] = gaussian_filter1d(trip_df['segment_speed_kmh'], sigma=1)

        processed_trips.append(trip_df)  # Store processed trip

    # Combine all processed trips back into one DataFrame
    combined_df = pd.concat(processed_trips, ignore_index=True)
    
    return combined_df

#================================================================================================================================


#================================================================================================================================



#================================================================================================================================



#================================================================================================================================



#================================================================================================================================


#================================================================================================================================



#================================================================================================================================

