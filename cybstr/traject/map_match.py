import os
import sys
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

import contextlib
from copy import copy
from pyproj import CRS
import contextily as ctx
import movingpandas as mpd
import matplotlib.pyplot as plt
from urllib.request import urlopen
from datetime import datetime, timedelta
from shapely.geometry import Point, LineString

from .. import pytrack

def map_match_v1(df, graph, day_col="DateTime", gps_id_col="registrationID", interp_dist=50, radius=90):
    """
    Map-match GPS points and return the original DataFrame with new columns 'matched_latitude' and 'matched_longitude'.
    
    Parameters
    ----------
    df : GeoDataFrame
        The GPS dataframe containing trajectory data for multiple trips.
    graph : networkx.MultiDiGraph
        The road network graph.
    day_col : str
        Column name for the day information in the dataframe.
    gps_id_col : str
        Column name for the GPS identifier in the dataframe.
    trip_id_col : str
        Column name for trip IDs in the dataframe.
    interp_dist : int
        Interpolation distance for the graph.
    radius : int
        Search radius for candidate extraction.

    Returns
    -------
    df: gpd.GeoDataFrame
        A GeoDataFrame containing the map-matched points for each GPS point based on road network graph provided.
    failed_matches: list
        List of points failed to match.
    """
    df["matched_latitude"] = None
    df["matched_longitude"] = None
    failed_matches = []

    points = list(zip(df["latitude"], df["longitude"]))

    try:
        # Perform candidate extraction
        print('Performing candidate extraction...')
        G_interp, candidates = pytrack.matching.candidate.get_candidates(graph, points, interp_dist=interp_dist, closest=True, radius=radius)
        
        # Create trellis graph and perform map matching
        print('Create trellis graph and performing viterbi search...')
        trellis = pytrack.matching.mpmatching_utils.create_trellis(candidates)
        path_prob, predecessor = pytrack.matching.mpmatching.viterbi_search(G_interp, trellis, "start", "target")
        node_ids = pytrack.matching.mpmatching_utils.create_path(G_interp, trellis, predecessor)

        for i, (lat, lon) in enumerate(points):
            sys.stdout.write(f"\rProcessing row {i+1}/{len(points)}...")
            sys.stdout.flush()

            if i < len(node_ids):
                matched_coord = G_interp.nodes[node_ids[i]]["geometry"]
                df.at[i, "matched_latitude"] = matched_coord.y
                df.at[i, "matched_longitude"] = matched_coord.x
            else:
                failed_matches.append({
                    "gps_id": df.at[i, gps_id_col],
                    "date_time": df.at[i, day_col],
                    "latitude": lat,
                    "longitude": lon
                })
    
    except Exception as e:
        print(f"Map matching failed: {e}")
        failed_matches = df[[gps_id_col, day_col, "latitude", "longitude"]].to_dict(orient="records")
    
    sys.stdout.write("\n")
    sys.stdout.flush()
    
    return df, failed_matches


#================================================================================================================================

def map_match_and_return_geodataframe(df, graph, day_col="DateTime", gps_id_col="registrationID", trip_id_col="trip_id", interp_dist=50, radius=90):
    """
    Map-match GPS trajectories for each trip and return a GeoDataFrame with the results.

    Parameters
    ----------
    df : GeoDataFrame
        The GPS dataframe containing trajectory data for multiple trips.
    graph : networkx.MultiDiGraph
        The road network graph.
    day_col : str
        Column name for the day information in the dataframe.
    gps_id_col : str
        Column name for the GPS identifier in the dataframe.
    trip_id_col : str
        Column name for trip IDs in the dataframe.
    interp_dist : int
        Interpolation distance for the graph.
    radius : int
        Search radius for candidate extraction.

    Returns
    -------
    traj_gpd: gpd.GeoDataFrame
        A GeoDataFrame containing the map-matched LineStrings for each trip.
    skipped_trips: list
        A list of trip # which failed due to lack of points.
    """
    results = []  # List to store rows for the GeoDataFrame
    skipped_trips = []  # List to store trips that failed due to insufficient points

    # Process each trip
    unique_trips = df[trip_id_col].unique()
    for trip_id in unique_trips:
        print(f"Processing trip ID: {trip_id}...")

        # Extract data for the current trip
        trip_data = df[df[trip_id_col] == trip_id]
        latitudes = trip_data["latitude"].to_list()
        longitudes = trip_data["longitude"].to_list()
        points = [(lat, lon) for lat, lon in zip(latitudes, longitudes)]

        try:
            # Perform candidate extraction
            G_interp, candidates = pytrack.matching.candidate.get_candidates(graph, points, interp_dist=interp_dist, closest=True, radius=radius)

            # Create a trellis DAG graph
            trellis = pytrack.matching.mpmatching_utils.create_trellis(candidates)

            # Perform map matching
            path_prob, predecessor = pytrack.matching.mpmatching.viterbi_search(G_interp, trellis, "start", "target")
            node_ids = pytrack.matching.mpmatching_utils.create_path(G_interp, trellis, predecessor)

            # Generate a LineString geometry for the matched path
            if len(node_ids) > 1:  # Ensure valid LineString
                edge_geom = LineString([G_interp.nodes[node]["geometry"] for node in node_ids])

                # Append the results
                results.append({
                    "day": trip_data[day_col].iloc[0].date(),  # Extract the day
                    "gps_id": trip_data[gps_id_col].iloc[0],   # Extract GPS ID
                    "trip_id": trip_id,                        # Trip ID
                    "geometry": edge_geom                      # Matched LineString
                })
            else:
                print(f"Skipping trip {trip_id}: insufficient points in the matched path.")
                skipped_trips.append(trip_id)

        except Exception as e:
            print(f"Failed to process trip {trip_id}: {e}")
            skipped_trips.append(trip_id)

    # Ensure results contain valid data
    if not results:
        print("No valid trajectories were processed.")
        return gpd.GeoDataFrame(columns=["day", "gps_id", "trip_id", "geometry"], crs="EPSG:4326"), skipped_trips

    # Create and return a GeoDataFrame
    traj_gpd = gpd.GeoDataFrame(results, crs="EPSG:4326")
    return traj_gpd, skipped_trips


#================================================================================================================================

def map_match_and_return_geodataframe_v2(df, log_name, graph, day_col="DateTime", gps_id_col="registrationID", trip_id_col="trip_id", interp_dist=50, radius=90, max_trips=100):
    """
    Map-match GPS trajectories for each trip and return a GeoDataFrame with the results.
    New changes: Added max_trips to prevent from processing all trips and produce log file instead of on screen outputs.

    Parameters
    ----------
    df : GeoDataFrame
        The GPS dataframe containing trajectory data for multiple trips.
    log_name: str
        Name of the log output and csv output
    graph : networkx.MultiDiGraph
        The road network graph.
    day_col : str
        Column name for the day information in the dataframe.
    gps_id_col : str
        Column name for the GPS identifier in the dataframe.
    trip_id_col : str
        Column name for trip IDs in the dataframe.
    interp_dist : int
        Interpolation distance for the graph.
    radius : int
        Search radius for candidate extraction.
    max_trips : int
        Maximum number of trips to process (default: 1000).

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the map-matched LineStrings for each trip.
    list
        A list of trip IDs that were skipped due to insufficient points or errors.
    """
    # Set up logging
    log_file = f"{log_name}.log"
    csv_file = f"{log_name}_trips.csv"

    with open(log_file, "w") as log:
        # Initialize storage
        results = []  # List to store rows for the GeoDataFrame
        skipped_trips = []  # List to store trips which failed
        
        # Limit trips if max_trips is specified
        unique_trips = df[trip_id_col].unique()
        if max_trips is not None:
            unique_trips = unique_trips[:max_trips]
        total_trips = len(unique_trips)

        for i, trip_id in enumerate(unique_trips, 1):
            # Show dynamic progress output
            sys.stdout.write(f"\rProcessing trip {i}/{total_trips}...")
            sys.stdout.flush()

            try:
                # Redirect output to suppress excessive messages
                with contextlib.redirect_stdout(log):
                    # Extract data for the current trip
                    trip_data = df[df[trip_id_col] == trip_id]
                    latitudes = trip_data["latitude"].to_list()
                    longitudes = trip_data["longitude"].to_list()
                    points = [(lat, lon) for lat, lon in zip(latitudes, longitudes)]

                    # Perform candidate extraction
                    G_interp, candidates = pytrack.matching.candidate.get_candidates(graph, points, interp_dist=interp_dist, closest=True, radius=radius)

                    # Log points with no candidates
                    no_candidate_indices = [idx for idx, cands in enumerate(candidates) if not cands]
                    if no_candidate_indices:
                        log.write(f"Trip ID {trip_id}: {len(no_candidate_indices)} points have no candidates: {tuple(no_candidate_indices)}\n")

                    # Create a trellis DAG graph
                    trellis = pytrack.matching.mpmatching_utils.create_trellis(candidates)

                    # Perform map matching
                    path_prob, predecessor = pytrack.matching.mpmatching.viterbi_search(G_interp, trellis, "start", "target")
                    node_ids = pytrack.matching.mpmatching_utils.create_path(G_interp, trellis, predecessor)

                # Generate a LineString geometry for the matched path
                if len(node_ids) > 1:  # Ensure valid LineString
                    edge_geom = LineString([G_interp.nodes[node]["geometry"] for node in node_ids])

                    # Append the results
                    results.append({
                        "day": trip_data[day_col].iloc[0], #.date(),  # Extract the day
                        "gps_id": trip_data[gps_id_col].iloc[0],   # Extract GPS ID
                        "trip_id": trip_id,                        # Trip ID
                        "geometry": edge_geom                      # Matched LineString
                    })
                else:
                    skipped_trips.append(trip_id)
                    log.write(f"Skipping trip {trip_id}: insufficient points in the matched path.\n")

            except Exception as e:
                skipped_trips.append(trip_id)
                log.write(f"Failed to process trip {trip_id}: {e}\n")

        # Clear progress output
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Create GeoDataFrame and save as CSV
        traj_gpd = gpd.GeoDataFrame(results, crs="EPSG:4326")
        traj_gpd.to_csv(csv_file, index=False)

        # Write final log messages
        log.write(f"Processed {total_trips} trips. Skipped {len(skipped_trips)} trips.\n")
        log.write(f"Skipped trip IDs: {skipped_trips}\n")

    return traj_gpd, skipped_trips

#================================================================================================================================

def map_match_and_return_geodataframe_v3(df, log_name, graph, day_col="DateTime", gps_id_col="registrationID",
                                        trip_id_col="trip_id", interp_dist=50, radius=90, max_trips=None, ):
    """
    Map-match GPS trajectories for each trip and return a GeoDataFrame with the results.
    New changes: Added a new column 'time_pos_dict' which will contain a dictionary in the format {'timestamp': 'geometry'}. 
    The datetime column now stores only date and not time.

    Parameters
    ----------
    df : GeoDataFrame
        The GPS dataframe containing trajectory data for multiple trips.
    log_name: str
        Name of the log output and csv output
    graph : networkx.MultiDiGraph
        The road network graph.
    day_col : str
        Column name for the day information in the dataframe.
    gps_id_col : str
        Column name for the GPS identifier in the dataframe.
    trip_id_col : str
        Column name for trip IDs in the dataframe.
    interp_dist : int
        Interpolation distance for the graph.
    radius : int
        Search radius for candidate extraction.
    max_trips : int, optional
        Maximum number of trips to process (default: None for all trips).

    Returns
    -------
    traj_gpd: GeoDataFrame
        A GeoDataFrame containing the map-matched LineStrings and a 'time_pos_dict' column.
    skipped_trips: list
        A list of trip IDs that were skipped due to insufficient points or errors.
    """
    # Set up logging
    log_file = f"{log_name}_log.txt"
    csv_file = f"{log_name}_trips.csv"

    with open(log_file, "w") as log:
        # Initialize storage
        results = []  # List to store rows for the GeoDataFrame
        skipped_trips = []  # List to store trips which failed

        # Limit trips if max_trips is specified
        unique_trips = df[trip_id_col].unique()
        if max_trips is not None:
            unique_trips = unique_trips[:max_trips]
        total_trips = len(unique_trips)

        for i, trip_id in enumerate(unique_trips, 1):
            # Show dynamic progress output
            sys.stdout.write(f"\rProcessing trip {i}/{total_trips}...")
            sys.stdout.flush()

            try:
                # Redirect output to suppress excessive messages
                with contextlib.redirect_stdout(log):
                    # Extract data for the current trip
                    trip_data = df[df[trip_id_col] == trip_id]
                    latitudes = trip_data["latitude"].to_list()
                    longitudes = trip_data["longitude"].to_list()
                    timestamps = trip_data[day_col].to_list()
                    points = [(lat, lon) for lat, lon in zip(latitudes, longitudes)]

                    # Perform candidate extraction
                    G_interp, candidates = pytrack.matching.candidate.get_candidates(graph, points, interp_dist=interp_dist, closest=True, radius=radius)

                    # Log points with no candidates
                    no_candidate_indices = [idx for idx, cands in enumerate(candidates) if not cands]
                    if no_candidate_indices:
                        log.write(f"Trip ID {trip_id}: {len(no_candidate_indices)} points have no candidates: {tuple(no_candidate_indices)}\n")

                    # Create a trellis DAG graph
                    trellis = pytrack.matching.mpmatching_utils.create_trellis(candidates)

                    # Perform map matching
                    path_prob, predecessor = pytrack.matching.mpmatching.viterbi_search(G_interp, trellis, "start", "target")
                    node_ids = pytrack.matching.mpmatching_utils.create_path(G_interp, trellis, predecessor)

                # Generate a LineString geometry for the matched path
                if len(node_ids) > 1:  # Ensure valid LineString
                    edge_geom = LineString([G_interp.nodes[node]["geometry"] for node in node_ids])

                    # Create the time_pos_dict
                    time_pos_dict = {timestamp: G_interp.nodes[node]["geometry"] for timestamp, node in zip(timestamps, node_ids)}

                    # Append the results
                    results.append({
                        "day": trip_data[day_col].iloc[0].date(), # Extract the day
                        "gps_id": trip_data[gps_id_col].iloc[0],  # Extract GPS ID
                        "trip_id": trip_id,  # Trip ID
                        "geometry": edge_geom,  # Matched LineString
                        "time_pos_dict": time_pos_dict  # Timestamp to position dictionary
                    })
                else:
                    skipped_trips.append(trip_id)
                    log.write(f"Skipping trip {trip_id}: insufficient points in the matched path.\n")

            except Exception as e:
                skipped_trips.append(trip_id)
                log.write(f"Failed to process trip {trip_id}: {e}\n")

        # Clear progress output
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Create GeoDataFrame and save as CSV
        traj_gpd = gpd.GeoDataFrame(results, crs="EPSG:4326")
        traj_gpd.to_csv(csv_file, index=False)

        # Write final log messages
        log.write(f"Processed {total_trips} trips. Skipped {len(skipped_trips)} trips.\n")
        log.write(f"Skipped trip IDs: {skipped_trips}\n")

    return traj_gpd, skipped_trips

#================================================================================================================================

import geopandas as gpd
import pandas as pd
import sys
import contextlib
from shapely.geometry import LineString

def map_match_and_return_geodataframe_v4(df, log_name, graph, day_col="DateTime", gps_id_col="registrationID",
                                        trip_id_col="trip_id", interp_dist=50, radius=90, max_trips=None):
    """
    Map-match GPS trajectories for each trip and return a GeoDataFrame with the results.
    New changes: Returns the df with columns 'matched_lat' and 'matched_long'

    Parameters
    ----------
    df : GeoDataFrame
        The GPS dataframe containing trajectory data for multiple trips.
    log_name: str
        Name of the log output and csv output.
    graph : networkx.MultiDiGraph
        The road network graph.
    day_col : str
        Column name for the day information in the dataframe.
    gps_id_col : str
        Column name for the GPS identifier in the dataframe.
    trip_id_col : str
        Column name for trip IDs in the dataframe.
    interp_dist : int
        Interpolation distance for the graph.
    radius : int
        Search radius for candidate extraction.
    max_trips : int, optional
        Maximum number of trips to process (default: None for all trips).

    Returns
    -------
    df : DataFrame
        The original DataFrame with two additional columns ('matched_lat', 'matched_long').
    traj_gpd : GeoDataFrame
        A GeoDataFrame containing the map-matched LineStrings and a 'time_pos_dict' column.
    list
        A list of trip IDs that were skipped due to insufficient points or errors.
    """
    # Set up logging
    log_file = f"{log_name}_log.txt"
    csv_file = f"{log_name}_trips.csv"

    with open(log_file, "w") as log:
        # Initialize storage
        results = []  # List to store rows for the GeoDataFrame
        skipped_trips = []  # List to store trips which failed
        matched_coords = {}  # Dictionary to store matched coordinates for each (gps_id, trip_id, timestamp)

        # Limit trips if max_trips is specified
        unique_trips = df[trip_id_col].unique()
        if max_trips is not None:
            unique_trips = unique_trips[:max_trips]
        total_trips = len(unique_trips)

        for i, trip_id in enumerate(unique_trips, 1):
            # Show dynamic progress output
            sys.stdout.write(f"\rProcessing trip {i}/{total_trips}...")
            sys.stdout.flush()

            try:
                # Redirect output to suppress excessive messages
                with contextlib.redirect_stdout(log):
                    # Extract data for the current trip
                    trip_data = df[df[trip_id_col] == trip_id]
                    latitudes = trip_data["latitude"].to_list()
                    longitudes = trip_data["longitude"].to_list()
                    timestamps = trip_data[day_col].to_list()
                    points = [(lat, lon) for lat, lon in zip(latitudes, longitudes)]

                    # Perform candidate extraction
                    G_interp, candidates = pytrack.matching.candidate.get_candidates(graph, points, interp_dist=interp_dist, closest=True, radius=radius)

                    # Log points with no candidates
                    no_candidate_indices = [idx for idx, cands in enumerate(candidates) if not cands]
                    if no_candidate_indices:
                        log.write(f"Trip ID {trip_id}: {len(no_candidate_indices)} points have no candidates: {tuple(no_candidate_indices)}\n")

                    # Create a trellis DAG graph
                    trellis = pytrack.matching.mpmatching_utils.create_trellis(candidates)

                    # Perform map matching
                    path_prob, predecessor = pytrack.matching.mpmatching.viterbi_search(G_interp, trellis, "start", "target")
                    node_ids = pytrack.matching.mpmatching_utils.create_path(G_interp, trellis, predecessor)

                # Generate a LineString geometry for the matched path
                if len(node_ids) > 1:  # Ensure valid LineString
                    edge_geom = LineString([G_interp.nodes[node]["geometry"] for node in node_ids])

                    # Create the time_pos_dict
                    time_pos_dict = {timestamp: G_interp.nodes[node]["geometry"] for timestamp, node in zip(timestamps, node_ids)}

                    # Store matched coordinates for each timestamp
                    for timestamp, geom in time_pos_dict.items():
                        matched_coords[(trip_data[gps_id_col].iloc[0], trip_id, timestamp)] = (geom.y, geom.x)  # Extract lat, long

                    # Append the results
                    results.append({
                        "day": trip_data[day_col].iloc[0].date(),  # Extract the day
                        "gps_id": trip_data[gps_id_col].iloc[0],  # Extract GPS ID
                        "trip_id": trip_id,  # Trip ID
                        "geometry": edge_geom,  # Matched LineString
                        "time_pos_dict": time_pos_dict  # Timestamp to position dictionary
                    })
                else:
                    skipped_trips.append(trip_id)
                    log.write(f"Skipping trip {trip_id}: insufficient points in the matched path.\n")

            except Exception as e:
                skipped_trips.append(trip_id)
                log.write(f"Failed to process trip {trip_id}: {e}\n")

        # Clear progress output
        sys.stdout.write("\n")
        sys.stdout.flush()
        
        if results:
            # Create GeoDataFrame and save as CSV
            traj_gpd = gpd.GeoDataFrame(results, crs="EPSG:4326")
            traj_gpd.to_csv(csv_file, index=False)

        # Write final log messages
        log.write(f"Processed {total_trips} trips. Skipped {len(skipped_trips)} trips.\n")
        log.write(f"Skipped trip IDs: {skipped_trips}\n")

    # Add matched latitude and longitude to df
    df["matched_lat"] = df.apply(lambda row: matched_coords.get((row[gps_id_col], row[trip_id_col], row[day_col]), (None, None))[0], axis=1)
    df["matched_long"] = df.apply(lambda row: matched_coords.get((row[gps_id_col], row[trip_id_col], row[day_col]), (None, None))[1], axis=1)

    return df, traj_gpd, skipped_trips

#================================================================================================================================

def map_points_to_linestring_by_trip_v1(gps_gdf, traj_gdf, crs="EPSG:4326", metric_crs="EPSG:3857"):
    """
    Maps each GPS point onto the closest point on the corresponding trip LineString.

    Args:
        gps_gdf (GeoDataFrame): GPS data with point geometries and trip_id.
        traj_gdf (GeoDataFrame): Trajectory data with LineString geometries and trip_id.
        crs (str): Geographic coordinate system (default: "EPSG:4326").
        metric_crs (str): Projected coordinate system for accurate distance calculations (default: "EPSG:3857").

    Returns:
        GeoDataFrame: Same GPS df with new columns 'mapped_lat' and 'mapped_long' in EPSG:3857
    """
    
    # Convert both GPS and trajectory data to metric CRS
    gps_gdf = gps_gdf.to_crs(metric_crs)
    traj_gdf = traj_gdf.to_crs(metric_crs)
    gps_gdf = gps_gdf.set_crs(metric_crs)

    # Create a dictionary of trip_id -> LineString
    traj_dict = traj_gdf.set_index("trip_id")["geometry"].to_dict()

    def map_point(row):
        """Finds the closest point on the corresponding trip LineString."""
        trip_id = row["trip_id"]
        gps_point = row["geometry"]
        
        # Check if trip_id exists in trajectory dictionary
        if trip_id in traj_dict:
            line = traj_dict[trip_id]
            mapped_point = line.interpolate(line.project(gps_point))
        else:
            mapped_point = gps_point  # No matching trip_id, use original point

        return mapped_point

    # Apply mapping function
    gps_gdf["mapped_geometry"] = gps_gdf.apply(map_point, axis=1)

    # Ensure the mapped geometry is in the correct CRS (EPSG:4326)
    gps_gdf["mapped_geometry"] = gps_gdf["mapped_geometry"].set_crs(metric_crs, allow_override=True)
    gps_gdf = gps_gdf.to_crs(crs)

    # Extract latitude & longitude from mapped points
    gps_gdf["mapped_lat"] = gps_gdf["mapped_geometry"].y
    gps_gdf["mapped_long"] = gps_gdf["mapped_geometry"].x

    return gps_gdf.drop(columns=["mapped_geometry"])

#================================================================================================================================

def map_points_to_linestring_by_trip_v2(gps_gdf_original, traj_gdf, crs="EPSG:4326", metric_crs="EPSG:3857"):
    """
    Maps each GPS point onto the closest point on the corresponding trip LineString
    and returns a new dataframe with only the mapped latitude, longitude, and geometry.

    Args:
        gps_gdf (GeoDataFrame): GPS data with point geometries and trip_id.
        traj_gdf (GeoDataFrame): Trajectory data with LineString geometries and trip_id.
        crs (str): Geographic coordinate system (default: "EPSG:4326").
        metric_crs (str): Projected coordinate system for accurate distance calculations (default: "EPSG:3857").

    Returns:
        GeoDataFrame: A new dataframe with columns ['trip_id', 'latitude', 'longitude', 'geometry']
    """
    # Convert both GPS and trajectory data to metric CRS for distance calculations
    gps_gdf = gps_gdf_original.copy()
    gps_gdf = gps_gdf.to_crs(metric_crs)
    traj_gdf = traj_gdf.to_crs(metric_crs)
    
    # Create a dictionary of trip_id -> LineString
    traj_dict = traj_gdf.set_index("trip_id")["geometry"].to_dict()

    def map_point(row):
        """Finds the closest point on the corresponding trip LineString."""
        trip_id = row["trip_id"]
        gps_point = row["geometry"]
        
        if trip_id in traj_dict:
            line = traj_dict[trip_id]
            mapped_point = line.interpolate(line.project(gps_point))
        else:
            mapped_point = gps_point  # No matching trip_id, use original point

        return mapped_point
    
    # Apply mapping function
    gps_gdf["geometry"] = gps_gdf.apply(map_point, axis=1)
    
    # Convert back to lat/lon (EPSG:4326)
    gps_gdf = gps_gdf.to_crs(crs)
    
    # Extract latitude & longitude
    gps_gdf["latitude"] = gps_gdf["geometry"].y
    gps_gdf["longitude"] = gps_gdf["geometry"].x
    
    # Keep only the necessary columns
    return gps_gdf[["DateTime", "registrationID", "trip_id", "latitude", "longitude", "geometry"]]