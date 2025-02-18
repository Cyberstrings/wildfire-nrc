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


#================================================================================================================================

def calculate_trip_id(df, gps_col_name, date_col_name, timelimt):
    """
    Calculate trip ids based on a time limit i.e. if 1hr, every subsequent GPS entry after 1hr will be a new trip.

    Parameters
    ----------
    df : GeoDataFrame
        The GPS dataframe containing location and time.
    gps_col_name: string
        The dataframe column containing the GPS IDs e.g. 'registrationID'. 
    date_col_name: string
        The dataframe column containing the GPS IDs e.g. 'DateTime'.
    timelimt: int (seconds)
        The cutoff time for a new trip i.e. if 3600, every subsequent entry after 3600 sec/1 hr is a new trip.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing two new columns:
        time_diff = The time difference between GPS points in seconds
        trip_id = A number denoting the ID of the trip
    """

    # Convert DateTime column to datetime type
    df[date_col_name] = pd.to_datetime(df[date_col_name])
    
    # Sort by registrationID and DateTime to ensure proper order
    df = df.sort_values(by=[gps_col_name, date_col_name])
    
    # Calculate time difference in seconds between consecutive rows
    df.loc[:,"time_diff"] = df[date_col_name].diff().dt.total_seconds()
    
    # Assign trip_id based on the desired time gap
    df.loc[:,"trip_id"] = (df["time_diff"] > timelimt).cumsum() + 1
    
    # Fill NaN for the first row's time_diff
    df.loc[:,"time_diff"] = df["time_diff"].fillna(0)

    # Return the df
    return df

#================================================================================================================================

def split_trajectory_v1(trajectory, timestamps):
    """
    Convert a trajectory LineString into line segments with timestamps.

    Parameters:
    - trajectory: shapely.geometry.LineString, full trajectory
    - timestamps: list, timestamps corresponding to each GPS point

    Returns:
    - list of tuples [(segment, start_time, end_time), ...]
    """
    segments = []
    coords = list(trajectory.coords)
    
    for i in range(len(coords) - 1):
        segment = LineString([coords[i], coords[i + 1]])
        segments.append((segment, timestamps[i], timestamps[i + 1]))
    
    return segments

#================================================================================================================================

def split_trajectory_v2(trajectory, time_pos_dict):
    """
    Convert a trajectory LineString into segments with timestamps.
    Extract timestamps in order of appearance, Ensure timestamps match the LineString coordinates
    Handle potential mismatches, Return properly formatted trajectory segments

    Parameters:
    - trajectory: LineString (GPS trajectory)
    - time_pos_dict: Dictionary {timestamp: Point} from the dataframe

    Returns:
    - list of tuples [(segment, start_time, end_time), ...]
    """
    segments = []

    # Ensure trajectory is a valid LineString
    if not isinstance(trajectory, LineString):
        print("Invalid trajectory: Not a LineString")
        return segments  # Return empty list

    coords = list(trajectory.coords)
    timestamps = sorted(time_pos_dict.keys())  # Extract timestamps in order

    # Validate if timestamps align with trajectory points
    if len(coords) != len(timestamps):
        print(f"Skipping trajectory: {len(coords)} coords, {len(timestamps)} timestamps (mismatch)")
        return segments  # Return empty list

    # Create segments
    for i in range(len(coords) - 1):
        segment = LineString([coords[i], coords[i + 1]])
        segments.append((segment, timestamps[i], timestamps[i + 1]))

    return segments


#================================================================================================================================

def find_intersecting_segments(segments, junction):
    """
    Identify trajectory segments that intersect the junction.

    Parameters:
    - segments: list of tuples [(segment, start_time, end_time)]
    - junction: shapely.geometry.Polygon or Point, representing the junction area

    Returns:
    - list of tuples [(intersection_point, estimated_time)]
    """
    crossing_times = []
    
    for segment, start_time, end_time in segments:
        print(type(segment))
        if segment.intersects(junction):
            intersection = segment.intersection(junction)

            if isinstance(intersection, Point):  # If a point of intersection exists
                # Interpolate timestamp at intersection
                estimated_time = interpolate_time(segment, start_time, end_time, intersection)
                crossing_times.append((intersection, estimated_time))
    
    return crossing_times

#================================================================================================================================

def interpolate_time(segment, start_time, end_time, intersection):
    """
    Estimate the timestamp when a trajectory crosses a junction using linear interpolation.

    Parameters:
    - segment: LineString (start and end of segment)
    - start_time: datetime, time at start of segment
    - end_time: datetime, time at end of segment
    - intersection: Point, intersection of segment with junction

    Returns:
    - datetime, estimated crossing time
    """
    start_point = Point(segment.coords[0])
    end_point = Point(segment.coords[1])
    
    total_distance = start_point.distance(end_point)
    intersection_distance = start_point.distance(intersection)

    # Linear time interpolation
    time_fraction = intersection_distance / total_distance
    estimated_time = start_time + (end_time - start_time) * time_fraction

    return estimated_time

#================================================================================================================================

def count_trajectory_orders(trajectories_df, shapefile_a, shapefile_b):
    """
    Counts the number of trajectories passing through both shapefiles (checkpoints) and determines their order (direction of travel).
    
    Parameters:
    ----------
    trajectories_df : GeoDataFrame
        A GeoDataFrame containing trajectories with LineString geometries.
    shapefile_a : GeoDataFrame
        GeoDataFrame representing the first shapefile.
    shapefile_b : GeoDataFrame
        GeoDataFrame representing the second shapefile.

    Returns:
    -------
    dict
        A dictionary containing counts:
        - "total": total trajectories passing both shapefiles
        - "A_to_B": trajectories passing first A, then B
        - "B_to_A": trajectories passing first B, then A
    """
    total_count = 0
    a_to_b_count = 0
    b_to_a_count = 0

    for _, row in trajectories_df.iterrows():
        trajectory = row["geometry"]  # Extract LineString

        # Get intersection points (ensure Point type)
        a_intersection = trajectory.intersection(shapefile_a.unary_union)
        b_intersection = trajectory.intersection(shapefile_b.unary_union)

        def extract_valid_point(intersection):
            """Ensures the intersection is a Point or extracts a representative Point."""
            if isinstance(intersection, Point):
                return intersection
            elif intersection.geom_type in ["MultiPoint", "LineString", "MultiLineString"]:
                return intersection.representative_point()  # Pick a representative point
            else:
                return None

        a_point = extract_valid_point(a_intersection)
        b_point = extract_valid_point(b_intersection)

        # Ensure both intersections are valid
        if not a_point or not b_point:
            continue  # Skip if not intersecting both

        # Convert to nearest points along the trajectory
        a_distance = trajectory.project(a_point)
        b_distance = trajectory.project(b_point)

        total_count += 1
        if a_distance < b_distance:
            a_to_b_count += 1
        else:
            b_to_a_count += 1

    return {
        "total": total_count,
        "A_to_B": a_to_b_count,
        "B_to_A": b_to_a_count
    }

#================================================================================================================================

def compute_distance_along_trajectory_v1(gps_gdf, traj_gdf, crs="EPSG:3153"):
    """
    Computes the distance of GPS points along the corresponding trajectory, measured from the start of the trajectory.

    Parameters:
    gps_gdf (GeoDataFrame): GPS data containing registrationID, DateTime, and trip_id.
    traj_gdf (GeoDataFrame): Trajectory data containing trip_id and geometry (LineString).
    crs (str): Coordinate reference system for distance calculations (default is EPSG:3857).

    Returns:
    GeoDataFrame: gps_gdf with an added column 'dist_along_traj'.
    """
    
    # Ensure GPS and trajectory data have the correct CRS
    gps_gdf = gps_gdf.to_crs(crs)
    display(gps_gdf)
    traj_gdf = traj_gdf.to_crs(crs)
    display(traj_gdf)

    # Sort GPS data by registrationID and DateTime
    gps_gdf = gps_gdf.sort_values(by=["registrationID", "DateTime"]).reset_index(drop=True)

    # Initialize a column for distance along trajectory
    gps_gdf["dist_frm_traj_start_m"] = 0.0

    # Iterate through each unique trip_id
    for trip_id in gps_gdf["trip_id"].unique():
        # Get GPS points for this trip
        trip_gps = gps_gdf[gps_gdf["trip_id"] == trip_id]

        # Get the corresponding trajectory
        traj_row = traj_gdf[traj_gdf["trip_id"] == trip_id]

        if traj_row.empty:
            # If no trajectory is found, keep the GPS points as they are
            continue

        # Extract the trajectory LineString
        trajectory = traj_row.iloc[0]["geometry"]

        # Compute distance along the trajectory
        distances = trip_gps["geometry"].apply(lambda point: trajectory.project(point))

        # Assign distances to the GPS dataframe
        gps_gdf.loc[trip_gps.index, "dist_frm_traj_start_m"] = distances - distances.min()  # Normalize to start from 0

    return gps_gdf

#================================================================================================================================

def compute_distance_along_trajectory_v2(gps_gdf, traj_gdf, crs="EPSG:3153"):
    """
    Computes the segment-wise distance along the trajectory between consecutive GPS points.
    Change from v1: Instead of measuring from start of trajectory, it measures from last GPS point; segmentwise.

    Parameters:
    gps_gdf (GeoDataFrame): GPS data containing registrationID, DateTime, and trip_id.
    traj_gdf (GeoDataFrame): Trajectory data containing trip_id and geometry (LineString).
    crs (str): Coordinate reference system for distance calculations (default is EPSG:3153).

    Returns:
    GeoDataFrame: gps_gdf with an added column 'segment_dist' representing the distance
                  between each successive GPS point along the trajectory in meters.
    """ 
    # Ensure correct CRS (convert from EPSG:4326 to EPSG:3153)
    gps_gdf = gps_gdf.to_crs(crs)
    traj_gdf = traj_gdf.to_crs(crs)
    print(gps_gdf.crs)

    # Sort GPS data by trip_id and DateTime
    gps_gdf = gps_gdf.sort_values(by=["trip_id", "DateTime"]).reset_index(drop=True)

    # Initialize distance column
    gps_gdf["segment_dist_m"] = 0.0

    # Process each trip separately
    for trip_id in gps_gdf["trip_id"].unique():
        trip_gps = gps_gdf[gps_gdf["trip_id"] == trip_id].copy()
        traj_row = traj_gdf[traj_gdf["trip_id"] == trip_id]

        if traj_row.empty or len(trip_gps) < 2:
            continue  # Skip if no trajectory or only one GPS point

        trajectory = traj_row.iloc[0]["geometry"]

        # Compute segment-wise distances
        segment_distances = [0]  # First point has no previous distance
        for i in range(1, len(trip_gps)):
            prev_point = trip_gps.iloc[i - 1].geometry
            curr_point = trip_gps.iloc[i].geometry
            prev_proj = trajectory.project(prev_point)  # Project onto trajectory
            curr_proj = trajectory.project(curr_point)
            segment_distances.append(abs(curr_proj - prev_proj))  # Distance along trajectory

        # Assign distances to the GPS dataframe
        gps_gdf.loc[trip_gps.index, "segment_dist_m"] = segment_distances

    return gps_gdf

#================================================================================================================================

def compute_segment_speed_v1(gps_gdf):
    """
    Computes the segment-wise speed based on previously calculated distance.

    Parameters:
    gps_gdf (GeoDataFrame): GPS data containing registrationID, DateTime, and segment_dist.

    Returns:
    GeoDataFrame: gps_gdf with an added column 'segment_speed' representing the speed between each successive GPS point along the trajectory.
    """
    
    # Group by registrationID, then sort by trip_id and DateTime
    gps_gdf = gps_gdf.sort_values(by=['registrationID', 'trip_id', 'DateTime'], ascending=[True, True, True])

    # Convert datetime column to datetime format
    gps_gdf["DateTime"] = pd.to_datetime(gps_gdf["DateTime"])

    # Compute time difference in hours
    gps_gdf["segment_timediff_hrs"] = gps_gdf["DateTime"].diff().dt.total_seconds().div(3600).fillna(0)

    # Compute speed in km/h
    gps_gdf["segment_speed_kmh"] = (gps_gdf["segment_dist_m"]/1000) / gps_gdf["segment_timediff_hrs"]

    return gps_gdf

#================================================================================================================================

def get_travelled_roads_v1(roads_df, trajectories_df, trip_id, crs="EPSG:4326", metric_crs="EPSG:3857"):
    """
    Returns the ordered list of road names a vehicle traveled along during a trip.

    Parameters:
    - roads_df (GeoDataFrame): The roads dataset with LINESTRING geometries and road names.
    - trajectories_df (GeoDataFrame): The matched trajectory dataset with LINESTRING geometries.
    - trip_id (int or str): The trip identifier to analyze.
    - crs (str): The original CRS of the data (default: "EPSG:4326").
    - metric_crs (str): The CRS for distance calculations (default: "EPSG:3857").

    Returns:
    - List[str]: Ordered list of road names traveled, without duplicates and NaN values.
    """
    
    # Convert to a metric CRS for accurate distance calculations
    roads_df = roads_df.to_crs(metric_crs)
    trajectories_df = trajectories_df.to_crs(metric_crs)
    
    # Extract the trajectory for the given trip_id
    trip = trajectories_df[trajectories_df["trip_id"] == trip_id]
    
    if trip.empty:
        raise ValueError(f"Trip ID {trip_id} not found in trajectories_df.")

    trajectory = trip.iloc[0].geometry  # Assuming only one trajectory per trip

    # Spatial join: Find roads intersecting the trajectory
    intersecting_roads = gpd.sjoin(roads_df, gpd.GeoDataFrame(geometry=[trajectory], crs=metric_crs), how="inner")

    if intersecting_roads.empty:
        return []  # No roads found

    # Extract unique road names, preserving order of first appearance
    road_sequence = []
    seen_roads = set()
    
    for point in trajectory.coords:  # Iterate over trajectory points
        point_geom = Point(point)
        
        # Find the closest intersecting road
        nearest_road = intersecting_roads.distance(point_geom).idxmin()
        road_name = intersecting_roads.loc[nearest_road, "FULLNAME"]
        
        if road_name and road_name not in seen_roads:
            road_sequence.append(road_name)
            seen_roads.add(road_name)

    return road_sequence

#================================================================================================================================

def get_travelled_roads_v2(roads_df, trajectories_df, trip_id, crs="EPSG:4326", metric_crs="EPSG:3857", max_distance=2):
    """
    Returns the ordered list of road names a vehicle traveled along during a trip, excluding intersection roads.

    Parameters:
    - roads_df (GeoDataFrame): Roads dataset with LINESTRING geometries and road names.
    - trajectories_df (GeoDataFrame): Matched trajectory dataset with LINESTRING geometries.
    - trip_id (int or str): The trip identifier to analyze.
    - crs (str): Original CRS of the data (default: "EPSG:4326").
    - metric_crs (str): CRS for distance calculations (default: "EPSG:3857").
    - max_distance (float): Maximum allowed distance (meters) between the trajectory and road for it to be considered traveled.
    Smaller max_distance means only roads with large overlaps are considered.

    Returns:
    - List[str]: Ordered list of road names traveled, without intersection roads or duplicates.
    """

    # Convert to a metric CRS for accurate distance calculations
    roads_df = roads_df.to_crs(metric_crs)
    trajectories_df = trajectories_df.to_crs(metric_crs)

    # Extract the trajectory for the given trip_id
    trip = trajectories_df[trajectories_df["trip_id"] == trip_id]
    
    if trip.empty:
        raise ValueError(f"Trip ID {trip_id} not found in trajectories_df.")

    trajectory = trip.iloc[0].geometry  # Assuming only one trajectory per trip

    # Spatial join: Find roads intersecting the trajectory
    intersecting_roads = gpd.sjoin(roads_df, gpd.GeoDataFrame(geometry=[trajectory], crs=metric_crs), how="inner")

    if intersecting_roads.empty:
        return []  # No roads found

    # Extract roads that are actually traveled on
    road_sequence = []
    seen_roads = set()
    
    for point in trajectory.coords:  # Iterate over trajectory points
        point_geom = Point(point)

        # Find the closest road to this trajectory point
        closest_road = None
        min_distance = float("inf")

        for _, road in intersecting_roads.iterrows():
            road_geom = road.geometry
            nearest_point_on_road = nearest_points(point_geom, road_geom)[1]  # Get nearest point on road
            
            distance = point_geom.distance(nearest_point_on_road)

            if distance < min_distance:
                min_distance = distance
                closest_road = road

        # Only add if the distance is within the allowed threshold
        if closest_road is not None and min_distance <= max_distance:
            road_name = closest_road["FULLNAME"]
            
            if road_name and road_name not in seen_roads:
                road_sequence.append(road_name)
                seen_roads.add(road_name)

    return road_sequence

#================================================================================================================================

def get_detailed_road_sequence_v1(roads_df, trajectories_df, trip_id, crs="EPSG:4326", metric_crs="EPSG:3857", max_distance=15):
    """
    Returns the ordered list of road names a vehicle traveled along during a trip, including repetitions.

    Parameters:
    - roads_df (GeoDataFrame): Roads dataset with LINESTRING geometries and road names.
    - trajectories_df (GeoDataFrame): Matched trajectory dataset with LINESTRING geometries.
    - trip_id (int or str): The trip identifier to analyze.
    - crs (str): Original CRS of the data (default: "EPSG:4326").
    - metric_crs (str): CRS for distance calculations (default: "EPSG:3857").
    - max_distance (float): Maximum allowed distance (meters) between the trajectory and road for it to be considered traveled.

    Returns:
    - List[str]: Full ordered list of road names traveled (including repetitions).
    """

    # Convert to a metric CRS for accurate distance calculations
    roads_df = roads_df.to_crs(metric_crs)
    trajectories_df = trajectories_df.to_crs(metric_crs)

    # Extract the trajectory for the given trip_id
    trip = trajectories_df[trajectories_df["trip_id"] == trip_id]
    
    if trip.empty:
        raise ValueError(f"Trip ID {trip_id} not found in trajectories_df.")

    trajectory = trip.iloc[0].geometry  # Assuming only one trajectory per trip

    # Spatial join: Find roads intersecting the trajectory
    intersecting_roads = gpd.sjoin(roads_df, gpd.GeoDataFrame(geometry=[trajectory], crs=metric_crs), how="inner")

    if intersecting_roads.empty:
        return []  # No roads found

    road_sequence = []  # Stores all road names in order

    for point in trajectory.coords:  # Iterate over all trajectory points
        point_geom = Point(point)

        # Find the closest road to this trajectory point
        closest_road = None
        min_distance = float("inf")

        for _, road in intersecting_roads.iterrows():
            road_geom = road.geometry
            nearest_point_on_road = nearest_points(point_geom, road_geom)[1]  # Get nearest point on road
            
            distance = point_geom.distance(nearest_point_on_road)

            if distance < min_distance:
                min_distance = distance
                closest_road = road

        # Only add if the distance is within the allowed threshold
        if closest_road is not None and min_distance <= max_distance:
            road_name = closest_road["FULLNAME"]
            if road_name:  # Ignore NaN or empty names
                road_sequence.append(road_name)

    return road_sequence
