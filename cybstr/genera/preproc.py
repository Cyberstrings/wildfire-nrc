
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

def drop_less_freq_ids(df, col_name, freq):
    """ 
    Drop columns with GPS IDs occuring less than 'freq' times

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe containing GPS IDs.
    col_name: string
        The dataframe column containing the GPS IDs e.g. 'registrationID'. 
    freq: int
        The frequency limit; e.g. if 3, will drop IDs which occur less than 3 times.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe after IDs have been dropped.
    num_less_freq_ids: int
        The number of IDs occurring less than 'freq' times.
    """
    # Drop duplicates
    df = df.drop_duplicates()

    # Count occurrences of each registrationID
    registration_counts = df[col_name].value_counts()

    # Identify IDs that occur less than 4 times
    less_freq_ids = registration_counts[registration_counts < freq].index

    # Calculate the number of such IDs
    num_less_freq_ids = len(less_freq_ids)

    # Drop rows with registrationIDs that occur only once
    df = df[~df[col_name].isin(less_freq_ids)]

    # Return
    return df, num_less_freq_ids

#================================================================================================================================

def drop_duplicate_gps(df, gps_col_name, date_col_name):
    """ 
    Drop columns with duplicate GPS ID + Time value

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe containing GPS IDs.
    gps_col_name: string
        The dataframe column containing the GPS IDs e.g. 'registrationID'. 
    date_col_name: string
        The dataframe column containing the GPS IDs e.g. 'DateTime'.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe after IDs have been dropped.
    rows_dropped: int
        The number of rows dropped.
    """
    # Drop duplicates
    df = df.drop_duplicates()
    df[date_col_name] = pd.to_datetime(df[date_col_name])

    # Shape before dropping duplicates
    rows_before = df.shape[0]

    # Drop rows with duplicate registrationID and DateTime
    df = df.drop_duplicates(subset=[gps_col_name, date_col_name])

    # Shape after dropping duplicates
    rows_after = df.shape[0]

    # Calculate and print the number of rows dropped
    rows_dropped = rows_before - rows_after
    
    # Display
    return df, rows_dropped
    
#================================================================================================================================

def remove_static_gps_entries(df):
    """
    Removes GPS entries where the device shows no movement (same latitude and longitude rounded to 3 decimal places).
    
    Args:
        df (pd.DataFrame): DataFrame containing GPS data with 'registrationID', 'latitude', and 'longitude'.
    
    Returns:
        pd.DataFrame: Filtered DataFrame with moving GPS entries only.
    """
    # Round latitude and longitude to 3 decimal places
    df["_lat_rounded"] = df["latitude"].round(3)
    df["_lon_rounded"] = df["longitude"].round(3)

    # Identify registrationIDs with no movement
    static_ids = df.groupby("registrationID").filter(lambda x: x[["_lat_rounded", "_lon_rounded"]].nunique().sum() == 2)["registrationID"].unique()

    # Count rows to be dropped
    rows_dropped = df[df["registrationID"].isin(static_ids)].shape[0]

    # Drop static registrationIDs
    df_filtered = df[~df["registrationID"].isin(static_ids)].copy()

    # Drop temporary columns
    df_filtered.drop(columns=["_lat_rounded", "_lon_rounded"], inplace=True)

    # Print the number of IDs and rows dropped
    print(f"Dropped {len(static_ids)} registrationIDs with no movement, removing {rows_dropped} rows.")

    return df_filtered
