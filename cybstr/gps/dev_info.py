
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

def print_unique_gps_ids_v1(df, col_name, limt):
    """ 
    Convert a dataframe to geodataframe; and creates a geometry column.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe containing GPS IDs.
    col_name: int
        The dataframe column containing the GPS IDs e.g. 'registrationID'. 
    limt: int
        The frequency limit; e.g. if 10, will only show GPS IDs occuring more than 10 times.

    Returns
    -------
    gps_list: list
        List of GPS IDs.
    """
    unique_device_ids = df[col_name].unique()
    unique_device_ids_list = unique_device_ids.tolist()

    # Group by registrationID and filter for those with more than 20 entries
    grouped_device_ids = df.groupby(col_name).size()
    filtered_device_ids = grouped_device_ids[grouped_device_ids > limt].index.tolist()
    
    # Display
    return filtered_device_ids

#================================================================================================================================