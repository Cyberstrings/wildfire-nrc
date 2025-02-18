
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

def divide_df_into_each_day(df, df_name, dates, output_dir):
    """ 
    Convert a dataframe to geodataframe; and creates a geometry column.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe containing GPS IDs.
    df_name: string
        Name of the dataframes to be saved as. 
    dates: string list[]
        List of all dates in string format e.g. ['2023-08-01', '2023-08-02', '2023-08-03',...]. 
    output_dir: string
        Output directory of the divided dfs.

    Returns
    -------
    0
    """
    # Save each day's dataframe as a separate CSV
    df['DateTime'] = pd.to_datetime(df["DateTime"], format='mixed')
    for date in dates:
        filename = os.path.join(output_dir, f"{df_name}_{date}.csv")
        temp_df = df[df['DateTime'].dt.date == pd.to_datetime(date).date()]
        temp_df.to_csv(filename, index=False)
        print(f"Saved {filename}")
    
    return 0

#================================================================================================================================