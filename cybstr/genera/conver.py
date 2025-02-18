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
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from shapely.geometry import Point, LineString

#================================================================================================================================

def convert_roads_geojson_to_osmnx_graphml_v1(path, crs_no):
    """ 
    Convert roads geojson file into osmnx graphml.
    Especially useful for converting exported geojson from Overpass Turbo into OSMNX multidigraph.
    NOTE: Suitable only for EPSG 4326 for now.

    Parameters
    ----------
    path: string
        Path to the roads geojson file.
    crs_no: int.
        The CRS number to be converted to. 

    Returns
    -------
    roads_graphml: nx.MultiDiGraph()
        MultiDiGraph of road networks.
    """
    # Load the GeoJSON file
    roads_gdf = gpd.read_file(path)
    
    # Ensure CRS is WGS84 (EPSG:4326)
    roads_gdf = roads_gdf.to_crs(epsg=crs_no)
    
    # Initialize a directed graph
    roads_graphml = nx.MultiDiGraph()
    
    # Extract nodes and edges
    node_id = 0  # To uniquely identify nodes
    node_map = {}  # To store unique nodes and their IDs
    
    for _, row in roads_gdf.iterrows():
        if isinstance(row.geometry, LineString):
            # Extract coordinates from the LineString
            coords = list(row.geometry.coords)
            
            # Add nodes and edges to the graph
            for i in range(len(coords) - 1):
                u, v = coords[i], coords[i + 1]
                
                # Add nodes (check if they already exist)
                if u not in node_map:
                    node_map[u] = node_id
                    roads_graphml.add_node(node_id, x=u[0], y=u[1])
                    node_id += 1
                if v not in node_map:
                    node_map[v] = node_id
                    roads_graphml.add_node(node_id, x=v[0], y=v[1])
                    node_id += 1
                
                # Add edge with attributes
                roads_graphml.add_edge(node_map[u], node_map[v], length=LineString([u, v]).length)
        else:
            print(f"Skipping non-LineString geometry at index {_}")

    if crs_no == 4326:
        # Add CRS to the graph
        roads_graphml.graph['crs'] = "EPSG:4326"
    else:
        print('The crs_no is not 4326; skipping graph[crs] addition')

    return roads_graphml

#================================================================================================================================

def convert_graphml_to_osm_v1(gml, save_path_file):
    """ 
    Convert a dataframe to geodataframe; and creates a geometry column.

    Parameters
    ----------
    gml: nx.MultiDiGraph
        Graphml File to be converted to .osm
    save_path_file: string.
        Path to save the .osm file

    Returns
    -------
    0
    """
    # Convert the graph to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(gml)

    # Create OSM XML root
    osm_root = ET.Element("osm", attrib={"version": "0.6", "generator": "custom_graph_export"})

    # Add nodes to the OSM XML
    node_map = {}  # To map node IDs
    for idx, (node_id, row) in enumerate(nodes.iterrows()):
        node_map[Point(row["x"], row["y"])] = idx + 1  # Create OSM-compatible node IDs
        ET.SubElement(
            osm_root, 
            "node",
            attrib={
                "id": str(idx + 1),
                "lat": str(row["y"]),
                "lon": str(row["x"])
            }
        )

    # Add edges (ways) to the OSM XML
    for idx, row in edges.iterrows():
        # Handle tuple index from MultiDiGraph
        way_id = "_".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)

        way = ET.SubElement(
            osm_root,
            "way",
            attrib={"id": way_id}
        )

        # Add nodes to the way
        for coord in row["geometry"].coords:
            # Find the nearest node ID from the map
            point = Point(coord)
            node_id = node_map.get(point)
            if node_id:
                ET.SubElement(way, "nd", attrib={"ref": str(node_id)})

    # Write to .osm file
    tree = ET.ElementTree(osm_root)
    tree.write(save_path_file, encoding="utf-8", xml_declaration=True)
    return 0