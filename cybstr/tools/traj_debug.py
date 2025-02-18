
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
from shapely.geometry import box

import folium
from scipy.spatial import KDTree
from matplotlib.colors import to_hex

#================================================================================================================================

def verify_if_osm_nodes_are_in_graph_v2(list_of_nodes, graph):
    """ 
    Checks if a list of OSM nodes are present in the original graph and in interpolated graph

    Parameters
    ----------
    list_of_nodes: int[] list
        List of nodes to check.
    graph : networkx.MultiDiGraph
        The road network graph.

    Returns
    -------
    node_coords: list
        List of coordinates of the nodes.
    """
    node_coords = []
    
    # Extract node coordinates from the graph
    for node_id in list_of_nodes:
        if node_id in graph.nodes:
            node_data = graph.nodes[node_id]
            node_coords.append((node_data["y"], node_data["x"]))  # Latitude, Longitude
            print(f"Node {node_id}: coordinate {node_data}.")
        else:
            print(f"Node {node_id} not found in the graph {graph}.")
    
    return node_coords

#================================================================================================================================

def display_osmnx_subgraphs_v1(graph):
    """ 
    Checks if a nx graph is fully connected and prints the different connected components in different colors

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The road network graph.

    Returns
    -------
    is_connected: nx.is_connected
        Boolean; true if graph is fully connected and false otherwise
    connected_components_sorted: list
        Sorted list of connected components
    plt: matplotlib graph
        Matplotlib map showing the nodes of graph components in diff colors
    """
    node_coords = []
    
    is_connected = nx.is_connected(graph.to_undirected())
    print(f"Is the graph connected? {is_connected}")

    if is_connected == False:
        # Find the largest connected component
        largest_cc = max(nx.connected_components(graph.to_undirected()), key=len)
        
        # Extract the subgraph corresponding to the largest connected component
        subgraph = graph.subgraph(largest_cc).copy()
        print(f"Largest connected component has {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges.")
        '''
        # Visualize the largest connected component
        fig, ax = ox.plot_graph(
            subgraph,
            node_size=5,
            edge_linewidth=0.5,
            bgcolor="black",
            show=True
        )
        '''
        # Convert the graph to an undirected graph
        undirected_graph = graph.to_undirected()
        
        # Find all connected components
        connected_components = list(nx.connected_components(undirected_graph))
        print('Number of connected components: ', len(connected_components))
        
        # Create a color map for the connected components
        colors = plt.cm.get_cmap("tab20", len(connected_components))  # Use a colormap with enough colors
        component_colors = [to_hex(colors(i)) for i in range(len(connected_components))]
        
        # Create a color dictionary for nodes
        node_color_map = {}
        for i, component in enumerate(connected_components):
            color = colors(i)
            for node in component:
                node_color_map[node] = color
        
        # Visualize the graph
        fig, ax = ox.plot_graph(
            graph,
            node_color=[node_color_map.get(node, (0, 0, 0, 0)) for node in graph.nodes],
            node_size=5,
            edge_linewidth=0.5,
            bgcolor="white",
            figsize=(15, 15),
            show=False,  # Prevent the graph from being displayed
            close=False,  # Keep the Matplotlib figure open
        )
        
        # Convert the graph to an undirected graph
        undirected_graph = graph.to_undirected()
        
        # Get all connected components (sorted by size, largest first)
        connected_components_sorted = sorted(nx.connected_components(graph.to_undirected()), key=len, reverse=True)   
        
        # Save the figure using Matplotlib
        output_path = "tempgen/graph_visualization_matplotlib.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")  # Adjust resolution and layout
        plt.close(fig)  # Close the figure after saving
        print('Graph saved in in dir tempgen current directory as osmnx_subgraphs.png')
    
    return is_connected, connected_components_sorted, plt
    
#================================================================================================================================

def check_if_nodes_are_in_bbox(graph, gdf):
    """ 
    Checks if a list of OSM nodes are present in the original graph and in interpolated graph

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The road network graph.
    gdf : gpd.GeoDataFrame
        Geodataframe containing the points to be evaluated

    Returns
    -------
    within_bounds: Boolean
        Contains row number and True/False based on whether the point lies within or outside the box
    """
    
    minx, miny, maxx, maxy = ox.graph_to_gdfs(graph, nodes=False).total_bounds
    print(f"Graph bounds: MinX: {minx}, MinY: {miny}, MaxX: {maxx}, MaxY: {maxy}")

    within_bounds = gdf.geometry.within(box(minx, miny, maxx, maxy))
    if not within_bounds.all():
        print("Some GPS points are outside the road network bounds.")
    else:
        print("All GPS points are inside the road network bounds.")
        
    return within_bounds

#================================================================================================================================

def check_which_point_in_which_component(graph, gdf):
    """ 
    Checks if a list of OSM nodes are present in the original graph and in interpolated graph

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The road network graph containing x and y fields.
    gdf : gpd.GeoDataFrame
        Geodataframe containing the points to be evaluated

    Returns
    -------
    within_bounds: Boolean
        Contains row number and True/False based on whether the point lies within or outside the box
    """
    gdfcopy = gdf.copy()
    # Extract node positions (x, y) from the graph
    node_positions = {
        node: (data["x"], data["y"])
        for node, data in graph.nodes(data=True)
        if "x" in data and "y" in data
    }
    
    # Build a KDTree for fast nearest neighbor search
    nodes, coords = zip(*node_positions.items())  # Separate node IDs and coordinates
    kdtree = KDTree(coords)
    
    # Identify connected components and create a mapping of node -> component ID
    connected_components = list(nx.connected_components(graph.to_undirected()))
    node_to_component = {
        node: component_id
        for component_id, component in enumerate(connected_components, start=1)
        for node in component
    }
    
    # Assign each point to the closest graph component
    def find_component(row):
        # Find the closest node in the graph
        _, idx = kdtree.query((row["x"], row["y"]))
        closest_node = nodes[idx]
        # Get the component ID for the closest node
        return node_to_component.get(closest_node, -1)  # -1 if no component is found
    
    # Apply the function to determine the component for each point
    gdfcopy.loc[:,"component_id"] = gdfcopy.apply(find_component, axis=1)
    
    # Print the result
    print(gdfcopy[["geometry", "component_id"]])

    return 0