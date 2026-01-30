"""
Convert trajectory dataframe to step-by-step training JSON using TrigramStepAnalyzer
Changes from v3 - Changed input format to Road XYZ (lat, long) from (lat, long) (Road XYZ). Fixed 0 output.
"""
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Optional
import ast

class TrigramStepAnalyzer:
    """
    Analyzes route candidates based on bigram transitions.
    Provides road type, popularity, distance from destination, neighborhood, and trigram probability.
    """
    
    def __init__(self, connectivity_dict, arterial_highway_cells, 
                 hourly_busy_cells, neighborhoods_h3_cells, trigram_counts):
        """
        Initialize the analyzer with required data structures.
        
        Parameters:
        -----------
        connectivity_dict : dict
            Graph connectivity {token: [list of connected tokens]}
        arterial_highway_cells : pd.DataFrame
            DataFrame with columns: road_name, type, id_list
        hourly_busy_cells : pd.DataFrame
            DataFrame with columns: hour, neighborhood, busy_cells_list
        neighborhoods_h3_cells : pd.DataFrame
            DataFrame with columns: region, name, h3_cells
        trigram_counts : pd.DataFrame
            DataFrame with columns: id, bigram, cell1, cell2, cell_final, count, probability
        """
        self.connectivity_dict = connectivity_dict
        self.arterial_highway_cells = arterial_highway_cells
        self.hourly_busy_cells = hourly_busy_cells
        self.neighborhoods_h3_cells = neighborhoods_h3_cells
        self.trigram_counts = trigram_counts
        
        # Build graph for shortest path calculations
        self.graph = self._build_graph()
        
        # Build lookup dictionaries for faster access
        self._build_lookups()
        
    def _build_graph(self):
        """Build NetworkX graph from connectivity dictionary."""
        G = nx.Graph()
        for node, neighbors in self.connectivity_dict.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        return G
    
    def _build_lookups(self):
        """Build lookup dictionaries for faster data access."""
        # Road type lookup: road_id -> type
        self.road_type_lookup = {}
        for _, row in self.arterial_highway_cells.iterrows():
            road_type = row['type']
            id_list = row['id_list']
            # Parse id_list if it's a string
            if isinstance(id_list, str):
                id_list = ast.literal_eval(id_list)
            for road_id in id_list:
                self.road_type_lookup[road_id] = road_type
        
        # Neighborhood lookup: h3_id -> neighborhood name
        self.neighborhood_lookup = {}
        for _, row in self.neighborhoods_h3_cells.iterrows():
            neighborhood = row['name']
            h3_cells = row['h3_cells']
            # Parse h3_cells if it's a string
            if isinstance(h3_cells, str):
                h3_cells = ast.literal_eval(h3_cells)
            for h3_id in h3_cells:
                self.neighborhood_lookup[h3_id] = neighborhood
        
        # Busy cells lookup: hour -> set of busy h3_ids
        self.busy_cells_lookup = {}
        for hour in range(24):
            busy_cells = set()
            hour_data = self.hourly_busy_cells[self.hourly_busy_cells['hour'] == hour]
            for _, row in hour_data.iterrows():
                cells_list = row['busy_cells_list']
                # Parse if string
                if isinstance(cells_list, str):
                    cells_list = ast.literal_eval(cells_list)
                busy_cells.update(cells_list)
            self.busy_cells_lookup[hour] = busy_cells
    
    def split_token(self, token: str) -> Tuple[str, int]:
        """
        Split token into h3_id and road_id.
        
        Parameters:
        -----------
        token : str
            Token in format 'h3_id_road_id'
            
        Returns:
        --------
        tuple : (h3_id, road_id)
        """
        parts = token.rsplit('_', 1)
        if len(parts) == 2:
            h3_id = parts[0]
            road_id = int(parts[1])
            return h3_id, road_id
        return token, None
    
    def get_candidates(self, last_token: str) -> List[str]:
        """
        Get candidate next tokens from connectivity dictionary.
        
        Parameters:
        -----------
        last_token : str
            The last token in current route
            
        Returns:
        --------
        list : List of candidate tokens
        """
        return self.connectivity_dict.get(last_token, [])
    
    def get_road_id(self, token: str) -> Optional[int]:
        """
        Extract road ID from token.
        
        Parameters:
        -----------
        token : str
            Token to extract road ID from
            
        Returns:
        --------
        int or None : Road ID
        """
        _, road_id = self.split_token(token)
        return road_id
    
    def get_road_type(self, token: str) -> str:
        """
        Get road type (Highway, Arterial, or Normal).
        
        Parameters:
        -----------
        token : str
            Token to check
            
        Returns:
        --------
        str : 'Highway', 'Arterial', or 'Normal'
        """
        road_id = self.get_road_id(token)
        if road_id is None:
            return 'Normal'
        return self.road_type_lookup.get(road_id, 'Normal')
    
    def get_popularity(self, token: str, time: float) -> str:
        """
        Check if token is in busy cells for given time.
        
        Parameters:
        -----------
        token : str
            Token to check
        time : float
            Time of day (e.g., 4.5 for 4:30 AM)
            
        Returns:
        --------
        str : 'Yes' or 'No'
        """
        h3_id, _ = self.split_token(token)
        hour = int(time)  # Convert time to hour
        
        if hour in self.busy_cells_lookup:
            return 'Yes' if h3_id in self.busy_cells_lookup[hour] else 'No'
        return 'No'
    
    def get_distance_to_destination(self, token: str, destination_token: str) -> float:
        """
        Calculate shortest path distance to destination.
        
        Parameters:
        -----------
        token : str
            Starting token
        destination_token : str
            Destination token
            
        Returns:
        --------
        float : Path length, or float('inf') if no path exists
        """
        # Try with full tokens
        if nx.has_path(self.graph, token, destination_token):
            try:
                path_length = nx.shortest_path_length(self.graph, token, destination_token)
                return path_length
            except:
                pass
        
        # Try dropping road_id from token
        h3_token, _ = self.split_token(token)
        if nx.has_path(self.graph, h3_token, destination_token):
            try:
                path_length = nx.shortest_path_length(self.graph, h3_token, destination_token)
                return path_length
            except:
                pass
        
        # Try dropping road_id from destination
        h3_dest, _ = self.split_token(destination_token)
        if nx.has_path(self.graph, token, h3_dest):
            try:
                path_length = nx.shortest_path_length(self.graph, token, h3_dest)
                return path_length
            except:
                pass
        
        # Try dropping road_id from both
        if nx.has_path(self.graph, h3_token, h3_dest):
            try:
                path_length = nx.shortest_path_length(self.graph, h3_token, h3_dest)
                return path_length
            except:
                pass
        
        # No path exists
        return float('inf')
    
    def get_neighborhood(self, token: str) -> Optional[str]:
        """
        Get neighborhood name for token.
        
        Parameters:
        -----------
        token : str
            Token to look up
            
        Returns:
        --------
        str or None : Neighborhood name
        """
        h3_id, _ = self.split_token(token)
        return self.neighborhood_lookup.get(h3_id)
    
    def get_trigram_probability(self, cell1: str, cell2: str, cell_final: str) -> float:
        """
        Get trigram probability from database.
        
        Parameters:
        -----------
        cell1 : str
            First cell in bigram
        cell2 : str
            Second cell in bigram
        cell_final : str
            Candidate next cell
            
        Returns:
        --------
        float : Probability, or 0.0 if not found
        """
        # Query trigram database
        result = self.trigram_counts[
            (self.trigram_counts['cell1'] == cell1) &
            (self.trigram_counts['cell2'] == cell2) &
            (self.trigram_counts['cell_final'] == cell_final)
        ]
        
        if len(result) > 0:
            prob = result.iloc[0]['probability']
            return round(prob, 2)
        return 0.0
    
    def analyze_candidates(self, route: List[str], time: float, 
                          destination_token: str) -> pd.DataFrame:
        """
        Analyze all candidates for given route and time.
        
        Parameters:
        -----------
        route : list
            List of tokens representing current route (at least 2 tokens)
        time : float
            Current time (e.g., 4.5 for 4:30 AM)
        destination_token : str
            Final destination token
            
        Returns:
        --------
        pd.DataFrame : DataFrame with columns:
            - candidate: candidate token
            - road_id: road ID
            - type: road type (Highway/Arterial/Normal)
            - popularity: Yes/No
            - dist_from_dest: distance to destination
            - neighborhood: neighborhood name
            - trigram_prob: trigram probability
        """
        if len(route) < 2:
            raise ValueError("Route must have at least 2 tokens")
        
        # Get last token and candidates
        last_token = route[-1]
        candidates = self.get_candidates(last_token)
        
        # Prepare bigram for trigram lookup
        cell1 = route[-2]
        cell2 = route[-1]
        
        # Analyze each candidate
        results = []
        for candidate in candidates:
            result = {
                'candidate': candidate,
                'road_id': self.get_road_id(candidate),
                'type': self.get_road_type(candidate),
                'popular': self.get_popularity(candidate, time),
                'dist_from_dest': self.get_distance_to_destination(candidate, destination_token),
                'neighborhood': self.get_neighborhood(candidate),
                'trigram_prob': self.get_trigram_probability(cell1, cell2, candidate)
            }
            results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by trigram probability (descending) and distance (ascending)
        if len(df) > 0:
            df = df.sort_values(['trigram_prob', 'dist_from_dest'], 
                               ascending=[False, True]).reset_index(drop=True)
        
        return df
    
    def get_best_candidate(self, route: List[str], time: float, 
                          destination_token: str) -> Optional[str]:
        """
        Get the best candidate based on analysis.
        
        Returns the candidate with highest trigram probability,
        breaking ties with lowest distance to destination.
        
        Parameters:
        -----------
        route : list
            List of tokens representing current route
        time : float
            Current time
        destination_token : str
            Final destination token
            
        Returns:
        --------
        str or None : Best candidate token
        """
        df = self.analyze_candidates(route, time, destination_token)
        
        if len(df) > 0:
            return df.iloc[0]['candidate']
        return None

import pandas as pd
import json
import math
import ast
from typing import List, Dict, Tuple
import os

class TrigramJSONGen:
    """
    Converts trajectory dataframe to step-by-step training JSON.
    """
    
    def __init__(self, analyzer, connectivity_dict):
        """
        Initialize with TrigramStepAnalyzer and connectivity dictionary.
        
        Parameters:
        -----------
        analyzer : TrigramStepAnalyzer
            Pre-configured analyzer
        connectivity_dict : dict
            Graph connectivity {token: [list of connected tokens]}
        """
        self.analyzer = analyzer
        self.connectivity_dict = connectivity_dict
    
    def parse_traj_cells(self, traj_cells):
        """Parse trajectory cells from string or list."""
        if isinstance(traj_cells, str):
            return ast.literal_eval(traj_cells)
        return list(traj_cells)
    
    def calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing from point 1 to point 2 in degrees."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lon_diff = math.radians(lon2 - lon1)
        
        x = math.sin(lon_diff) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon_diff)
        
        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def bearing_to_direction(self, bearing: float) -> str:
        """Convert bearing to compass direction."""
        directions = ["North", "Northeast", "East", "Southeast",
                     "South", "Southwest", "West", "Northwest"]
        index = int((bearing + 22.5) / 45) % 8
        return directions[index]
    
    def find_dir(self, coord_str1: str, coord_str2: str) -> str:
        """
        Find direction from coord1 to coord2.
        
        Parameters:
        -----------
        coord_str1 : str
            Format: "Road 2515 (49.8647, -119.3543)"
        coord_str2 : str
            Format: "Road 2515 (49.8647, -119.3543)"
            
        Returns:
        --------
        str : Direction (e.g., "North", "Southeast")
        """
        # Extract coordinates from strings
        import re
        pattern = r'\(([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\)'
        
        match1 = re.search(pattern, coord_str1)
        match2 = re.search(pattern, coord_str2)
        
        if match1 and match2:
            lat1, lon1 = float(match1.group(1)), float(match1.group(2))
            lat2, lon2 = float(match2.group(1)), float(match2.group(2))
            
            bearing = self.calculate_bearing(lat1, lon1, lat2, lon2)
            return self.bearing_to_direction(bearing)
        
        return "Unknown"
    
    def weekend_or_weekday(self, day: str) -> str:
        """Determine if day is Weekend or Weekday."""
        try:
            date = pd.to_datetime(day)
            return "Weekday" if date.weekday() < 5 else "Weekend"
        except:
            return "Weekday"
    
    def calculate_road_percentages(self, route: List[str]) -> Tuple[float, float]:
        """Calculate highway and arterial percentages."""
        if len(route) == 0:
            return 0.0, 0.0
        
        highway_count = 0
        arterial_count = 0
        
        for token in route:
            road_type = self.analyzer.get_road_type(token)
            if road_type == 'Highway':
                highway_count += 1
            elif road_type == 'Arterial':
                arterial_count += 1
        
        total = len(route)
        highway_pct = round(highway_count / total, 2)
        arterial_pct = round(arterial_count / total, 2)
        
        return highway_pct, arterial_pct
    
    def convert_trajectory_to_examples(self, row: pd.Series) -> List[Dict]:
        """Convert one trajectory into multiple training examples."""
        # Parse trajectory cells    
        traj_cells = self.parse_traj_cells(row['traj_cells'])
        
        # FIX: traj_cells_converted might already be a list
        traj_cells_converted = row['traj_cells_converted']
        if isinstance(traj_cells_converted, str):
            traj_cells_converted = eval(traj_cells_converted)

        if len(traj_cells) < 3:
            return []
        
        # Extract constants (A-I)
        A = row['origin_h3']
        B = traj_cells_converted[0]
        C = row['origin_type']
        D = row['dest_h3']
        E = traj_cells_converted[-1]
        F = row['dest_type']
        G = self.find_dir(B, E)
        H = int(row['start_time'])
        I = self.weekend_or_weekday(row['day'])
        
        examples = []
        
        # Start from i=1 (need at least 2 cells for context)
        for i in range(1, len(traj_cells) - 1):
            try:  # ADD THIS TRY-EXCEPT BLOCK
                # Variables that change with i
                J = traj_cells_converted[i-1]
                K = traj_cells_converted[i]
                L = i + 1  # Points travelled (1-indexed for display)
                
                # Get neighborhood and road percentages
                current_token = traj_cells[i]
                M = self.analyzer.get_neighborhood(current_token) or "Unknown"
                N, O = self.calculate_road_percentages(traj_cells[:i+1])
                
                # Get candidates
                candidate_cells = self.connectivity_dict.get(current_token, [])
                
                if len(candidate_cells) == 0:
                    continue
                
                # Find the actual next cell
                actual_next = traj_cells[i+1]
                
                # Check if actual next is in candidates
                if actual_next not in candidate_cells:
                    continue  # Skip this example
                
                # Build options
                options = []
                output_idx = None
                
                for k, candidate in enumerate(candidate_cells):
                    # Convert candidate to lat/lon format
                    P_k = self.convert_cell_to_coord(candidate)
                    
                    # Build Q[k] information
                    direction = self.find_dir(K, P_k)
                    road_type = self.analyzer.get_road_type(candidate)
                    popular = self.analyzer.get_popularity(candidate, row['start_time'] + i * 0.01)
                    dist = self.analyzer.get_distance_to_destination(candidate, traj_cells[-1])
                    neighborhood = self.analyzer.get_neighborhood(candidate) or "Unknown"
                    
                    # Get trigram probability
                    cell1 = traj_cells[i-1]
                    cell2 = traj_cells[i]
                    trigram_prob = self.analyzer.get_trigram_probability(cell1, cell2, candidate)
                    
                    # Format distance
                    dist_str = str(int(dist)) if dist != float('inf') else "unreachable"
                    
                    option_text = (
                        f"{k}. {P_k} (Direction: {direction}; Type: {road_type}; "
                        f"Popular at this time: {popular}; Dist. from destination: {dist_str} cells; "
                        f"Neighborhood: {neighborhood}; Transition Probability: {trigram_prob:.2f});"
                    )
                    options.append(option_text)
                    
                    # Check if this is the correct answer
                    if candidate == actual_next:
                        output_idx = k
                
                # Build input text
                input_text = (
                    f"ROUTE CONTEXT:\n"
                    f" Origin - {A}, {B};\n"
                    f" Origin Type - {C};\n"
                    f" Destination - {D}, {E};\n"
                    f" Destination Type - {F};\n"
                    f" Destination Direction - {G};\n"
                    f" Origin Time - {H};\n"
                    f" Day - {I};\n\n"
                    f" PATH CONTEXT:\n"
                    f" Last two points - {J}, {K};\n"
                    f" Points travelled - {L};\n"
                    f" Current Neighborhood - {M};\n"
                    f" Road Percentage - Highway {N}, Arterial {O};\n\n"
                    f" NEXT OPTIONS:\n"
                    f" " + "\n ".join(options) + "\n\n"
                    f" QUESTION: Which next point is most plausible, given the context?"
                )
                
                examples.append({
                    'input': input_text,
                    'output': str(output_idx)
                })
            
            except Exception as e:
                # Skip this example if any error occurs (including NodeNotFound)
                continue  # ADD THIS EXCEPTION HANDLER
        
        return examples
    
    def convert_cell_to_coord(self, cell_token: str) -> str:
        """
        Convert cell token to coordinate string.
        
        Parameters:
        -----------
        cell_token : str
            Format: "8a12d1610087fff_4675"
            
        Returns:
        --------
        str : Format: "Road 4675 (49.8647, -119.3543)"
        """
        h3_id, road_id = self.analyzer.split_token(cell_token)
        
        # Get lat/lon from h3_id
        try:
            import h3
            lat, lon = h3.cell_to_latlng(h3_id)
            return f"Road {road_id} ({lat:.4f}, {lon:.4f})"
        except:
            return f"Road {road_id} (0.0000, 0.0000)"
    
    def convert_dataframe_to_json(self,
                                  df: pd.DataFrame,
                                  output_dir: str = "./route_training_data",
                                  train_y_ids: List[int] = [0, 1, 3, 5],
                                  test_y_ids: List[int] = [2, 4],
                                  train_ratio: float = 0.8,
                                  max_trajectories: int = None) -> Dict[str, List[Dict]]:
        """Convert dataframe to train/eval/test JSON files."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*70)
        print("CONVERTING TRAJECTORIES TO TRAINING FORMAT")
        print("="*70)
        
        # Split by y_id
        print(f"\n📊 Splitting by y_id...")
        print(f"   Train y_ids: {train_y_ids}")
        print(f"   Test y_ids: {test_y_ids}")
        
        df_train_full = df[df['y_id'].isin(train_y_ids)].copy()
        df_test = df[df['y_id'].isin(test_y_ids)].copy()
        
        train_split_idx = int(len(df_train_full) * train_ratio)
        df_train = df_train_full[:train_split_idx]
        df_eval = df_train_full[train_split_idx:]
        
        print(f"\n📈 Trajectory split:")
        print(f"   Train: {len(df_train)} trajectories")
        print(f"   Eval: {len(df_eval)} trajectories")
        print(f"   Test: {len(df_test)} trajectories")
        
        splits = {
            "train": df_train,
            "eval": df_eval,
            "test": df_test
        }
        
        results = {}
        
        for split_name, split_df in splits.items():
            print(f"\n{'='*70}")
            print(f"Processing {split_name.upper()} set...")
            print(f"{'='*70}")
            
            all_examples = []
            traj_count = 0
            
            for idx, row in split_df.iterrows():
                if max_trajectories and traj_count >= max_trajectories:
                    break
                
                examples = self.convert_trajectory_to_examples(row)
                
                if len(examples) > 0:
                    all_examples.extend(examples)
                    traj_count += 1
                
                if traj_count % 50 == 0 and traj_count > 0:
                    print(f"  Processed {traj_count} trajectories → {len(all_examples)} examples...")
            
            print(f"\n✓ Converted {traj_count} trajectories → {len(all_examples)} step examples")
            
            output_file = os.path.join(output_dir, f"{split_name}.json")
            print(f"💾 Saving to {output_file}...")
            
            with open(output_file, 'w') as f:
                json.dump(all_examples, f, indent=2)
            
            print(f"✓ Saved {split_name}.json")
            
            results[split_name] = all_examples
        
        print(f"\n{'='*70}")
        print("✅ CONVERSION COMPLETE")
        print(f"{'='*70}")
        print(f"\nFiles saved to {output_dir}/:")
        print(f"  - train.json ({len(results['train'])} step examples)")
        print(f"  - eval.json ({len(results['eval'])} step examples)")
        print(f"  - test.json ({len(results['test'])} step examples)")
        
        return results



# Read CSV and parse string as list
###################################

import pandas as pd
import ast

# Save the result
trajs_with_neighborhoods_fin = pd.read_csv('/media/tim/data/outputs/simu_2025_05/trajs_with_neighborhoods_fin.csv')

# Convert string to list
trajs_with_neighborhoods_fin["traj_cells"] = (
    trajs_with_neighborhoods_fin["traj_cells"]
    .apply(ast.literal_eval)
)

# Finding candidates of a cell from the graph network
#####################################################

def load_connectivity_graph(filename='kelowna_h3_connectivity.txt'):
    """
    Load the connectivity graph from a text file into a dictionary.

    Parameters:
    - filename: Path to the connectivity graph text file

    Returns:
    - connectivity_dict: Dictionary mapping each H3 cell_roadID to its connected neighbors
    """
    connectivity_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            # Split on the first colon to separate key from values
            if ':' in line:
                key, values = line.strip().split(':', 1)
                key = key.strip()

                # Parse the comma-separated neighbor values
                if values.strip():
                    neighbors = [v.strip() for v in values.split(',') if v.strip()]
                else:
                    neighbors = []

                connectivity_dict[key] = neighbors

    print(f"Loaded connectivity graph with {len(connectivity_dict)} nodes")
    return connectivity_dict

# Usage:
connectivity_dict =  load_connectivity_graph('/media/tim/data/outputs/tpat_2025_05/kelowna_h3_connectivity.txt')

# Load highway and arterial roads
#################################

import pandas as pd
import ast

# Define paths to road cells files
arterial_highway_cells = pd.read_csv('/media/tim/data/outputs/rout_2025_08/arterial_highway_cells.csv')

# Convert string to list
arterial_highway_cells["id_list"] = (arterial_highway_cells["id_list"].apply(ast.literal_eval))

# Load neighborhood-h3 cells
############################

import pandas as pd


neighborhoods_h3_cells = pd.read_csv('/media/tim/data/outputs/tpat_2025_04/neighborhoods_h3_cells.csv')

# Convert string to list
neighborhoods_h3_cells["h3_cells"] = (neighborhoods_h3_cells["h3_cells"].apply(ast.literal_eval))

# Load congested cell information
#################################

import pandas as pd

# Define path to congested cells file
hourly_busy_cells_combined = pd.read_csv('/media/tim/data/outputs/rout_2025_08/hourly_busy_cells_combined.csv')

# Load trigram information
##########################

import pandas as pd

trigram_counts = pd.read_csv('/media/tim/data/outputs/traj_2025_07/trigram_counts.csv')

trajs_demo = trajs_with_neighborhoods_fin[trajs_with_neighborhoods_fin['origin_h3']=='West Kelowna Estates / Rose Valley']
trajs_demo = trajs_demo[trajs_demo['dest_h3']!='Unknown']

# Convert the h3 cell to lat/long value
#######################################

import pandas as pd
import h3
import ast

# Extract the H3 part (before underscore) and convert to latlong
def h3_to_coordinates(h3_id):
    h3_cell = h3_id.split('_')[0]
    lat, lng = h3.cell_to_latlng(h3_cell)
    return (lat, lng)

def extract_road_id(h3_road_id):
    road_id = h3_road_id.split('_')[-1]
    return int(road_id)

# Convert traj_cells string to list of ((lat, long), road: X) format.
def convert_traj_cells(traj_cells_str):
    if isinstance(traj_cells_str, str):
        cells_list = ast.literal_eval(traj_cells_str)
    else:
        cells_list = traj_cells_str

    converted_points = []

    for h3_road_id in cells_list:
        # Get coordinates
        lat, lng = h3_to_coordinates(h3_road_id)

        # Get road ID
        road_id = extract_road_id(h3_road_id)

        if lat is not None and lng is not None and road_id is not None:
            # Format with 4 decimal places for coordinates
            ## point_str = f"({lat:.4f}, {lng:.4f}) (road {road_id})"

            # Another format
            point_str = f"'Road {road_id} ({lat:.4f}, {lng:.4f})'"

            converted_points.append(point_str)

    # Return as a string representation of list
    return f"[{', '.join(converted_points)}]"

# Convert the h3 cell column of dataframe into lat/long value
#############################################################

import pandas as pd
import h3
import ast

def process_dataframe(df):

    # Apply conversion to traj_cells column
    print("Converting traj_cells column...")
    df['traj_cells_converted'] = df['traj_cells'].apply(convert_traj_cells)

    # Print some examples
    print("\nExamples of converted trajectories:")
    print("-" * 80)

    for idx, row in df.head(5).iterrows():
        original = str(row['traj_cells'])[:100] + "..." if len(str(row['traj_cells'])) > 100 else str(row['traj_cells'])
        converted = row['traj_cells_converted']

        print(f"Row {idx}:")
        print(f"  Original: {original}")
        print(f"  Converted: {converted[:150]}..." if len(converted) > 150 else f"  Converted: {converted}")
        print()

    return df

# Main execution
trajs_demo_v2 = process_dataframe(trajs_demo)

# Search for any string
#######################

def search_token_in_trajectories(df, token):
    """
    Search for token in traj_cells_converted column and display results.

    Args:
        df: DataFrame
        token: String to search for
    """
    # Filter rows containing the token
    for i in range(1, len(df)):
        traj = str(df.iloc[i]['traj_cells_converted'])
        if token in traj:
            print(f"Row {i}: FOUND - {traj[:100]}...")
            print(f"day: {df.iloc[i]['day']}")
            print(f"trip_id: {df.iloc[i]['trip_id']}")
            print(f"Full trajectory: {df.iloc[i]['traj_cells_converted']}")
            print("-" * 80)

    #return results

# Reload TrigramJSONGen file
import importlib
import sys

# Reload it
### importlib.reload(sys.modules['TrigramJSONGen'])

# Auxiliary functions to generate JSON file
###########################################

# import TrigramStepAnalyzer
# import TrigramJSONGen

analyzer = TrigramStepAnalyzer(
     connectivity_dict=connectivity_dict,
     arterial_highway_cells=arterial_highway_cells,
     hourly_busy_cells=hourly_busy_cells_combined,
     neighborhoods_h3_cells=neighborhoods_h3_cells,
     trigram_counts=trigram_counts
)

# Initialize
generator = TrigramJSONGen(analyzer, connectivity_dict)

# Convert
results = generator.convert_dataframe_to_json(
    trajs_demo_v2,
    output_dir="./route_data",
    train_y_ids=[0, 1, 3, 5],
    test_y_ids=[2, 4]
)
