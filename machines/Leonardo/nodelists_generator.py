"""
Leonardo Supercomputer Nodelist Generator
Generates optimized nodelists for SLURM job scheduling based on network topology constraints
"""

import pandas as pd
import re
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import itertools

PARTITION_NAME_MAP = {
    "1": "boost_usr_prod",
    "2": "dcgp_usr_prod",
}

class TopologyConstraint(Enum):
    """Network topology constraints for node selection"""
    SAME_RACK = "same_rack"              # All nodes in same rack
    SAME_CELL = "same_cell"              # All nodes in same cell
    SAME_SWITCH = "same_switch"          # All nodes on same L1 switch
    DIFFERENT_CELLS = "different_cells"  # Nodes spread across different cells
    DIFFERENT_RACKS = "different_racks"  # Nodes spread across different racks
    MINIMIZE_L2_HOPS = "minimize_l2"     # Minimize inter-cell (L2) communication
    NO_CONSTRAINT = "no_constraint"      # No topology constraint


@dataclass
class SlurmResources:
    """SLURM resource requirements"""
    num_nodes: int
    tasks_per_node: Optional[int] = None
    cpus_per_task: Optional[int] = None
    gpus_per_node: Optional[int] = None
    partition: Optional[int] = None  # 1=Booster, 2=DCGP
    
    def __post_init__(self):
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")


def load_leonardo_system_data(csv_path: str) -> pd.DataFrame:
        """
        Load Leonardo system data from CSV or create sample data.
        
        Args:
            csv_path: Path to CSV file with system topology
            
        Returns:
            DataFrame with system topology
        """
        print(f"Loading system data from {csv_path}...")
        pattern = re.compile(
            r"NODE\s+(\d+)\s+RACK\s+(\d+)\s+CELL\s+(\d+)\s+ROW\s+(\d+)\s+PARTITION\s+(\d+)\s+SWITCH\s+(\d+)"
        )
        
        records = []
        with open(csv_path) as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    records.append(match.groups())
        
        return pd.DataFrame(
            records,
            columns=["NODE", "RACK", "CELL", "ROW", "PARTITION", "SWITCH"]
        ).astype(int)


class LeonardoNodelistGenerator:
    """
    Generates optimized nodelists for Leonardo supercomputer based on
    network topology and SLURM resource requirements.
    """
    
    def __init__(self, system_df: Union[None,pd.DataFrame] = None, csv_path: Union[None,str] = None):
        """
        Initialize with system topology dataframe.
        
        Args:
            system_df: DataFrame with columns [NODE, RACK, CELL, ROW, PARTITION, SWITCH]
        """
        if system_df is not None:
            self.df = system_df.copy()
        elif csv_path:
            self.df = load_leonardo_system_data(csv_path)
        else:
            raise Exception('Must provide either system_df or csv_path')
        self._validate_dataframe()
        self._precompute_topology()
            
        
    def _validate_dataframe(self):
        """Validate the input dataframe has required columns"""
        required_cols = ["NODE", "RACK", "CELL", "ROW", "PARTITION", "SWITCH"]
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _precompute_topology(self):
        """Precompute topology mappings for efficient lookup"""
        # Group nodes by various topology levels
        self.nodes_by_cell = self.df.groupby('CELL')['NODE'].apply(list).to_dict()
        self.nodes_by_rack = self.df.groupby('RACK')['NODE'].apply(list).to_dict()
        self.nodes_by_switch = self.df.groupby('SWITCH')['NODE'].apply(list).to_dict()
        self.nodes_by_partition = self.df.groupby('PARTITION')['NODE'].apply(list).to_dict()
        
        # Create reverse mappings
        self.node_to_cell = self.df.set_index('NODE')['CELL'].to_dict()
        self.node_to_rack = self.df.set_index('NODE')['RACK'].to_dict()
        self.node_to_switch = self.df.set_index('NODE')['SWITCH'].to_dict()
        self.node_to_partition = self.df.set_index('NODE')['PARTITION'].to_dict()
        
        # Map cells to racks and switches
        self.cell_to_racks = self.df.groupby('CELL')['RACK'].apply(lambda x: list(x.unique())).to_dict()
        self.cell_to_switches = self.df.groupby('CELL')['SWITCH'].apply(lambda x: list(x.unique())).to_dict()
        
        # Determine nodes per switch (for Booster vs DCGP)
        self.switch_capacities = self.df.groupby('SWITCH')['NODE'].count().to_dict()
    
    def generate_nodelists(
        self,
        resources: SlurmResources,
        constraint: TopologyConstraint = TopologyConstraint.NO_CONSTRAINT,
        max_nodelists: int = 10,
        min_nodelists: int = 5
    ) -> List[List[int]]:
        """
        Generate nodelists that satisfy resource and topology constraints.
        
        Args:
            resources: SLURM resource requirements
            constraint: Network topology constraint
            max_nodelists: Maximum number of nodelists to generate
            min_nodelists: Minimum number of nodelists to generate
            
        Returns:
            List of nodelists, where each nodelist is a list of node IDs
        """
        # Filter by partition if specified
        available_nodes = self._filter_by_partition(resources.partition)
        
        # Generate candidate nodelists based on constraint
        if constraint == TopologyConstraint.SAME_CELL:
            candidates = self._generate_same_cell(available_nodes, resources.num_nodes)
        elif constraint == TopologyConstraint.SAME_RACK:
            candidates = self._generate_same_rack(available_nodes, resources.num_nodes)
        elif constraint == TopologyConstraint.SAME_SWITCH:
            candidates = self._generate_same_switch(available_nodes, resources.num_nodes)
        elif constraint == TopologyConstraint.DIFFERENT_CELLS:
            candidates = self._generate_different_cells(available_nodes, resources.num_nodes)
        elif constraint == TopologyConstraint.DIFFERENT_RACKS:
            candidates = self._generate_different_racks(available_nodes, resources.num_nodes)
        elif constraint == TopologyConstraint.MINIMIZE_L2_HOPS:
            candidates = self._generate_minimize_l2(available_nodes, resources.num_nodes)
        else:  # NO_CONSTRAINT
            candidates = self._generate_no_constraint(available_nodes, resources.num_nodes, max_nodelists)
        
        # Ensure we have enough nodelists
        nodelists = list(candidates)[:max_nodelists]
        
        if len(nodelists) < min_nodelists:
            print(f"Warning: Only {len(nodelists)} nodelists generated (requested min: {min_nodelists})")
        
        return nodelists
    
    def _filter_by_partition(self, partition: Optional[int]) -> Set[int]:
        """Filter available nodes by partition"""
        if partition is None:
            return set(self.df['NODE'].values)
        return set(self.nodes_by_partition.get(partition, []))
    
    def _generate_same_cell(self, available_nodes: Set[int], num_nodes: int) -> List[List[int]]:
        """Generate nodelists with all nodes in the same cell"""
        nodelists = []
        
        for cell, nodes in self.nodes_by_cell.items():
            cell_nodes = [n for n in nodes if n in available_nodes]
            if len(cell_nodes) >= num_nodes:
                # Generate multiple combinations from this cell
                for i in range(min(3, len(cell_nodes) - num_nodes + 1)):
                    nodelist = sorted(cell_nodes[i:i+num_nodes])
                    nodelists.append(nodelist)
        
        return nodelists
    
    def _generate_same_rack(self, available_nodes: Set[int], num_nodes: int) -> List[List[int]]:
        """Generate nodelists with all nodes in the same rack"""
        nodelists = []
        
        for rack, nodes in self.nodes_by_rack.items():
            rack_nodes = [n for n in nodes if n in available_nodes]
            if len(rack_nodes) >= num_nodes:
                nodelist = sorted(rack_nodes[:num_nodes])
                nodelists.append(nodelist)
        
        return nodelists
    
    def _generate_same_switch(self, available_nodes: Set[int], num_nodes: int) -> List[List[int]]:
        """Generate nodelists with all nodes on the same L1 switch"""
        nodelists = []
        
        for switch, nodes in self.nodes_by_switch.items():
            switch_nodes = [n for n in nodes if n in available_nodes]
            if len(switch_nodes) >= num_nodes:
                nodelist = sorted(switch_nodes[:num_nodes])
                nodelists.append(nodelist)
        
        return nodelists
    
    def _generate_different_cells(self, available_nodes: Set[int], num_nodes: int) -> List[List[int]]:
        """Generate nodelists with nodes spread across different cells"""
        nodelists = []
        
        # Get cells sorted by number of available nodes
        cells_with_nodes = []
        for cell, nodes in self.nodes_by_cell.items():
            cell_nodes = [n for n in nodes if n in available_nodes]
            if cell_nodes:
                cells_with_nodes.append((cell, cell_nodes))
        
        cells_with_nodes.sort(key=lambda x: len(x[1]), reverse=True)
        
        if len(cells_with_nodes) < 2:
            return []
        
        # Strategy: distribute nodes across cells as evenly as possible
        for start_idx in range(min(3, len(cells_with_nodes))):
            nodelist = []
            cells_to_use = cells_with_nodes[start_idx:]
            
            nodes_per_cell = num_nodes // len(cells_to_use)
            remainder = num_nodes % len(cells_to_use)
            
            for i, (cell, nodes) in enumerate(cells_to_use):
                n_from_cell = nodes_per_cell + (1 if i < remainder else 0)
                if len(nodes) >= n_from_cell:
                    nodelist.extend(nodes[:n_from_cell])
                else:
                    break
            
            if len(nodelist) == num_nodes:
                nodelists.append(sorted(nodelist))
        
        return nodelists
    
    def _generate_different_racks(self, available_nodes: Set[int], num_nodes: int) -> List[List[int]]:
        """Generate nodelists with nodes spread across different racks"""
        nodelists = []
        
        # Get racks sorted by number of available nodes
        racks_with_nodes = []
        for rack, nodes in self.nodes_by_rack.items():
            rack_nodes = [n for n in nodes if n in available_nodes]
            if rack_nodes:
                racks_with_nodes.append((rack, rack_nodes))
        
        racks_with_nodes.sort(key=lambda x: len(x[1]), reverse=True)
        
        if len(racks_with_nodes) < 2:
            return []
        
        # Distribute nodes across racks
        for start_idx in range(min(3, len(racks_with_nodes))):
            nodelist = []
            racks_to_use = racks_with_nodes[start_idx:]
            
            nodes_per_rack = max(1, num_nodes // len(racks_to_use))
            
            for rack, nodes in racks_to_use:
                nodelist.extend(nodes[:nodes_per_rack])
                if len(nodelist) >= num_nodes:
                    break
            
            if len(nodelist) >= num_nodes:
                nodelists.append(sorted(nodelist[:num_nodes]))
        
        return nodelists
    
    def _generate_minimize_l2(self, available_nodes: Set[int], num_nodes: int) -> List[List[int]]:
        """Generate nodelists minimizing inter-cell (L2) communication"""
        # Prioritize same-cell, then fall back to minimal cell usage
        same_cell = self._generate_same_cell(available_nodes, num_nodes)
        if same_cell:
            return same_cell
        
        # Find minimum number of cells needed
        nodelists = []
        cells_with_nodes = []
        for cell, nodes in self.nodes_by_cell.items():
            cell_nodes = [n for n in nodes if n in available_nodes]
            if cell_nodes:
                cells_with_nodes.append((cell, cell_nodes))
        
        cells_with_nodes.sort(key=lambda x: len(x[1]), reverse=True)
        
        # Try to use as few cells as possible
        for num_cells in range(2, min(5, len(cells_with_nodes) + 1)):
            for cell_combo in itertools.combinations(cells_with_nodes, num_cells):
                nodelist = []
                for cell, nodes in cell_combo:
                    nodelist.extend(nodes)
                    if len(nodelist) >= num_nodes:
                        break
                
                if len(nodelist) >= num_nodes:
                    nodelists.append(sorted(nodelist[:num_nodes]))
                    if len(nodelists) >= 5:
                        return nodelists
        
        return nodelists
    
    def _generate_no_constraint(self, available_nodes: Set[int], num_nodes: int, max_lists: int) -> List[List[int]]:
        """Generate nodelists without topology constraints"""
        nodes_list = sorted(list(available_nodes))
        
        if len(nodes_list) < num_nodes:
            return []
        
        nodelists = []
        
        # Generate diverse nodelists by starting at different positions
        step = max(1, (len(nodes_list) - num_nodes) // max_lists)
        
        for i in range(0, min(len(nodes_list) - num_nodes + 1, max_lists * step), step):
            nodelist = nodes_list[i:i+num_nodes]
            nodelists.append(nodelist)
        
        return nodelists
    
    def format_nodelist_for_slurm(self, nodelist: List[int]) -> str:
        """
        Format nodelist for SLURM using compact range notation with 'lrdn' prefix
        and 4-digit zero-padded node IDs.
        
        Args:
            nodelist: List of node IDs
            
        Returns:
            SLURM-compatible nodelist string (e.g., "lrdn[0001-0010,0015,0020-0025]")
        """
        if not nodelist:
            return ""
        
        sorted_nodes = sorted(nodelist)
        ranges = []
        start = sorted_nodes[0]
        end = sorted_nodes[0]
        
        for node in sorted_nodes[1:]:
            if node == end + 1:
                end = node
            else:
                if start == end:
                    ranges.append(f"{start:04d}")
                else:
                    ranges.append(f"{start:04d}-{end:04d}")
                start = node
                end = node
        
        # Add final range
        if start == end:
            ranges.append(f"{start:04d}")
        else:
            ranges.append(f"{start:04d}-{end:04d}")
        
        return f"lrdn[{','.join(ranges)}]"
    
    def analyze_nodelist_topology(self, nodelist: List[int]) -> Dict:
        """
        Analyze the topology characteristics of a nodelist.
        
        Args:
            nodelist: List of node IDs
            
        Returns:
            Dictionary with topology statistics
        """
        cells = set(self.node_to_cell[n] for n in nodelist)
        racks = set(self.node_to_rack[n] for n in nodelist)
        switches = set(self.node_to_switch[n] for n in nodelist)
        partitions = set(self.node_to_partition[n] for n in nodelist)
        
        return {
            'num_nodes': len(nodelist),
            'num_cells': len(cells),
            'num_racks': len(racks),
            'num_switches': len(switches),
            'cells': sorted(list(cells)),
            'racks': sorted(list(racks)),
            'partitions': sorted(list(partitions)),
            'uses_l2': len(cells) > 1,
            'topology_type': self._classify_topology(len(cells), len(racks), len(switches))
        }
    
    def _classify_topology(self, num_cells: int, num_racks: int, num_switches: int) -> str:
        """Classify the topology of a nodelist"""
        if num_switches == 1:
            return "same_switch"
        elif num_racks == 1:
            return "same_rack"
        elif num_cells == 1:
            return "same_cell"
        else:
            return "multi_cell"


    def rank_nodelists_by_availability(
        self,
        nodelists: List[List[int]],
        partition: Optional[str] = None,
        consider_pending: bool = True,
        time_weight: float = 0.3
    ) -> List[Tuple[List[int], float, Dict]]:
        """
        Rank nodelists by their likelihood of being scheduled quickly based on current queue state.
        
        Args:
            nodelists: List of candidate nodelists to rank
            partition: Partition to query (None = all partitions)
            consider_pending: Whether to consider pending jobs in scoring
            time_weight: Weight for time-based scoring (0-1). Higher = prioritize jobs ending sooner
            
        Returns:
            List of tuples (nodelist, score, details) sorted by score (higher = better availability)
        """
        # Get queue state
        queue_state = self._query_slurm_queue(partition)
        
        # Parse and build node usage maps
        node_usage = self._build_node_usage_map(queue_state, consider_pending)
        
        # Score each nodelist
        scored_nodelists = []
        for nodelist in nodelists:
            score, details = self._score_nodelist_availability(nodelist, node_usage, time_weight)
            scored_nodelists.append((nodelist, score, details))
        
        # Sort by score (descending - higher is better)
        scored_nodelists.sort(key=lambda x: x[1], reverse=True)
        
        return scored_nodelists
    
    def _query_slurm_queue(self, partition: Optional[str] = None) -> List[Dict]:
        """
        Query SLURM queue and parse results.
        
        Args:
            partition: Partition to query (None = all partitions)
            
        Returns:
            List of job dictionaries with parsed information
        """
        import subprocess
        import shlex
        
        # Define format string with relevant fields
        # %D=nodes, %N=nodelist, %T=state, %L=time_left, %e=end_time, %S=start_time, %P=partition
        format_str = "%i|%D|%N|%T|%L|%e|%S|%P|%Q|%r"
        
        # Build squeue command
        cmd = ["squeue", "--noheader", "-o", format_str]
        if partition:
            cmd.extend(["-p", PARTITION_NAME_MAP[str(partition)]])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Warning: squeue command failed: {e}")
            return []
        except FileNotFoundError:
            print("Warning: squeue command not found. Returning empty queue state.")
            return []
        
        # Parse output
        jobs = []
        for line in output.split('\n'):
            if not line.strip():
                continue
            
            parts = line.split('|')
            if len(parts) < 10:
                continue
            
            job = {
                'job_id': parts[0].strip(),
                'num_nodes': self._parse_int_safe(parts[1]),
                'nodelist': parts[2].strip(),
                'state': parts[3].strip(),
                'time_left': parts[4].strip(),
                'end_time': parts[5].strip(),
                'start_time': parts[6].strip(),
                'partition': parts[7].strip(),
                'priority': self._parse_int_safe(parts[8]),
                'reason': parts[9].strip()
            }
            
            # Parse nodelist into individual nodes
            job['nodes'] = self._expand_slurm_nodelist(job['nodelist'])
            
            # Parse time left into seconds
            job['time_left_seconds'] = self._parse_time_to_seconds(job['time_left'])
            
            jobs.append(job)
        
        return jobs
    
    def _parse_int_safe(self, value: str, default: int = 0) -> int:
        """Safely parse integer from string"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def _expand_slurm_nodelist(self, nodelist_str: str) -> List[int]:
        """
        Expand SLURM nodelist notation into individual node IDs.
        
        Examples:
            "node[0001-0010]" -> [1, 2, 3, ..., 10]
            "node[0001,0005,0010-0012]" -> [1, 5, 10, 11, 12]
        """
        import re
        
        if not nodelist_str or nodelist_str in ['None', 'N/A', '']:
            return []
        
        nodes = []
        
        # Match patterns like node[0001-0010,0015,0020-0025]
        match = re.search(r'\[([^\]]+)\]', nodelist_str)
        if not match:
            # Try to extract single node number
            numbers = re.findall(r'\d+', nodelist_str)
            if numbers:
                return [int(numbers[-1])]  # Take last number as node ID
            return []
        
        ranges_str = match.group(1)
        
        for part in ranges_str.split(','):
            part = part.strip()
            if '-' in part:
                # Range: 0001-0010
                start, end = part.split('-')
                start_num = int(start)
                end_num = int(end)
                nodes.extend(range(start_num, end_num + 1))
            else:
                # Single node: 0001
                nodes.append(int(part))
        
        return nodes
    
    def _parse_time_to_seconds(self, time_str: str) -> Optional[float]:
        """
        Parse SLURM time format to seconds.
        
        Formats: "days-hours:minutes:seconds", "hours:minutes:seconds", "minutes:seconds"
        Special: "UNLIMITED", "NOT_SET", "INVALID"
        """
        if not time_str or time_str in ['UNLIMITED', 'NOT_SET', 'INVALID', 'N/A']:
            return None
        
        try:
            # Handle days
            if '-' in time_str:
                days_str, time_str = time_str.split('-')
                days = int(days_str)
            else:
                days = 0
            
            # Split time components
            parts = time_str.split(':')
            
            if len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
            elif len(parts) == 2:
                hours = 0
                minutes, seconds = map(int, parts)
            elif len(parts) == 1:
                hours = 0
                minutes = 0
                seconds = int(parts[0])
            else:
                return None
            
            total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
            return float(total_seconds)
        
        except (ValueError, AttributeError):
            return None
    
    def _build_node_usage_map(self, jobs: List[Dict], consider_pending: bool) -> Dict[int, Dict]:
        """
        Build a map of node usage from queue state.
        
        Args:
            jobs: List of parsed job dictionaries
            consider_pending: Whether to include pending jobs
            
        Returns:
            Dictionary mapping node_id -> usage information
        """
        node_usage = {}
        
        for job in jobs:
            state = job['state']
            
            # Skip completed/failed jobs
            if state in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL']:
                continue
            
            # Skip pending jobs if not considering them
            if state == 'PENDING' and not consider_pending:
                continue
            
            for node in job['nodes']:
                if node not in node_usage:
                    node_usage[node] = {
                        'running_jobs': [],
                        'pending_jobs': [],
                        'earliest_free_time': None,
                        'total_priority_pending': 0
                    }
                
                if state == 'RUNNING':
                    node_usage[node]['running_jobs'].append(job)
                    
                    # Update earliest free time
                    if job['time_left_seconds'] is not None:
                        free_time = job['time_left_seconds']
                        if (node_usage[node]['earliest_free_time'] is None or 
                            free_time < node_usage[node]['earliest_free_time']):
                            node_usage[node]['earliest_free_time'] = free_time
                
                elif state == 'PENDING':
                    node_usage[node]['pending_jobs'].append(job)
                    node_usage[node]['total_priority_pending'] += job.get('priority', 0)
        
        return node_usage
    
    def _score_nodelist_availability(
        self,
        nodelist: List[int],
        node_usage: Dict[int, Dict],
        time_weight: float
    ) -> Tuple[float, Dict]:
        """
        Score a nodelist based on current node availability.
        
        Higher score = better availability (more likely to be scheduled soon)
        
        Args:
            nodelist: List of node IDs to score
            node_usage: Node usage map from _build_node_usage_map
            time_weight: Weight for time-based scoring (0-1)
            
        Returns:
            Tuple of (score, details_dict)
        """
        num_nodes = len(nodelist)
        
        # Initialize counters
        free_nodes = 0
        running_nodes = 0
        pending_nodes = 0
        total_time_to_free = 0
        max_time_to_free = 0
        total_pending_priority = 0
        
        for node in nodelist:
            if node not in node_usage:
                # Node is completely free
                free_nodes += 1
            else:
                usage = node_usage[node]
                
                if usage['running_jobs']:
                    running_nodes += 1
                    if usage['earliest_free_time'] is not None:
                        total_time_to_free += usage['earliest_free_time']
                        max_time_to_free = max(max_time_to_free, usage['earliest_free_time'])
                    else:
                        # Unknown end time - penalize heavily
                        max_time_to_free = float('inf')
                        total_time_to_free += 1e9
                
                if usage['pending_jobs']:
                    pending_nodes += 1
                    total_pending_priority += usage['total_priority_pending']
        
        # Calculate component scores (normalized 0-1)
        
        # 1. Free nodes score (higher = more free nodes)
        free_score = free_nodes / num_nodes
        
        # 2. Running nodes penalty (lower = fewer running nodes)
        running_penalty = running_nodes / num_nodes
        
        # 3. Time-based score (exponential decay - jobs ending soon are better)
        if max_time_to_free == float('inf') or max_time_to_free > 86400 * 7:  # > 7 days
            time_score = 0.0
        elif max_time_to_free == 0:
            time_score = 1.0
        else:
            # Exponential decay: jobs ending in 1 hour = 0.9, 1 day = 0.5, 3 days = 0.1
            decay_rate = 1.0 / 86400  # 1 day half-life
            time_score = np.exp(-decay_rate * max_time_to_free)
        
        # 4. Pending jobs penalty (lower = fewer/lower priority pending jobs)
        if pending_nodes == 0:
            pending_score = 1.0
        else:
            # Normalize by assuming max priority per node is 1e6
            avg_priority = total_pending_priority / num_nodes
            pending_score = max(0, 1.0 - (avg_priority / 1e6))
        
        # 5. Contiguity bonus - nodes in same cell/rack are better for scheduling
        topology = self.analyze_nodelist_topology(nodelist)
        if topology['num_cells'] == 1:
            contiguity_bonus = 0.2
        elif topology['num_racks'] <= 3:
            contiguity_bonus = 0.1
        else:
            contiguity_bonus = 0.0
        
        # Weighted combination
        base_weight = 1.0 - time_weight
        score = (
            free_score * 0.4 * base_weight +           # 40% weight on free nodes
            (1 - running_penalty) * 0.3 * base_weight + # 30% weight on not running
            time_score * time_weight +                  # Variable weight on time
            pending_score * 0.2 * base_weight +         # 20% weight on pending
            contiguity_bonus * 0.1                      # 10% bonus for topology
        )
        
        # Details for debugging/reporting
        details = {
            'free_nodes': free_nodes,
            'running_nodes': running_nodes,
            'pending_nodes': pending_nodes,
            'free_percentage': free_nodes / num_nodes * 100,
            'max_time_to_free_hours': max_time_to_free / 3600 if max_time_to_free != float('inf') else None,
            'avg_time_to_free_hours': (total_time_to_free / num_nodes) / 3600 if total_time_to_free < 1e8 else None,
            'total_pending_priority': total_pending_priority,
            'contiguity': topology['topology_type'],
            'score_components': {
                'free_score': free_score,
                'running_penalty': running_penalty,
                'time_score': time_score,
                'pending_score': pending_score,
                'contiguity_bonus': contiguity_bonus
            }
        }
        
        return score, details

def parse_topology_file(path: str) -> pd.DataFrame:
    pattern = re.compile(
        r"NODE\s+(\d+)\s+RACK\s+(\d+)\s+CELL\s+(\d+)\s+ROW\s+(\d+)\s+PARTITION\s+(\d+)\s+SWITCH\s+(\d+)"
    )
    
    records = []
    with open(path) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                records.append(match.groups())
    
    df = pd.DataFrame(
        records,
        columns=["NODE", "RACK", "CELL", "ROW", "PARTITION", "SWITCH"]
    ).astype(int)
    
    return df

def example_usage():
    """Example usage of the Leonardo nodelist generator"""
    
    # Load your system dataframe
    # df = pd.read_csv('leonardo_system.csv')
    # For demonstration, create a small sample
    # df = pd.DataFrame({
    #     'NODE': list(range(1, 181)),
    #     'RACK': [i//30 + 1 for i in range(180)],
    #     'CELL': [1] * 180,
    #     'ROW': [1] * 180,
    #     'PARTITION': [1] * 180,
    #     'SWITCH': [10100 + (i//10)*2 for i in range(180)]
    # })
    df = parse_topology_file("leo_map.txt")
    
    # Initialize generator
    generator = LeonardoNodelistGenerator(df)
    
    # Define resource requirements
    resources = SlurmResources(
        num_nodes=32,
        tasks_per_node=4,
        cpus_per_task=8,
        gpus_per_node=4,
        partition=1
    )
    
    # Generate nodelists with different constraints
    print("=== Generating Candidate Nodelists ===")
    nodelists = generator.generate_nodelists(
        resources,
        constraint=TopologyConstraint.SAME_CELL,
        max_nodelists=10
    )
    
    print(f"Generated {len(nodelists)} candidate nodelists")
    print(nodelists)
    
    # Rank by availability based on current queue
    print("\n=== Ranking Nodelists by Availability ===")
    ranked = generator.rank_nodelists_by_availability(
        nodelists,
        partition="boost_usr_prod",  # Adjust to your partition name
        consider_pending=True,
        time_weight=0.3
    )
    
    print(f"\nTop 5 nodelists by availability:")
    for i, (nodelist, score, details) in enumerate(ranked[:5]):
        slurm_format = generator.format_nodelist_for_slurm(nodelist)
        print(f"\n{i+1}. Score: {score:.3f}")
        print(f"   Nodelist: {slurm_format}")
        print(f"   Free: {details['free_nodes']}/{len(nodelist)} ({details['free_percentage']:.1f}%)")
        print(f"   Running: {details['running_nodes']}, Pending: {details['pending_nodes']}")
        if details['max_time_to_free_hours'] is not None:
            print(f"   Max time to free: {details['max_time_to_free_hours']:.1f} hours")
        print(f"   Topology: {details['contiguity']}")


if __name__ == "__main__":
    example_usage()

