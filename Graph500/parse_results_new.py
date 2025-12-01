from pathlib import Path
from pprint import pprint
import sys
import sbatchman as sbm
import numpy as np
import re
from collections import defaultdict
from typing import Dict, Tuple
import pandas as pd
import graph500.ccutils.ccutils_parser as ccutils_parser

sys.path.append(str(Path(__file__).parent.parent / 'machines' / 'Leonardo'))
from nodelists_generator import LeonardoNodelistGenerator

sys.path.append(str(Path(__file__).parent.parent))
from py_utils.utils import create_color_map, create_marker_map

OUT_DIR = Path('results_test')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def dict_get(d, key):
  res = d.get(key)
  if res is None:
    raise Exception(f'{key} not found')
  return res

def raise_none(v, descr: str):
  if v is None:
    raise Exception(f'{descr} not found')
  return v

def main():
  leo_map = LeonardoNodelistGenerator()
  jobs = sbm.jobs_list(status=[sbm.Status.COMPLETED])
  
  for j in jobs:
    res = ccutils_parser.parse_ccutils_output(raise_none(j.get_stdout(), 'stdout'))
    
    # Extract nodes distance from topology
    ranks_nodes_map = {}
    nodes = dict_get(res, 'node_names')
    nodes = raise_none(nodes.get_mpi_print('node_names'), 'node_names')
    for rank in raise_none(nodes.get_all_ranks(), 'nodes.get_all_ranks()'):
      ranks_nodes_map[rank] = int(raise_none(nodes.get_rank_output(rank), f'rank {rank} node name').split('.')[0][4:])
    
    print(f'{ranks_nodes_map=}')
    
    details = dict_get(res, 'detailed_results')
    packet_bandwidth = dict_get(details.mpi_all_prints, 'packet_bandwidth')
    network_packets = []
    for dest in packet_bandwidth.get_all_ranks():
      rank_output = packet_bandwidth.get_rank_output(dest)
      if not rank_output:
        print('WARNING: no rank output')
        continue
      for msg in rank_output.splitlines()[4].strip().split(' '):
        src, size, time = msg.split(',')
        network_packets.append([src, dest, size, time, leo_map.get_node_distance(ranks_nodes_map[int(src)], ranks_nodes_map[dest])])
    
    network_packets_df = pd.DataFrame(network_packets, columns=['src', 'dest', 'size', 'time', 'distance'])
    network_packets_df['size'] = network_packets_df['size'].astype(np.int32)
    network_packets_df['distance'] = network_packets_df['distance'].astype(np.int8)
    network_packets_df['time'] = network_packets_df['time'].astype(np.double)
    negative_mask = network_packets_df['time'] < 0
    if negative_mask.any():
      if (network_packets_df[negative_mask]['time'].abs() > 0.001).any():
        print("WARNING: Negative time values greater than 1ms detected")
      network_packets_df.loc[negative_mask, 'time'] = 0
    network_packets_df.sort_values('size', inplace=True)
    # network_packets_df.to_parquet("network_packets.parquet")
    
    # Remove outliers based on time deviation from average
    mean_time = network_packets_df['time'].mean()
    std_time = network_packets_df['time'].std()
    
    # Filtering strategy: 'std' for standard deviation, 'qnt' for a quantile range
    FILTERING_STRATEGY = 'qnt'

    if FILTERING_STRATEGY == 'std':
      network_packets_df = network_packets_df[
        (network_packets_df['time'] >= mean_time - 1.0 * std_time) &
        (network_packets_df['time'] <= mean_time + 1.0 * std_time)
      ]
    elif FILTERING_STRATEGY == 'qnt':
      q = network_packets_df['time'].quantile(0.33)
      network_packets_df = network_packets_df[
        (network_packets_df['time'] <= q)
      ]
      
    print(network_packets_df)
    print(f'Distinct distances: {network_packets_df['distance'].unique()}')
    
    import matplotlib.pyplot as plt
    distance_color_map = create_color_map(network_packets_df['distance'].unique())
    dest_marker_map = create_marker_map(network_packets_df['dest'].unique())
    legend_added = set()
    for distance in sorted(network_packets_df['distance'].unique()):
      mask = network_packets_df['distance'] == distance
      for dest in network_packets_df[mask]['dest'].unique():
        dest_mask = mask & (network_packets_df['dest'] == dest)
        label = f'Distance: {distance}' if distance not in legend_added else None
        legend_added.add(distance)
        plt.scatter(
          network_packets_df[dest_mask]['size'] / 1024.0,
          network_packets_df[dest_mask]['time'] * 1000.0,
          c=[distance_color_map[distance]],
          marker=dest_marker_map[dest],
          label=label,
          alpha=0.2
        )
    vars = raise_none(j.variables, 'job variables')
    tit_vals = [f'{k}:{v}' for k,v in vars.items()]
    plt.title(' - '.join(tit_vals), fontsize=8)
    plt.xlabel('Packet Size [KiB]')
    plt.ylabel('Send Time [ms]')
    plt.legend(loc='best')
    plt.yticks(np.linspace((network_packets_df['time'] * 1000.0).min(), (network_packets_df['time'] * 1000.0).max(), 10))
    plt.savefig((OUT_DIR / '-'.join(tit_vals)).with_suffix('.png'))
  
if __name__ == "__main__":
  main()