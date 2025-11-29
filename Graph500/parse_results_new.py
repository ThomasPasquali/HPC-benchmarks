from pathlib import Path
from pprint import pprint
import sbatchman as sbm
import numpy as np
import re
from collections import defaultdict
from typing import Dict, Tuple
import pandas as pd
import graph500.ccutils.ccutils_parser as ccutils_parser

S='''

'''

OUT_DIR = Path('results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def dict_get(d, key):
  res = d.get(key)
  if not res:
    raise Exception(f'{key} not found')
  return res

def main():
  res = ccutils_parser.parse_ccutils_output(str(S))
  details = dict_get(res, 'detailed_results')
  packet_bandwidth = dict_get(details.mpi_all_prints, 'packet_bandwidth')
  network_packets = []
  for dest in packet_bandwidth.get_all_ranks():
    rank_output = packet_bandwidth.get_rank_output(dest)
    if not rank_output:
      print('WARNING: not rank output')
      continue
    for msg in rank_output.strip().split(' '):
      src, size, time = msg.split(',')
      network_packets.append([src, dest, size, time])
  
  network_packets_df = pd.DataFrame(network_packets, columns=['src', 'dest', 'size', 'time'])
  network_packets_df.to_parquet("network_packets.parquet")
  
  import matplotlib.pyplot as plt
  plt.scatter(network_packets_df['size'], network_packets_df['time']) # , s=area, c=colors, alpha=0.5)
  plt.show()
  
if __name__ == "__main__":
  main()