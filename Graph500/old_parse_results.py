from pathlib import Path
import sbatchman as sbm
import numpy as np
import re
from collections import defaultdict
from typing import Dict, Tuple
import pandas as pd

OUT_DIR = Path('results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_metrics_file(filepath: Path, rank_filter=None, run_filter=None) -> Tuple[float, float, Dict, Dict]:
  barrier_times = defaultdict(dict)
  comm_stats = defaultdict(lambda: defaultdict(dict))
  teps = 0.0
  cut_teps = 0.0
  with open(filepath, "r") as f:
    for line in f:
      # print(line[:-1])
      if 'harmonic_mean_TEPS' in line:
        line = re.subn(r'\s{2,}', ' ', line)[0]
        teps = float(line.split(' ')[-1])
        continue
      if 'harmonic_mean_cut_TEPS' in line:
        line = re.subn(r'\s{2,}', ' ', line)[0]
        cut_teps = float(line.split(' ')[-1])
        continue
      if not line.startswith("[METRIC]"):
        continue

      tokens = line.strip().split()
      try:
        rank = int(tokens[1].split("=")[1])
        run = int(tokens[2].split("=")[1])

        if rank_filter is not None and rank != rank_filter:
          continue
        if run_filter is not None and run != run_filter:
          continue

        if "barrier_wait_time" in line:
          time = float(tokens[3].split("=")[1])
          barrier_times[rank][run] = time
        else:
          dest = int(tokens[3].split("=")[1])
          n_comms = int(tokens[4].split("=")[1])
          volume = int(tokens[5].split("=")[1])
          comm_stats[rank][run][dest] = (n_comms, volume)
      except (IndexError, ValueError):
        raise

  return cut_teps, teps, barrier_times, comm_stats

def main():
  data = defaultdict(list)
  jobs = sbm.jobs_list(from_active=True, from_archived=False, status=[sbm.Status.COMPLETED])
  for job in jobs:
    m = re.match(r'(\w+)_(\d+)nodes', job.config_name)
    if not m:
      continue

    command_parts = job.command.split(' ')
    edgefactor = int(command_parts[-1])
    scale = int(command_parts[-2])
    program = None
    p = None
    for p in command_parts:
      if 'graph500_reference_bfs' in p:
        program = p
        break
    if p is None:
      raise Exception(f'Could not find executable in command "{job.command}"')
    partition, nodes_str = m.groups()
    nodes = int(nodes_str)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # print((nodes, scale, edgefactor))
    # if nodes == 4 and edgefactor == 8 and scale == 14:
    # print('='*100)
    cut_teps, teps, barrier_times, comm_stats = parse_metrics_file(job.get_stdout_path())
    # else:
    #   continue

    if not barrier_times:
      continue

    num_ranks = len(barrier_times)
    num_runs = len(next(iter(barrier_times.values())))  # use first rank to get run count

    # Create barrier time matrix (ranks x runs)
    barrier_matrix = np.zeros((num_runs, num_ranks))
    for r in barrier_times:
      for run in barrier_times[r]:
        barrier_matrix[run, r] = barrier_times[r][run]

    volume_matrix = np.zeros((num_runs, num_ranks, num_ranks))
    count_matrix = np.zeros((num_runs, num_ranks, num_ranks))
    for src, run_dict in comm_stats.items():
      for run, dst_dict in run_dict.items():
        for dst, (n_comms, volume) in dst_dict.items():
          volume_matrix[run][src][dst] += volume
          count_matrix[run][src][dst] += n_comms

    impl = 'classic'
    if 'smallbuf' in program:
      impl = 'smallbuf'
    elif 'largebuf' in program:
      impl = 'largebuf'
      
    cluster = job.cluster_name
    key = (cluster, partition, impl, scale, edgefactor)

    data[key].append((
      (nodes, teps, cut_teps),    # TEPS
      nodes,                      # for filename suffixes
      barrier_matrix,             # Barrier Times
      volume_matrix,              # Volume matrix
      count_matrix                # Count matrix
    ))

  # Prepare DataFrame
  df_records_aggr = []
  df_records = []
  for (cluster, partition, impl, scale, edgefactor), entries in data.items():
    for entry in entries:
      (nodes, teps, cut_teps), _, barrier_matrix, volume_matrix, count_matrix = entry

      mean_packet_size = np.nanmean(np.divide(
        volume_matrix, count_matrix,
        out=np.full_like(volume_matrix, np.nan, dtype=float),
        where=count_matrix != 0
      ))
      df_records_aggr.append({
        "cluster": cluster,
        "partition": partition,
        "impl": impl,
        "scale": int(scale),
        "edgefactor": int(edgefactor),
        "nodes": nodes,
        "teps": teps,
        "cut_teps": cut_teps,
        "mean_barrier_time": np.mean(barrier_matrix),
        "std_barrier_time": np.std(barrier_matrix),
        "total_comm_volume": np.sum(volume_matrix),
        "total_comm_count": np.sum(count_matrix),
        "mean_packet_size": 0.0 if np.isnan(mean_packet_size) else mean_packet_size,
      })

      for run_i in range(len(barrier_matrix)):
        mean_packet_size = np.nanmean(np.divide(
          volume_matrix[run_i], count_matrix[run_i],
          out=np.full_like(volume_matrix[run_i], np.nan, dtype=float),
          where=count_matrix[run_i] != 0
        ))
        # print(volume_matrix[run_i])
        # print(count_matrix[run_i])
        # print(np.divide(
        #     volume_matrix[run_i], count_matrix[run_i],
        #     out=np.full_like(volume_matrix[run_i], np.nan, dtype=float),
        #     where=count_matrix[run_i] != 0
        #   ))
        # print(mean_packet_size)
        # print(np.nansum(np.divide(
        #     volume_matrix[run_i], count_matrix[run_i],
        #     out=np.full_like(volume_matrix[run_i], np.nan, dtype=float),
        #     where=count_matrix[run_i] != 0
        #   )))
        # print('='*100)
        
        df_records.append({
          "cluster": cluster,
          "partition": partition,
          "impl": impl,
          "scale": int(scale),
          "edgefactor": int(edgefactor),
          "nodes": nodes,
          "run": run_i,
          "mean_barrier_time": np.mean(barrier_matrix[run_i]),
          "std_barrier_time": np.std(barrier_matrix[run_i]),
          "total_comm_volume": np.sum(volume_matrix[run_i]),
          "total_comm_count": np.sum(count_matrix[run_i]),
          "mean_packet_size": 0.0 if np.isnan(mean_packet_size) else mean_packet_size,
        })
        

  df_aggr = pd.DataFrame(df_records_aggr)
  path = OUT_DIR / f"graph500_{sbm.get_cluster_name()}_summary_aggr.csv"
  df_aggr.to_csv(path, index=False)
  print(f"Wrote CSV summary with {len(df_aggr)} rows to {path.resolve().absolute()}")

  df = pd.DataFrame(df_records)
  path = OUT_DIR / f"graph500_{sbm.get_cluster_name()}_summary.csv"
  df.to_csv(path, index=False)
  print(f"Wrote CSV summary with {len(df)} rows to {path.resolve().absolute()}")
  
if __name__ == "__main__":
  main()