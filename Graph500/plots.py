from pprint import pprint
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import seaborn as sns
import sbatchman as sbm

OUT_DIR = Path('results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_metrics_file(filepath: Path, rank_filter=None, run_filter=None) -> Tuple[float, float, Dict, Dict]:
  barrier_times = defaultdict(dict)
  comm_stats = defaultdict(lambda: defaultdict(dict))
  teps = 0.0
  cut_teps = 0.0
  with open(filepath, "r") as f:
    for line in f:
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
        continue

  return cut_teps, teps, barrier_times, comm_stats

def make_plots(data):
  # Group by (scale, ef)
  teps_plot_data = defaultdict(lambda: defaultdict(list))  # {(scale, ef): {(partition, impl): [(nodes, teps, cut_teps)]}}
  for (partition, impl, scale, ef), entries in data.items():
    for entry in entries:
      (nodes, teps, cut_teps) = entry[0]
      teps_plot_data[(scale, ef)][(partition, impl)].append((nodes, teps, cut_teps))

  for (scale, ef), groups in teps_plot_data.items():
    plt.figure(figsize=(10, 6))

    for (partition, impl), values in groups.items():
      values.sort()  # sort by node count
      nodes, teps_vals, cut_teps_vals = zip(*values)
      label_base = f"{partition}-{impl}"
      plt.plot(nodes, teps_vals, marker='o', label=f"{label_base} TEPS")
      plt.plot(nodes, cut_teps_vals, marker='s', linestyle='--', label=f"{label_base} CUT_TEPS")

    plt.title(f"TEPS and CUT_TEPS vs Nodes (Scale={scale}, Edgefactor={ef})")
    plt.xlabel("Nodes")
    plt.ylabel("TEPS")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'Graph500_teps_vs_nodes_s{scale}_ef{ef}.png')
    plt.close()


  for (partition, impl, scale, ef), entries in data.items():
    barrier_sums_by_partition = defaultdict(lambda: defaultdict(list))  # {(scale, ef): {partition: [sums]}}
    nodes_list, teps_list, teps_cut_list = zip(*[entry[0] for entry in entries])

    # === Per-job plots ===
    for i, (_, _, barrier_matrix, comm_volume, comm_count) in enumerate(entries):
      nodes = nodes_list[i]

      # 1a. Barrier time per rank (sum over runs)
      bt_sum = barrier_matrix.sum(axis=1)
      plt.figure(figsize=(8, 4))
      plt.bar(np.arange(len(bt_sum)), bt_sum)
      plt.title(f"Barrier Time per Rank (Sum) - {partition}, {impl}, s{scale}, ef{ef}, {nodes}n")
      plt.xlabel("Rank")
      plt.ylabel("Total Barrier Time")
      plt.tight_layout()
      plt.savefig(OUT_DIR / f'Graph500_{partition}_{impl}_barrier_sum_barplot_s{scale}_ef{ef}_{nodes}n.png')
      plt.close()

      key_summary = (scale, edgefactor)
      sum_bt = barrier_matrix.sum(axis=1).tolist()
      barrier_sums_by_partition[key_summary][(partition, impl)].extend(sum_bt)

      # 2. Barrier Heatmap
      plt.figure(figsize=(10, 6))
      sns.heatmap(barrier_matrix, cmap="viridis")
      plt.title(f"Barrier Times - Scale {scale}, EF {ef}, Nodes {nodes}")
      plt.xlabel("Run #")
      plt.ylabel("Rank")
      plt.tight_layout()
      plt.savefig(OUT_DIR / f'Graph500_{partition}_{impl}_barrier_s{scale}_ef{ef}_{nodes}n.png')
      plt.close()

      # 3. Communication Heatmaps
      if nodes > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(comm_volume, cmap='magma', square=True)
        plt.title(f"Communication Volume ({nodes} nodes)")
        plt.xlabel("Destination Rank")
        plt.ylabel("Source Rank")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'Graph500_{partition}_{impl}_comm_volume_s{scale}_ef{ef}_{nodes}n.png')
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.heatmap(comm_count, cmap='YlGnBu', square=True)
        plt.title(f"Communication Count ({nodes} nodes)")
        plt.xlabel("Destination Rank")
        plt.ylabel("Source Rank")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'Graph500_{partition}_{impl}_comm_count_s{scale}_ef{ef}_{nodes}n.png')
        plt.close()

      # 4. Barrier vs Comm Volume per rank
      barrier_avg = barrier_matrix.mean(axis=1)
      total_vol_sent = comm_volume.sum(axis=1)
      plt.figure(figsize=(6, 4))
      plt.scatter(total_vol_sent, barrier_avg)
      plt.xlabel("Total Volume Sent (per rank)")
      plt.ylabel("Average Barrier Time (per rank)")
      plt.title(f"Barrier Time vs Volume Sent ({nodes} nodes)")
      plt.grid(True)
      plt.tight_layout()
      plt.savefig(OUT_DIR / f'Graph500_{partition}_{impl}_barrier_vs_comm_s{scale}_ef{ef}_{nodes}n.png')
      plt.close()

      # Compute avg packet size matrix
      if nodes > 1:
        with np.errstate(divide='ignore', invalid='ignore'):
          avg_pkt_size = np.divide(comm_volume, comm_count)
          avg_pkt_size[np.isnan(avg_pkt_size)] = 0  # Optional: replace NaNs

        plt.figure(figsize=(8, 6))
        sns.heatmap(avg_pkt_size, cmap='coolwarm', square=True)
        plt.title(f"Avg Packet Size - {partition}, {impl}, s{scale}, ef{ef}, {nodes}n")
        plt.xlabel("Destination Rank")
        plt.ylabel("Source Rank")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'Graph500_{partition}_{impl}_avg_packet_size_heatmap_s{scale}_ef{ef}_{nodes}n.png')
        plt.close()

        per_rank_pkt_size = []
        labels = []

        for src in range(avg_pkt_size.shape[0]):
          nonzero = avg_pkt_size[src, avg_pkt_size[src] > 0]
          if len(nonzero) > 0:
            per_rank_pkt_size.append(nonzero)
            labels.append(str(src))

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=per_rank_pkt_size)
        plt.xticks(np.arange(len(labels)), labels, rotation=45)
        plt.title(f"Avg Packet Size by Source Rank - {partition}, {impl}, s{scale}, ef{ef}, {nodes}n")
        plt.xlabel("Source Rank")
        plt.ylabel("Average Packet Size")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'Graph500_{partition}_{impl}_avg_packet_size_boxplot_s{scale}_ef{ef}_{nodes}n.png')
        plt.close()

    # === Plots on aggregates ===
    for (scale, ef), part_data in barrier_sums_by_partition.items():
      labels = []
      values = []
      for (partition, impl), sums in part_data.items():
        labels.extend([f"{partition}-{impl}"] * len(sums))
        values.extend(sums)

      plt.figure(figsize=(10, 6))
      sns.boxplot(x=labels, y=values)
      plt.xticks(rotation=45)
      plt.title(f"Barrier Time Sum per Rank (Grouped by Partition & Impl) - s{scale}, ef{ef}")
      plt.ylabel("Total Barrier Time")
      plt.xlabel("Partition-Impl")
      plt.tight_layout()
      plt.savefig(OUT_DIR / f'Graph500_barrier_boxplot_s{scale}_ef{ef}.png')
      plt.close()


if __name__ == "__main__":
  data = defaultdict(list)

  jobs = sbm.jobs_list(from_active=True, from_archived=True, status=[sbm.Status.COMPLETED])
  for job in jobs:
    m = re.match(r'(\w+)_(\d+)nodes', job.config_name)
    if not m:
      continue

    scale, edgefactor = job.command.split(' ')[4:6]
    program = job.command.split(' ')[3]
    partition, nodes_str = m.groups()
    nodes = int(nodes_str)

    cut_teps, teps, barrier_times, comm_stats = parse_metrics_file(job.get_stdout_path())

    if not barrier_times:
      continue

    num_ranks = len(barrier_times)
    num_runs = len(next(iter(barrier_times.values())))  # use first rank to get run count

    # Create barrier time matrix (ranks x runs)
    barrier_matrix = np.zeros((num_ranks, num_runs))
    for r in barrier_times:
      for run in barrier_times[r]:
        barrier_matrix[r, run] = barrier_times[r][run]

    volume_matrix = np.zeros((num_ranks, num_ranks))
    count_matrix = np.zeros((num_ranks, num_ranks))
    for src, run_dict in comm_stats.items():
      for run, dst_dict in run_dict.items():
        for dst, (n_comms, volume) in dst_dict.items():
          volume_matrix[src][dst] += volume
          count_matrix[src][dst] += n_comms

    impl = 'smallbuf' if 'smallbuf' in program else 'classic'
    key = (partition, impl, scale, edgefactor)

    data[key].append((
      (nodes, teps, cut_teps),    # TEPS
      nodes,                      # for filename suffixes
      barrier_matrix,             # Barrier Times
      volume_matrix,              # Volume matrix
      count_matrix                # Count matrix
    ))

  make_plots(data)
