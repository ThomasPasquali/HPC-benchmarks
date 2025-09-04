import itertools
from pprint import pprint
import re
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import sbatchman as sbm

FONT_TITLE = 18
FONT_AXES = 18
FONT_TICKS = 16
FONT_LEGEND = 12

plt.rc('axes', titlesize=FONT_AXES)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_AXES)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_LEGEND)  # legend fontsize
plt.rc('figure', titlesize=FONT_TITLE)  # fontsize of the figure title

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

# def make_plots(data):
#   # Group by (scale, ef)
#   teps_plot_data = defaultdict(lambda: defaultdict(list))  # {(scale, ef): {(partition, impl): [(nodes, teps, cut_teps)]}}
#   for (partition, impl, scale, ef), entries in data.items():
#     for entry in entries:
#       (nodes, teps, cut_teps) = entry[0]
#       teps_plot_data[(scale, ef)][(partition, impl)].append((nodes, teps, cut_teps))

#   for (scale, ef), groups in teps_plot_data.items():
#     plt.figure(figsize=(10, 6))

#     for (partition, impl), values in groups.items():
#       values.sort()  # sort by node count
#       nodes, teps_vals, cut_teps_vals = zip(*values)
#       label_base = f"{partition}-{impl}"
#       plt.plot(nodes, teps_vals, marker='o', label=f"{label_base} TEPS")
#       plt.plot(nodes, cut_teps_vals, marker='s', linestyle='--', label=f"{label_base} CUT_TEPS")

#     plt.title(f"TEPS and CUT_TEPS vs Nodes (Scale={scale}, Edgefactor={ef})")
#     plt.xlabel("Nodes")
#     plt.ylabel("TEPS")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(OUT_DIR / f'Graph500_teps_vs_nodes_s{scale}_ef{ef}.png')
#     plt.close()


#   for (partition, impl, scale, ef), entries in data.items():
#     barrier_sums_by_partition = defaultdict(lambda: defaultdict(list))  # {(scale, ef): {partition: [sums]}}
#     nodes_list, teps_list, teps_cut_list = zip(*[entry[0] for entry in entries])

#     # === Per-job plots ===
#     for i, (_, _, barrier_matrix, comm_volume, comm_count) in enumerate(entries):
#       nodes = nodes_list[i]

#       # 1a. Barrier time per rank (sum over runs)
#       bt_sum = barrier_matrix.sum(axis=1)
#       plt.figure(figsize=(8, 4))
#       plt.bar(np.arange(len(bt_sum)), bt_sum)
#       plt.title(f"Barrier Time per Rank (Sum) - {partition}, {impl}, s{scale}, ef{ef}, {nodes}n")
#       plt.xlabel("Rank")
#       plt.ylabel("Total Barrier Time")
#       plt.tight_layout()
#       plt.savefig(OUT_DIR / f'Graph500_{partition}_{impl}_barrier_sum_barplot_s{scale}_ef{ef}_{nodes}n.png')
#       plt.close()

#       key_summary = (scale, edgefactor)
#       sum_bt = barrier_matrix.sum(axis=1).tolist()
#       barrier_sums_by_partition[key_summary][(partition, impl)].extend(sum_bt)

#       # 2. Barrier Heatmap
#       plt.figure(figsize=(10, 6))
#       sns.heatmap(barrier_matrix, cmap="viridis")
#       plt.title(f"Barrier Times - Scale {scale}, EF {ef}, Nodes {nodes}")
#       plt.xlabel("Run #")
#       plt.ylabel("Rank")
#       plt.tight_layout()
#       plt.savefig(OUT_DIR / f'Graph500_{partition}_{impl}_barrier_s{scale}_ef{ef}_{nodes}n.png')
#       plt.close()

#       # 3. Communication Heatmaps
#       if nodes > 1:
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(comm_volume, cmap='magma', square=True)
#         plt.title(f"Communication Volume ({nodes} nodes)")
#         plt.xlabel("Destination Rank")
#         plt.ylabel("Source Rank")
#         plt.tight_layout()
#         plt.savefig(OUT_DIR / f'Graph500_{partition}_{impl}_comm_volume_s{scale}_ef{ef}_{nodes}n.png')
#         plt.close()

#         plt.figure(figsize=(8, 6))
#         sns.heatmap(comm_count, cmap='YlGnBu', square=True)
#         plt.title(f"Communication Count ({nodes} nodes)")
#         plt.xlabel("Destination Rank")
#         plt.ylabel("Source Rank")
#         plt.tight_layout()
#         plt.savefig(OUT_DIR / f'Graph500_{partition}_{impl}_comm_count_s{scale}_ef{ef}_{nodes}n.png')
#         plt.close()

#       # 4. Barrier vs Comm Volume per rank
#       barrier_avg = barrier_matrix.mean(axis=1)
#       total_vol_sent = comm_volume.sum(axis=1)
#       plt.figure(figsize=(6, 4))
#       plt.scatter(total_vol_sent, barrier_avg)
#       plt.xlabel("Total Volume Sent (per rank)")
#       plt.ylabel("Average Barrier Time (per rank)")
#       plt.title(f"Barrier Time vs Volume Sent ({nodes} nodes)")
#       plt.grid(True)
#       plt.tight_layout()
#       plt.savefig(OUT_DIR / f'Graph500_{partition}_{impl}_barrier_vs_comm_s{scale}_ef{ef}_{nodes}n.png')
#       plt.close()

#       # Compute avg packet size matrix
#       if nodes > 1:
#         with np.errstate(divide='ignore', invalid='ignore'):
#           avg_pkt_size = np.divide(comm_volume, comm_count)
#           avg_pkt_size[np.isnan(avg_pkt_size)] = 0  # Optional: replace NaNs

#         plt.figure(figsize=(8, 6))
#         sns.heatmap(avg_pkt_size, cmap='coolwarm', square=True)
#         plt.title(f"Avg Packet Size - {partition}, {impl}, s{scale}, ef{ef}, {nodes}n")
#         plt.xlabel("Destination Rank")
#         plt.ylabel("Source Rank")
#         plt.tight_layout()
#         plt.savefig(OUT_DIR / f'Graph500_{partition}_{impl}_avg_packet_size_heatmap_s{scale}_ef{ef}_{nodes}n.png')
#         plt.close()

#         per_rank_pkt_size = []
#         labels = []

#         for src in range(avg_pkt_size.shape[0]):
#           nonzero = avg_pkt_size[src, avg_pkt_size[src] > 0]
#           if len(nonzero) > 0:
#             per_rank_pkt_size.append(nonzero)
#             labels.append(str(src))

#         plt.figure(figsize=(12, 6))
#         sns.boxplot(data=per_rank_pkt_size)
#         plt.xticks(np.arange(len(labels)), labels, rotation=45)
#         plt.title(f"Avg Packet Size by Source Rank - {partition}, {impl}, s{scale}, ef{ef}, {nodes}n")
#         plt.xlabel("Source Rank")
#         plt.ylabel("Average Packet Size")
#         plt.tight_layout()
#         plt.savefig(OUT_DIR / f'Graph500_{partition}_{impl}_avg_packet_size_boxplot_s{scale}_ef{ef}_{nodes}n.png')
#         plt.close()

#     # === Plots on aggregates ===
#     for (scale, ef), part_data in barrier_sums_by_partition.items():
#       labels = []
#       values = []
#       for (partition, impl), sums in part_data.items():
#         labels.extend([f"{partition}-{impl}"] * len(sums))
#         values.extend(sums)

#       plt.figure(figsize=(10, 6))
#       sns.boxplot(x=labels, y=values)
#       plt.xticks(rotation=45)
#       plt.title(f"Barrier Time Sum per Rank (Grouped by Partition & Impl) - s{scale}, ef{ef}")
#       plt.ylabel("Total Barrier Time")
#       plt.xlabel("Partition-Impl")
#       plt.tight_layout()
#       plt.savefig(OUT_DIR / f'Graph500_barrier_boxplot_s{scale}_ef{ef}.png')
#       plt.close()


def make_plots(df_aggr: pd.DataFrame, df: pd.DataFrame):
  # line_styles = itertools.cycle(["-", "--", "-.", ":"])
  line_styles = {
    'haicgu-ib': '-',
    'haicgu-eth': '--',
  }
  markers = {
    'smallbuf': 'o',
    'classic': 'x',
  }

  # TEPS and CUT_TEPS vs Nodes
  for (scale, ef), group in df_aggr.groupby(['scale', 'edgefactor']):
    plt.figure(figsize=(12, 6))

    x_ticks_nodes = set()
    for (cluster, partition, impl), impl_group in group.groupby(['cluster', 'partition', 'impl']):
      impl_group_sorted = impl_group.sort_values('nodes')
      nodes = impl_group_sorted['nodes']
      teps_vals = impl_group_sorted['teps']
      cut_teps_vals = impl_group_sorted['cut_teps']

      # linestyle = next(line_styles)
      linestyle = line_styles.get(f'{cluster}-{partition}', ':')
      marker = markers.get(impl, 's')
      label_base = f"{cluster}-{partition}-{impl}"
      plt.plot(nodes, teps_vals, marker=marker, linestyle=linestyle, label=f"{label_base} TEPS")
      plt.plot(nodes, cut_teps_vals, marker=marker, linestyle=linestyle, label=f"{label_base} CUT_TEPS")
      x_ticks_nodes |= set(nodes.values)

    plt.title(f"TEPS and CUT_TEPS vs Nodes - Scale={scale}, Edgefactor={ef}")
    plt.xlabel("Nodes")
    plt.ylabel("TEPS and CUT_TEPS")
    plt.xticks(sorted(list(x_ticks_nodes)))
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(title="Cluster-Partition-Implementation Metric")
    plt.tight_layout()
    path = OUT_DIR / 'scaling' / f'Graph500_teps_vs_nodes_s{scale}_ef{ef}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    print(f'Plot saved to {path}')
    plt.close()

  # ONLY CUT_TEPS vs Nodes
  for (scale, ef), group in df_aggr.groupby(['scale', 'edgefactor']):
    plt.figure(figsize=(12, 6))

    x_ticks_nodes = set()
    for (cluster, partition, impl), impl_group in group.groupby(['cluster', 'partition', 'impl']):
      impl_group_sorted = impl_group.sort_values('nodes')
      nodes = impl_group_sorted['nodes']
      cut_teps_vals = impl_group_sorted['cut_teps']

      # linestyle = next(line_styles)
      linestyle = line_styles.get(f'{cluster}-{partition}', ':')
      marker = markers.get(impl, 's')
      label_base = f"{cluster}-{partition}-{impl}"
      plt.plot(nodes, cut_teps_vals, marker=marker, linestyle=linestyle, label=f"{label_base} CUT_TEPS")
      x_ticks_nodes |= set(nodes.values)

    plt.title(f"CUT_TEPS vs Nodes - Scale={scale}, Edgefactor={ef}")
    plt.xlabel("Nodes")
    plt.ylabel("CUT_TEPS")
    plt.xticks(sorted(list(x_ticks_nodes)))
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(title="Cluster-Partition-Implementation Metric")
    plt.tight_layout()
    path = OUT_DIR / 'scaling' / f'Graph500_teps_vs_nodes_s{scale}_ef{ef}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    print(f'Plot saved to {path}')
    plt.close()

  # Boxplot of mean barrier time per partition-impl, grouped by (scale, edgefactor)
  # for (scale, ef), group in df_aggr.groupby(['scale', 'edgefactor']):
  #   plt.figure(figsize=(12, 6))
  #   group = group.sort_values(['impl', 'cluster', 'partition', 'nodes'])
  #   group['label'] = group['cluster'] + '-' + group['partition'] + '-' + group['impl'] + '-' + group['nodes'].astype(str)
  #   print(group['mean_barrier_time'])
  #   exit()
  #   sns.boxplot(x='label', y='mean_barrier_time', data=group)
  #   plt.xticks(rotation=90, fontsize=10)

  #   # Add vertical divisor lines after each (cluster, partition, impl) group
  #   group_keys = group[['cluster', 'partition', 'impl']].astype(str).agg('-'.join, axis=1)
  #   prev_key = None
  #   divisor_pos = []
  #   divisor_text = []
  #   for i, key in enumerate(group_keys):
  #     if prev_key is not None and key != prev_key:
  #       if len(divisor_text) <= 0:
  #         divisor_text.append(prev_key)
  #       divisor_text.append(key)
  #       divisor_pos.append(i - 0.5)
  #     prev_key = key
  #   ymax = plt.ylim()[1]
  #   for pos, text in zip([0.0]+divisor_pos, divisor_text):
  #     plt.text(pos + 0.3, ymax*0.95, text, fontsize=10) # ha='center', va='center'
  #     if pos > 0:
  #       plt.axvline(x=pos, color='black', linestyle='-', linewidth=2)

  #   plt.title(f"Barrier Time (arith. mean over ranks and runs) - Scale={scale}, Edgefactor={ef}")
  #   plt.ylabel("Mean Barrier Time [s]")
  #   plt.xlabel("Cluster-Partition-Implementation-Nodes")
  #   plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
  #   plt.tight_layout()
  #   path = OUT_DIR / 'barrier' / f'Graph500_barrier_boxplot_s{scale}_ef{ef}.png'
  #   path.parent.mkdir(parents=True, exist_ok=True)
  #   plt.savefig(path)
  #   print(f'Plot saved to {path}')
  #   plt.close()

  ## OLD VERSION of barrier time and avg packet size boxplots
  # df_filtered = df[df["nodes"] > 2]
  # # Ensure (cluster, partition) is a combined label for hue
  # df_filtered["cluster_partition"] = df_filtered["cluster"] + "-" + df_filtered["partition"]

  # # Get all unique combinations of (scale, edgefactor)
  # group_keys = df_filtered[["scale", "edgefactor"]].drop_duplicates()
  # for _, row in group_keys.iterrows():
  #   scale = row["scale"]
  #   edgefactor = row["edgefactor"]

  #   # Filter for this group
  #   subset = df_filtered[(df_filtered["scale"] == scale) & (df_filtered["edgefactor"] == edgefactor)]

  #   # Create the figure with two side-by-side plots
  #   fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
  #   fig.suptitle(f"Scale: {scale}, Edgefactor: {edgefactor}", fontsize=16)

  #   # Left: mean_barrier_time
  #   sns.boxplot(
  #     data=subset,
  #     x="impl",
  #     y="mean_barrier_time",
  #     hue="cluster_partition",
  #     ax=axes[0]
  #   )
  #   axes[0].set_title("Avg Barrier Time (arith. mean)")
  #   axes[0].set_ylabel("Avg Barrier Time [s]")
  #   axes[0].set_xlabel("Implementation")
  #   axes[0].legend()

  #   # Right: mean_packet_size
  #   sns.boxplot(
  #     data=subset,
  #     x="impl",
  #     y="mean_packet_size",
  #     hue="cluster_partition",
  #     ax=axes[1]
  #   )
  #   axes[1].set_title("Avg. Packet Size (arith. mean)")
  #   axes[1].set_ylabel("Avg. Packet Size [Bytes]")
  #   axes[1].set_xlabel("Implementation")
  #   axes[1].legend()

  #   plt.tight_layout()  # Leave space for the legend
  #   path = OUT_DIR / 'barrier_and_avgsize' / f'Graph500_barrier_avgpacketsize_boxplot_s{scale}_ef{edgefactor}.png'
  #   path.parent.mkdir(parents=True, exist_ok=True)
  #   plt.savefig(path)
  #   print(f'Plot saved to {path}')
  #   plt.close()

  # Flag to toggle outlier removal
  REMOVE_OUTLIERS = False

  # Filter input DataFrame
  df_filtered = df[df["nodes"] > 2].copy()
  df_filtered["cluster_partition"] = df_filtered["cluster"] + "-" + df_filtered["partition"]

  # Function to remove outliers using IQR method
  def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    return data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]

  # Get all unique combinations of (scale, edgefactor)
  group_keys = df_filtered[["scale", "edgefactor", "nodes"]].drop_duplicates()

  for _, row in group_keys.iterrows():
    scale = row["scale"]
    edgefactor = row["edgefactor"]
    nodes = row["nodes"]

    subset = df_filtered[
      (df_filtered["scale"] == scale)
      & (df_filtered["edgefactor"] == edgefactor)
      & (df_filtered["nodes"] == nodes)
    ]

    # Optionally remove outliers
    subset_full = subset
    if REMOVE_OUTLIERS:
      subset = remove_outliers_iqr(subset, "mean_barrier_time")
      # We don't filter "mean_packet_size" since it’s now plotted as a barplot over means

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(f"Nodes: {nodes}, Scale: {scale}, Edgefactor: {edgefactor}", fontsize=18)

    # Left: boxplot of mean_barrier_time
    sns.boxplot(
      data=subset_full,
      x="impl",
      y="mean_barrier_time",
      hue="cluster_partition",
      ax=axes[0],
      showfliers=not REMOVE_OUTLIERS,
    )
    axes[0].set_title("Avg Barrier Time (arith. mean)")
    axes[0].set_ylabel("Avg Barrier Time [s]")
    axes[0].set_xlabel("Implementation")
    axes[0].legend()

    # Right: barplot of mean_packet_size (mean ± std)
    grouped = subset.groupby(["impl", "cluster_partition"])["mean_packet_size"].agg(['mean', 'std']).reset_index()
    sns.barplot(
      data=grouped,
      x="impl",
      y="mean",
      hue="cluster_partition",
      ax=axes[1],
      capsize=0.1,
      err_kws={'linewidth': 1.5},
    )
    # Add manual error bars aligned with bar centers
    # We retrieve bar locations from the bar containers
    for bars, (_, group) in zip(axes[1].containers, grouped.groupby("cluster_partition")):
      for bar, (_, row) in zip(bars, group.iterrows()):
        height = bar.get_height()
        axes[1].errorbar(
          x=bar.get_x() + bar.get_width() / 2,
          y=height,
          yerr=row["std"],
          fmt='none',
          c='black',
          capsize=4,
          linewidth=1
        )

    axes[1].set_title("Avg. Packet Size (mean ± std)")
    axes[1].set_ylabel("Avg. Packet Size [Bytes]")
    axes[1].set_xlabel("Implementation")
    axes[1].legend()

    plt.tight_layout()
    path = OUT_DIR / 'barrier_and_avgsize' / f'Graph500_barrier_avgpacketsize{"_noutliers" if REMOVE_OUTLIERS else ""}_s{scale}_ef{edgefactor}_n{nodes}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    print(f'Plot saved to {path}')
    plt.close()



if __name__ == "__main__":
  if len(sys.argv) > 1 and (len(sys.argv) - 1) % 2 != 0:
    print(f'Usage: python3 {sys.argv[0]} path/to/summary_aggr1 path/to/summary1 [path/to/summary_aggr2 path/to/summary2 ...]')
    exit(1)
  elif len(sys.argv) > 2:
    df_aggr_list = []
    df_list = []

    for i in range(1, len(sys.argv), 2):
      data_aggr_path = Path(sys.argv[i])
      data_path = Path(sys.argv[i + 1])

      if not (data_aggr_path.exists() and data_aggr_path.is_file() and data_path.exists() and data_path.is_file()):
        print(f'CSV files do not exist: {data_aggr_path}, {data_path}')
        exit(2)

      print(f'Reading aggregated data from file: "{data_aggr_path}"')
      df_aggr = pd.read_csv(data_aggr_path)
      df_aggr_list.append(df_aggr)
      print(f'Reading data from file: "{data_path}"')
      df = pd.read_csv(data_path)
      df_list.append(df)

    df_aggr = pd.concat(df_aggr_list, ignore_index=True)
    df = pd.concat(df_list, ignore_index=True)
  else:
    data = defaultdict(list)
    jobs = sbm.jobs_list(from_active=True, from_archived=False, status=[sbm.Status.COMPLETED])
    for job in jobs:
      m = re.match(r'(\w+)_(\d+)nodes', job.config_name)
      if not m:
        continue

      command_parts = job.command.split(' ')
      edgefactor = command_parts[-1]
      scale = command_parts[-2]
      program = None
      for p in command_parts:
        if 'graph500_reference_bfs' in p:
          program = p
          break
      if p is None:
        raise Exception(f'Could not find executable in command "{job.command}"')
      partition, nodes_str = m.groups()
      nodes = int(nodes_str)

      cut_teps, teps, barrier_times, comm_stats = parse_metrics_file(job.get_stdout_path())

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
          "mean_packet_size": np.mean(np.divide(volume_matrix, count_matrix, where=count_matrix != 0))
        })

        for run_i in range(len(barrier_matrix)):
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
            "mean_packet_size": np.mean(np.divide(volume_matrix[run_i], count_matrix[run_i], where=count_matrix[run_i] != 0))
          })

    df_aggr = pd.DataFrame(df_records_aggr)
    path = OUT_DIR / f"graph500_{sbm.get_cluster_name()}_summary_aggr.csv"
    df_aggr.to_csv(path, index=False)
    print(f"Wrote CSV summary with {len(df_aggr)} rows to {path.resolve().absolute()}")

    df = pd.DataFrame(df_records)
    path = OUT_DIR / f"graph500_{sbm.get_cluster_name()}_summary.csv"
    df.to_csv(path, index=False)
    print(f"Wrote CSV summary with {len(df)} rows to {path.resolve().absolute()}")

  make_plots(df_aggr, df)
