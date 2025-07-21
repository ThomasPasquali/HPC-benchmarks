import itertools
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sbatchman as sbm


FONT_TITLE = 18
FONT_AXES = 18
FONT_TICKS = 16
FONT_LEGEND = 14

plt.rc('axes', titlesize=FONT_AXES)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_AXES)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_LEGEND)  # legend fontsize
plt.rc('figure', titlesize=FONT_TITLE)  # fontsize of the figure title


CACHE_SIZES = {
  'pioneer':    [(64 * 1024, 'L1 cache'), (1 * 1024 * 1024, 'L2 cache'),  (64 * 1024 * 1024, 'L3 cache')],
  'bananaf3':   [(32 * 1024, 'L1 cache'), (512 * 1024     , 'L2 cache'),  (512 * 1024,       'TCM')     ],
  'arriesgado': [(64 * 1024, 'L1 cache'), (2 * 1024 * 1024, 'L2 cache'),  (0,                '')        ],
}


def parse_random_chase(path, hw_name):
  data = np.genfromtxt(path, skip_header=2, usecols=(0, 1))
  return pd.DataFrame(data, columns=['x', 'y']).assign(program='random-chase', hw=hw_name)


def parse_linear_chase(path, hw_name):
  data = np.genfromtxt(path, skip_header=2, usecols=(0, 1))
  return pd.DataFrame(data, columns=['x', 'y']).assign(program='linear-chase', hw=hw_name)


def parse_fused_linear_chase(path, hw_name):
  try:
    raw = np.genfromtxt(path, skip_header=4)
  except ValueError:
    raw = np.genfromtxt(path, skip_header=4, skip_footer=1)

  stride = raw[:, 0]
  dfs = []
  for fuse in range(8):  # fuse factors 1-8
    y = raw[:, fuse + 1]
    df = pd.DataFrame({
      'x': stride,
      'y': y,
      'fuse': fuse + 1,
      'program': 'fused-linear-chase',
      'hw': hw_name
    })
    dfs.append(df)
  return pd.concat(dfs, ignore_index=True)


PARSERS = {
  'random-chase': parse_random_chase,
  'linear-chase': parse_linear_chase,
  'fused-linear-chase': parse_fused_linear_chase
}


def generate_dataframe_from_jobs(jobs):
  dfs = []
  for job in jobs:
    prog = job.tag
    hw = job.config_name
    if prog not in PARSERS:
      print(f"Skipping unrecognized program: {prog}")
      continue
    df = PARSERS[prog](job.get_stdout_path(), hw)
    dfs.append(df)
  return pd.concat(dfs, ignore_index=True)


def plot_random(df, dst: Path, hws_color_map, hws_linestyle_map, hws_marker_map, fuse_color_map):
  fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
  
  occupied_cache_size_text_pos = []
  for hw_name, group in df.groupby("hw"):
    ax.plot(group['x'], group['y'], marker="o", markersize=3, linewidth=1, label=hw_name, color=hws_color_map[hw_name])
    l1, l2, l3 = CACHE_SIZES.get(hw_name, ((0,''), (0,''), (0,'')))
    for pos, name in l1, l2, l3:
      if pos > 0:
        ax.axvline(x=pos, linestyle="--", color=hws_color_map[hw_name], linewidth=0.9)
        occupied = any([abs(pos-p)<2048 for p in occupied_cache_size_text_pos])
        ax.text(pos*(1.03 if occupied else 0.78), df['y'].max()/1.7, f'{hw_name} {name}', rotation=90, color=hws_color_map[hw_name])
        occupied_cache_size_text_pos.append(pos)

  ax.set_xscale("log", base=2)
  ax.set_xlabel("Memory Area [Bytes]")
  ax.set_ylabel("Avg access time [ns]")
  ax.set_title("Access Times vs Memory Area")
  ax.grid(True, linestyle="-", alpha=0.8)
  ax.legend()
  fig.tight_layout()
  fig.savefig(dst)
  print(f'Plot saved to {dst}')
  plt.close(fig)


def plot_linear(df, dst: Path, hws_color_map, hws_linestyle_map, hws_marker_map, fuse_color_map):
  fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
  for hw_name, group in df.groupby("hw"):
    ax.plot(group['x'], group['y'], marker="o", markersize=3, linewidth=1, label=hw_name, color=hws_color_map[hw_name])

  ax.set_xlabel("Stride [Bytes]")
  ax.set_ylabel("Avg access time [ns]")
  ax.set_title("Access Time vs Stride")
  ax.grid(True, linestyle="-", alpha=0.6)
  ax.legend()
  fig.tight_layout()
  fig.savefig(dst)
  print(f'Plot saved to {dst}')
  plt.close(fig)


def plot_fused(df, dst: Path, hws_color_map, hws_linestyle_map, hws_marker_map, fuse_color_map):
  fig, ax = plt.subplots(figsize=(15, 9), dpi=100)
  for (hw_name, fuse), group in df.groupby(["hw", "fuse"]):
    label = f"{hw_name} - Fuse {int(fuse)}"
    ax.plot(group['x'], group['y'], marker=hws_marker_map[hw_name], markersize=3, linewidth=1, label=label, linestyle=hws_linestyle_map[hw_name], color=fuse_color_map[fuse])

  ax.set_xlabel("Stride [Bytes]")
  ax.set_ylabel("Bandwidth [GiB/s]")
  ax.set_title("Bandwidth varying Stride and Fuse")
  ax.grid(True, linestyle="-", alpha=0.8)
  ax.legend()
  fig.tight_layout()
  fig.savefig(dst)
  print(f'Plot saved to {dst}')
  plt.close(fig)


PLOTTERS = {
  'random-chase': plot_random,
  'linear-chase': plot_linear,
  'fused-linear-chase': plot_fused
}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", type=Path, default=Path("./results"))
  parser.add_argument("--csv", type=Path, help="Optional CSV file to read data from")
  args = parser.parse_args()

  args.output_dir.mkdir(parents=True, exist_ok=True)

  if args.csv:
    print(f"Reading data from CSV: {args.csv}")
    df = pd.read_csv(args.csv)
  else:
    print("Generating data from jobs...")
    jobs = sbm.jobs_list(from_active=False, from_archived=True)
    df = generate_dataframe_from_jobs(jobs)
    path = args.output_dir / 'pointer_chasing_data.csv'
    df.to_csv(path, index=False)
    print(f"Saved dataframe to CSV: {path}")
    
  color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
  hws_color_map = dict(zip(df['hw'].unique(), itertools.cycle(color_cycle)))
  fuse_color_map = dict(zip(df['fuse'].unique(), itertools.cycle(color_cycle)))
  
  linestyle_cycle = itertools.cycle(["-", "--", "-.", ":"])
  hws_linestyle_map = dict(zip(df['hw'].unique(), linestyle_cycle))
  
  marker_cycle = itertools.cycle(["o", "v", "P", "X"])
  hws_marker_map = dict(zip(df['hw'].unique(), marker_cycle))

  for program, plot_func in PLOTTERS.items():
    df_subset = df[df['program'] == program]
    if df_subset.empty:
      continue

    # Combined plot over all hardware configs
    plot_func(df_subset, args.output_dir / f"combined_{program}.png", hws_color_map, hws_linestyle_map, hws_marker_map, fuse_color_map)

    # # Individual plots per hardware
    # for hw, hw_df in df_subset.groupby("hw"):
    #   plot_func(hw_df, args.output_dir / f"{hw}_{program}.png", hws_color_map, hws_linestyle_map, fuse_color_map)

  print("✔ All plots generated.")


if __name__ == "__main__":
  main()
