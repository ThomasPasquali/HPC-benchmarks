import re
import sys
import pandas as pd
import seaborn as sns
import sbatchman as sbm
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from py_utils.cli import get_basic_cli_parser, load_csv_files
from py_utils.constants import *

plt.rc('axes', titlesize=FONT_AXES)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_AXES)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_LEGEND)  # legend fontsize
plt.rc('figure', titlesize=FONT_TITLE)  # fontsize of the figure title

USE_LOG_SCALE = True

_JOB_RE = re.compile(r"(\w+)_(\d+)cpus")

def parse_sout(job: sbm.Job) -> pd.DataFrame:
  hw, cores = _JOB_RE.match(job.config_name).groups()
  program = job.tag.split('_')[0]
  cores = int(cores)
  df = pd.read_csv(job.get_stdout_path(), skiprows=1)
  df['hw'] = hw
  df['program'] = program
  df['cores'] = cores
  return df

def generate_dataframe_from_jobs(jobs) -> pd.DataFrame:
  return pd.concat([parse_sout(job) for job in jobs], ignore_index=True)


def _set_common_plot_config(df, ax):
  if USE_LOG_SCALE:
    ax.set_xscale("log", base=2)
  cores_unique = sorted(df['cores'].unique())
  ax.set_xticks(cores_unique)
  ax.set_xticklabels(cores_unique)
  ax.set_xlabel("Cores")
  ax.set_ylabel("Throughput (Mops)")
  ax.grid(True)
  plt.tight_layout()

def compare_programs_same_hw(df, hw_filter, dst: Path):
  """
  For a given hardware, compare how different programs scale with cores.
  """
  subset = df[df['hw'] == hw_filter]
  plt.figure(figsize=(8, 6))
  ax = sns.lineplot(
    data=subset,
    x='cores', y='throughput_mops',
    hue='program',
    style='program',
    markers=True,
    estimator='mean', errorbar='sd'
  )
  plt.title(f"Scaling of Programs on {hw_filter}")
  plt.ylabel("Throughput (Mops)")
  plt.xlabel("Cores")
  plt.grid(True)
  _set_common_plot_config(subset, ax)
  plt.tight_layout()
  plt.savefig(dst, dpi=300)
  print(f'Plot saved to {dst}')
  plt.close()

def compare_hw_same_program(df, program_filter, dst: Path):
  """
  For a given program, compare how it scales across different hardware.
  """
  subset = df[df['program'] == program_filter]
  plt.figure(figsize=(8, 6))
  ax = sns.lineplot(
    data=subset,
    x='cores', y='throughput_mops',
    hue='hw',
    style='hw',
    markers=True,
    estimator='mean', errorbar='sd'
  )
  plt.title(f"Scaling of {program_filter} Across Hardware")
  plt.ylabel("Throughput (Mops)")
  plt.xlabel("Cores")
  plt.grid(True)
  _set_common_plot_config(subset, ax)
  plt.tight_layout()
  plt.savefig(dst, dpi=300)
  print(f'Plot saved to {dst}')
  plt.close()

def compare_programs_hw_combinations(df, dst: Path):
  """
  Compare all (program, hardware) combinations, showing scalability with cores.
  """
  df['config'] = df['program'] + " @ " + df['hw']
  plt.figure(figsize=(10, 6))
  ax = sns.lineplot(
    data=df,
    x='cores', y='throughput_mops',
    hue='config',
    style='config',
    markers=True,
    estimator='mean', errorbar='sd'
  )
  plt.title("Scaling of All Program/Hardware Combinations")
  plt.ylabel("Throughput (Mops)")
  plt.xlabel("Cores")
  plt.grid(True)
  _set_common_plot_config(df, ax)
  plt.tight_layout()
  plt.savefig(dst, dpi=300)
  print(f'Plot saved to {dst}')
  plt.close()
    

def main():
  parser = get_basic_cli_parser()
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)

  if args.csv:
    print(f"Reading data from CSV file(s): {args.csv}")
    df = load_csv_files(args.csv)
  else:
    print("Generating data SbatchMan from jobs...")
    jobs = sbm.jobs_list(from_active=True, from_archived=False, status=[sbm.Status.COMPLETED])
    df = generate_dataframe_from_jobs(jobs)
    path = args.output_dir / f'{sbm.get_cluster_name()}_threads_sync.csv'
    df.to_csv(path, index=False)
    print(f"Saved dataframe to CSV: {path}")
    
  print(df)
  programs = sorted(df['program'].unique())
  hws = sorted(df['hw'].unique())
  hws_color_map = dict(zip(hws, COLORS_CYCLE))
  hws_linestyle_map = dict(zip(hws, LINESTYLES_CYCLE))
  hws_marker_map = dict(zip(hws, itertools.cycle(MARKERS_LIST)))

  for hw in hws:
    path: Path = args.output_dir / 'hardware' / f'thread_sync_{hw}.png'
    path.parent.mkdir(exist_ok=True, parents=True)
    compare_programs_same_hw(df, hw, path)
    
  for program in programs:
    path: Path = args.output_dir / 'primitives' / f'thread_sync_{program}.png'
    path.parent.mkdir(exist_ok=True, parents=True)
    compare_hw_same_program(df, program, path)
    
  compare_programs_hw_combinations(df, args.output_dir / 'thread_sync_full_comparison.png')

  print("✅ All plots generated.")


if __name__ == "__main__":
  main()