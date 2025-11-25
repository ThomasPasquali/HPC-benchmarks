#!/usr/bin/env python3
"""
parse_results.py

Usage:
  python parse_results.py [folder]

Default folder: hpcg-out
Parses files matching: */HPCG-Benchmark_3.1_*.txt
Outputs: results/hpcg_results_{cluster_name}.csv
"""
import argparse
import glob
import os
from pathlib import Path
import pandas as pd
import re
import sbatchman as sbm
from metrics import METRICS_TO_EXTRACT

def parse_hpcg_file(filename: str) -> dict:
  """
  Parse an HPCG text file into a nested dict.
  If a node has both a scalar value and children, the scalar is stored under '_value'.
  """
  root = {}

  with open(filename, "r", encoding="utf-8") as f:
    for raw in f:
      line = raw.strip()
      if not line or line.startswith("#"):
        continue
      if "=" not in line:
        continue

      key_part, value_part = line.split("=", 1)
      key = key_part.strip()
      value_raw = value_part.strip()

      # normalize empty value -> None
      if value_raw == "":
        value = None
      else:
        # try int, then float, else keep string
        if re.match(r"^-?\d+$", value_raw):
          value = int(value_raw)
        else:
          try:
            value = float(value_raw)
          except ValueError:
            value = value_raw

      keys = [k.strip() for k in key.split("::")]

      # walk/inset into nested dict, converting scalars -> dicts as needed
      node = root
      for k in keys[:-1]:
        if k not in node:
          node[k] = {}
        elif not isinstance(node[k], dict):
            # convert scalar leaf into a dict, preserve old scalar under '_value'
          node[k] = {"_value": node[k]}
        node = node[k]

      last = keys[-1]
      if last in node:
        if isinstance(node[last], dict):
          # already has children -> keep scalar under '_value'
          node[last]["_value"] = value
        else:
          # existing scalar (rare): overwrite with new scalar value
          node[last] = value
      else:
        node[last] = value

  return root

def _get_section_value(parsed: dict, section: str, key_candidates):
  if not isinstance(parsed, dict):
    return None
  sec = parsed.get(section)
  if not isinstance(sec, dict):
    return None

  if isinstance(key_candidates, str):
      key_candidates = [key_candidates]
  for cand in key_candidates:
      if cand in sec:
        return sec[cand]

  lower_map = {k.lower(): k for k in sec.keys()}
  for cand in key_candidates:
      found = lower_map.get(cand.lower())
      if found:
        return sec[found]
  return None

def collect_metrics(parsed: dict) -> dict:
  out = {}

  processes = _get_section_value(parsed, "Machine Summary", ["Distributed Processes"])
  threads = _get_section_value(parsed, "Machine Summary", ["Threads per processes", "Threads per process"])
  try:
    out["processes"] = int(processes) if processes is not None else None
  except (ValueError, TypeError):
    out["processes"] = None
  try:
    out["threads"] = int(threads) if threads is not None else None
  except (ValueError, TypeError):
    out["threads"] = None

  if out.get("processes") is not None and out.get("threads") is not None:
    out["total_cores"] = out["processes"] * out["threads"]
  else:
    out["total_cores"] = None

  for m in METRICS_TO_EXTRACT:
    value = _get_section_value(parsed, m["section"], m["candidates"])
    out[m["out_key"]] = value

  final_section = parsed.get("Final Summary")
  if isinstance(final_section, dict):
    out["final_result_valid"] = final_section.get("Result") or \
                                final_section.get("HPCG result is VALID with a GFLOP/s rating of") or \
                                final_section.get("Results are valid but execution time (sec) is")
  else:
    out["final_result_valid"] = None

  return out

def main():
  parser = argparse.ArgumentParser(description="Parse HPCG outputs into a pandas DataFrame for scaling plots.")
  parser.add_argument("folder", nargs="?", default="hpcg-out", help="Root folder containing partition subfolders")
  parser.add_argument("--out", "-o", default=f"results/hpcg_results_{sbm.get_cluster_name()}.csv", help=f"CSV output path (default: results/hpcg_results_{sbm.get_cluster_name()}.csv)")
  args = parser.parse_args()

  pattern = os.path.join(args.folder, "*", "HPCG-Benchmark_3.1_*.txt")
  files = sorted(glob.glob(pattern))
  if not files:
    print(f"No files found matching {pattern}")
    return

  rows = []
  for fname in files:
    partition = os.path.basename(os.path.dirname(fname))  # subfolder name
    parsed = parse_hpcg_file(fname)
    metrics = collect_metrics(parsed)
    metrics["file"] = os.path.basename(fname)
    metrics["cluster"] = sbm.get_cluster_name()
    metrics["partition"] = partition
    rows.append(metrics)

  df = pd.DataFrame(rows)
  cols = ["partition", "file", "processes", "threads", "total_cores", 
          "global_nx", "global_ny", "global_nz", "global_points",
          "num_equations", "time_sec", "gflops"]
  cols_present = [c for c in cols if c in df.columns]
  df = df[cols_present + [c for c in df.columns if c not in cols_present]]

  print(df)

  # Save to CSV for later plotting
  OUT_CSV=Path(args.out)
  OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(OUT_CSV, index=False)
  print(f"Results saved to {OUT_CSV.absolute()}")


if __name__ == "__main__":
  main()