#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple
import sbatchman as sbm
import sys
import pandas as pd
import re
import sbatchman as sbm
from metrics import METRICS_TO_EXTRACT

sys.path.append(str(Path(__file__).parent.parent))
import ccutils.parser.ccutils_parser as ccutils_parser
import py_utils.import_export as import_export
from py_utils.utils.utils import raise_none, dict_get

OUT_DIR = Path('results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_hpcg_output(text: str) -> dict:
  """
  Parse an HPCG text file into a nested dict.
  If a node has both a scalar value and children, the scalar is stored under '_value'.
  """
  root = {}

  for raw in text.splitlines():
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

def extract_metrics_df(dp_section) -> pd.DataFrame:
  """
  Each row corresponds to one rank x iteration x measurement.
  Handles metrics with different lengths safely.
  """
  rows = []
  rank_outputs = dp_section.mpi_all_prints["ccutils_rank_json"].rank_outputs

  for rank, json_str in rank_outputs.items():
    parsed = json.loads(json_str)

    for iter_key in sorted(parsed.keys(), key=int):
      iter_data = parsed[iter_key]

      # Determine the maximum number of measurements among list metrics
      list_metrics = ["dotp", "dotp_allreduce", "exchange_halo", "halo_kernels", "halo_msg_sizes"]
      n = max(len(iter_data.get(m, [])) for m in list_metrics)
      n = max(n, 1)  # ensure at least one row

      for idx in range(n):
        row = {
          "rank": rank,
          "iteration": int(iter_key),
          "cg_time": iter_data.get("cg_time", [None])[0],
          "dotp": iter_data.get("dotp", [None]*n)[idx] if idx < len(iter_data.get("dotp", [])) else None,
          "dotp_allreduce": iter_data.get("dotp_allreduce", [None]*n)[idx] if idx < len(iter_data.get("dotp_allreduce", [])) else None,
          "spmv": iter_data.get("spmv", [None]*n)[idx] if "spmv" in iter_data and idx < len(iter_data.get("spmv", [])) else None,
          "mg": iter_data.get("mg", [None]*n)[idx] if "mg" in iter_data and idx < len(iter_data.get("mg", [])) else None,
          "waxpby": iter_data.get("waxpby", [None]*n)[idx] if "waxpby" in iter_data and idx < len(iter_data.get("waxpby", [])) else None,
          "exchange_halo": iter_data.get("exchange_halo", [None]*n)[idx] if idx < len(iter_data.get("exchange_halo", [])) else None,
          "halo_kernels": iter_data.get("halo_kernels", [None]*n)[idx] if idx < len(iter_data.get("halo_kernels", [])) else None,
          "halo_msg_size_bytes": iter_data.get("halo_msg_sizes", [None]*n)[idx] if idx < len(iter_data.get("halo_msg_sizes", [])) else None,
        }
        rows.append(row)

  return pd.DataFrame(rows)

def parse_job(j: sbm.Job) -> Tuple[Dict[Any, Any], pd.DataFrame]:
  stdout = raise_none(j.get_stdout(), "stdout")
  res = ccutils_parser.parse_ccutils_output(stdout)
  cg_section = dict_get(res, "cg")
  hpcg_section = dict_get(res, "hpcg_output")
  
  if not j.variables:
    raise Exception(f'job "{j}" has no variables')
  
  meta = {
    'partition': j.variables.get("partition", "unknown"),
    'nodes': dict_get(cg_section.json_data, "world_size"),
    "cluster": sbm.get_cluster_name()
  }
  parsed_hpcg_output = parse_hpcg_output(hpcg_section.raw_text)
  hpcg_metrics = collect_metrics(parsed_hpcg_output)
  hpcg_metrics_keys = [
    "threads", "total_cores", "global_nx", "global_ny", "global_nz",
    "num_equations", "gflops", "mem"
  ]
  for k in hpcg_metrics_keys:
    meta[k] = dict_get(hpcg_metrics, k)
  
  return meta, extract_metrics_df(cg_section)
  

def main():
  jobs = sbm.jobs_list(status=[sbm.Status.COMPLETED], from_active=True, from_archived=False)

  meta_df_pairs = [parse_job(j) for j in jobs]
  out_file = OUT_DIR / f'hpcg_{sbm.get_cluster_name()}_data.parquet'
  for meta, df in meta_df_pairs[:1]:
    print('-'*50)
    print(meta)
    print(df)
    df.to_csv('test_res.csv')
  import_export.write_multiple_to_parquet(meta_df_pairs, out_file)
  
  # rows = []
  # for fname in files:
  #   partition = os.path.basename(os.path.dirname(fname))  # subfolder name
  #   parsed = parse_hpcg_file(fname)
  #   metrics = collect_metrics(parsed)
  #   metrics["file"] = os.path.basename(fname)
  #   metrics["cluster"] = sbm.get_cluster_name()
  #   metrics["partition"] = partition
  #   rows.append(metrics)

  # df = pd.DataFrame(rows)
  # cols = ["partition", "file", "processes", "threads", "total_cores", 
  #         "global_nx", "global_ny", "global_nz", "global_points",
  #         "num_equations", "time_sec", "gflops"]
  # cols_present = [c for c in cols if c in df.columns]
  # df = df[cols_present + [c for c in df.columns if c not in cols_present]]

  # print(df)

  # # Save to CSV for later plotting
  # OUT_CSV=Path(args.out)
  # OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
  # df.to_csv(OUT_CSV, index=False)
  # print(f"Results saved to {OUT_CSV.absolute()}")


if __name__ == "__main__":
  main()