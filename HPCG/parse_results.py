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
import ccutils.ccutils_parser as ccutils_parser
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

def extract_metrics_dict(dp_section):
    dfs = {
        "dotp": [],
        "spmv_halo": [],
        "waxpby": [],
        "cg_times": [],
        "mg": [],
        "halo_precond": []
    }
    
    rank_outputs = dp_section.mpi_all_prints["ccutils_rank_json"].rank_outputs
    for rank, json_str in rank_outputs.items():
        parsed = json.loads(json_str)
        for iter_key in sorted(parsed.keys(), key=int):
            iter_data = parsed[iter_key]
            spmv_list = iter_data.get("spmv", [])
            halo_kernels = iter_data.get("halo_kernels", [])
            exchange_halo_list = iter_data.get("exchange_halo", [])
            halo_msg_sizes = iter_data.get("halo_msg_sizes", [])
            
            max_len = max(
                len(iter_data.get("dotp", [])),
                len(iter_data.get("dotp_allreduce", [])),
                len(iter_data.get("waxpby", [])),
                len(iter_data.get("mg", [])),
                1
            )
            # DOTP
            for idx in range(max_len):
                dfs["dotp"].append({
                    "rank": rank,
                    "iteration": int(iter_key),
                    "dotp": iter_data.get("dotp", [None]*max_len)[idx] if idx < len(iter_data.get("dotp", [])) else None,
                    "dotp_allreduce": iter_data.get("dotp_allreduce", [None]*max_len)[idx] if idx < len(iter_data.get("dotp_allreduce", [])) else None
                })
            # SPMV + HALO (only halo_kernel == "SPMV")
            spmv_counter = 0
            for halo_idx, kernel in enumerate(halo_kernels):
                if kernel == "SPMV":
                    dfs["spmv_halo"].append({
                        "rank": rank,
                        "iteration": int(iter_key),
                        "spmv": spmv_list[spmv_counter] if spmv_counter < len(spmv_list) else None,
                        "exchange_halo": exchange_halo_list[halo_idx] if halo_idx < len(exchange_halo_list) else None,
                        "halo_msg_size_bytes": halo_msg_sizes[halo_idx] if halo_idx < len(halo_msg_sizes) else None
                    })
                    spmv_counter += 1
            # MG (standalone)
            mg_list = iter_data.get("mg", [])
            for idx, val in enumerate(mg_list):
                dfs["mg"].append({
                    "rank": rank,
                    "iteration": int(iter_key),
                    "mg": val
                })
            # HALO (preconditioning only)
            for halo_idx, kernel in enumerate(halo_kernels):
                if kernel and "preconditioning_" in str(kernel):
                    dfs["halo_precond"].append({
                        "rank": rank,
                        "iteration": int(iter_key),
                        "exchange_halo": exchange_halo_list[halo_idx] if halo_idx < len(exchange_halo_list) else None,
                        "halo_msg_size_bytes": halo_msg_sizes[halo_idx] if halo_idx < len(halo_msg_sizes) else None
                    })
            # WAXPBY
            waxpby_list = iter_data.get("waxpby", [])
            for idx, val in enumerate(waxpby_list):
                dfs["waxpby"].append({
                    "rank": rank,
                    "iteration": int(iter_key),
                    "waxpby": val
                })
            # CG TIMES (one row per rank x iteration)
            dfs["cg_times"].append({
                "rank": rank,
                "iteration": int(iter_key),
                "cg_times": iter_data.get("cg_times", [None])[0]
            })
    # Convert lists to DataFrames
    for key in dfs:
        dfs[key] = pd.DataFrame(dfs[key]).reset_index(drop=True)
    
    return dfs
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
  
  return meta, extract_metrics_dict(cg_section)
  

def main():
    jobs = sbm.jobs_list(status=[sbm.Status.COMPLETED], from_active=True, from_archived=False)
    print(f"jobs[0]: {jobs[0]}")
    meta_df_pairs = [parse_job(j) for j in jobs]
    out_file = OUT_DIR / f'hpcg_{sbm.get_cluster_name()}_data.parquet'
    
    # Print original data
    print("\n=== ORIGINAL DATA (before write) ===")
    print(f"Number of jobs: {len(meta_df_pairs)}")
    print(f"\nFirst job metadata: {meta_df_pairs[0][0]}")
    print(f"\nDataframe keys: {list(meta_df_pairs[0][1].keys())}")
    for df_name, df in meta_df_pairs[0][1].items():
        print(f"\n{df_name} shape: {df.shape}")
        print(df.head())
    
    # Write to parquet
    import_export.write_multiple_to_parquet(meta_df_pairs, out_file)
    
    # Read back from parquet
    import_exported, metadata_df = import_export.read_multiple_from_parquet(out_file)
    
  #   # Print imported data #FIXME: remove this part, it was just to test the import/export
  #   print("\n=== IMPORTED DATA (after read) ===")
  #   print(f"Number of jobs: {len(import_exported)}")
  #   print(f"\nFirst job metadata: {import_exported[0][0]}")
  #   print(f"\nDataframe keys: {list(import_exported[0][1].keys())}")
  #   for df_name, df in import_exported[0][1].items():
  #       print(f"\n{df_name} shape: {df.shape}")
  #       print(df.head())
    
  #   # Verify metadata DataFrame
  #   if metadata_df is not None:
  #       print("\n=== METADATA DATAFRAME ===")
  #       print(metadata_df)
    
  #   # Test equality
  #   print("\n=== VERIFICATION ===")
  #   print(f"Same number of jobs: {len(meta_df_pairs) == len(import_exported)}")
  #   print(f"First metadata matches: {meta_df_pairs[0][0] == import_exported[0][0]}")
  #   for df_name in meta_df_pairs[0][1].keys():
  #       original = meta_df_pairs[0][1][df_name]
  #       imported = import_exported[0][1][df_name]
  #       matches = original.equals(imported)
  #       print(f"{df_name} DataFrames match: {matches}")
  #       if not matches:
  #           print(f"  Original shape: {original.shape}, Imported shape: {imported.shape}")
    
  #   # ADD THE DEBUGGING CODE HERE:
  #   print("\n=== DETAILED COMPARISON ===")
  #   for df_name in ["spmv_halo", "halo_precond"]:
  #       original = meta_df_pairs[0][1][df_name]
  #       imported = import_exported[0][1][df_name]
        
  #       print(f"\n{df_name}:")
  #       print(f"  Columns match: {list(original.columns) == list(imported.columns)}")
  #       print(f"  Original columns: {list(original.columns)}")
  #       print(f"  Imported columns: {list(imported.columns)}")
  #       print(f"  Dtypes match: {(original.dtypes == imported.dtypes).all()}")
  #       print(f"  Original dtypes:\n{original.dtypes}")
  #       print(f"  Imported dtypes:\n{imported.dtypes}")
        
  #       # Check for NaN differences
  #       print(f"  Original NaN count:\n{original.isna().sum()}")
  #       print(f"  Imported NaN count:\n{imported.isna().sum()}")
        
  #       # Check first few rows
  #       print(f"  Original head:\n{original.head()}")
  #       print(f"  Imported head:\n{imported.head()}")
  # # ADD THIS NEW SECTION HERE:
  #   print("\n=== LIST COMPARISON ===")
  #   for df_name in ["spmv_halo", "halo_precond"]:
  #       original = meta_df_pairs[0][1][df_name]
  #       imported = import_exported[0][1][df_name]
        
  #       print(f"\n{df_name}:")
  #       # Check if the lists are actually equal
  #       for idx in range(min(5, len(original))):
  #           orig_list = original.iloc[idx]['halo_msg_size_bytes']
  #           imp_list = imported.iloc[idx]['halo_msg_size_bytes']
  #           print(f"  Row {idx}: {orig_list} == {imp_list} -> {orig_list == imp_list}")
  #           print(f"    Type: original={type(orig_list)}, imported={type(imp_list)}")
  #           if isinstance(orig_list, list) and isinstance(imp_list, list):
  #               print(f"    List contents equal: {orig_list == imp_list}")
  
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