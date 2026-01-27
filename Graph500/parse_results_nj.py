import re
import sys
import argparse
import yaml
from pathlib import Path
from typing import Any, Dict, Tuple, List

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

import ccutils.parser.ccutils_parser as ccutils_parser
import py_utils.import_export as import_export
from py_utils.utils.utils import raise_none, dict_get

OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Filesystem Job Wrapper
# ----------------------------

class FSJob:
    def __init__(self, metadata_path: Path, stdout_path: Path, stderr_path: Path):
        self.metadata_path = metadata_path
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path

        with open(metadata_path) as f:
            self.variables = yaml.safe_load(f)

        # Infer cluster name from folder
        # graph500_nanjing-inter → nanjing-inter
        self.cluster_name = metadata_path.parent.name.replace("graph500_", "")

    def get_stdout(self) -> str:
        return self.stdout_path.read_text()

    def get_stderr(self) -> str:
        return self.stderr_path.read_text()


# ----------------------------
# Parser Logic (Mostly Same)
# ----------------------------

def parse_job(
    j: FSJob, run_indices=range(64)
) -> Tuple[Dict[Any, Any], Dict[str, pd.DataFrame]] | None:
    """
    Returns: (meta, {packets_df, barrier_df})
    """
    stdout = raise_none(j.get_stdout(), "stdout")
    res = ccutils_parser.parse_ccutils_output(stdout)

    for k in ['node_names', 'bfs_config', 'graph_stats', 'general_results', 'detailed_results']:
        if k not in res.keys():
            return None

    # ----------------------------
    # Rank → Node mapping
    # ----------------------------
    ranks_nodes_map = {}
    ranks_nodes_map_raw = {}

    nodes = dict_get(res, "node_names")
    nodes = raise_none(nodes.get_mpi_print("node_names"), "node_names")

    for r in raise_none(nodes.get_all_ranks(), "nodes.get_all_ranks()"):
        node_str = raise_none(nodes.get_rank_output(r), f"node for rank {r}")
        ranks_nodes_map_raw[r] = node_str

        if j.cluster_name == "leonardo":
            ranks_nodes_map[r] = int(node_str.split(".")[0][4:])
        else:
            ranks_nodes_map[r] = node_str

    # ----------------------------
    # Extract Results
    # ----------------------------
    details = dict_get(res, "detailed_results")
    packet_bw = dict_get(details.mpi_all_prints, "packet_bandwidth")
    barrier_times = dict_get(details.mpi_all_prints, "barrier_times")
    general = dict_get(res, "general_results").raw_text

    # ----------------------------
    # Metadata
    # ----------------------------
    meta = {}
    vars = j.variables

    for k in ["nodes", "edgefactor", "scale", "partition"]:
        meta[k] = vars[k]

    meta["buffer_size"] = vars["bin"].split("_")[-1]
    meta["cluster"] = j.cluster_name
    meta["rank_node_map"] = ranks_nodes_map_raw

    # ----------------------------
    # TEPS
    # ----------------------------
    teps = -1
    for line in general.strip().splitlines():
        if "harmonic_mean_TEPS" in line:
            line = re.subn(r"\s{2,}", " ", line)[0]
            teps = float(line.split(" ")[-1])
            break

    meta["teps"] = teps

    # ----------------------------
    # Build DataFrames
    # ----------------------------
    out_packets_dfs = []
    out_barrier_dfs = []

    for run_i in run_indices:
        rows = []

        if int(vars["nodes"]) > 1:
            for dest in packet_bw.get_all_ranks():
                rank_output = packet_bw.get_rank_output(dest)
                if not rank_output:
                    continue

                for msg in rank_output.splitlines()[run_i].strip().split(" "):
                    if not msg:
                        continue

                    src, size, t = msg.split(",")
                    rows.append(
                        [int(src), int(dest), int(size), float(t)]
                    )

            df = pd.DataFrame(rows, columns=["src", "dest", "size", "time"])
            df["run"] = run_i

            # Clean negative times
            neg = df["time"] < 0
            df.loc[neg, "time"] = 0.0

            if not df.empty:
                out_packets_dfs.append(df)

            # ----------------------------
            # Barrier
            # ----------------------------
            rows = []
            for rank in barrier_times.get_all_ranks():
                rank_output = barrier_times.get_rank_output(rank)
                if not rank_output:
                    continue

                for time in rank_output.strip().split(" "):
                    rows.append([int(rank), float(time)])

            df = pd.DataFrame(rows, columns=["rank", "time"])
            df["run"] = run_i

            if not df.empty:
                out_barrier_dfs.append(df)

    return meta, {
        "packets": (
            pd.concat(out_packets_dfs, ignore_index=True)
            if out_packets_dfs
            else pd.DataFrame()
        ),
        "barrier": (
            pd.concat(out_barrier_dfs, ignore_index=True)
            if out_barrier_dfs
            else pd.DataFrame()
        ),
    }


# ----------------------------
# Directory Scanner
# ----------------------------

def find_jobs(root: Path) -> List[FSJob]:
    jobs = []

    for metadata in root.rglob("metadata_*.yaml"):
        base = metadata.name.replace("metadata_", "").replace(".yaml", "")

        stdout = metadata.parent / f"stdout_{base}.txt"
        stderr = metadata.parent / f"stderr_{base}.txt"

        if not stdout.exists():
            print(f"⚠️ Missing stdout for {metadata}")
            continue

        if not stderr.exists():
            print(f"⚠️ Missing stderr for {metadata}")
            continue

        jobs.append(FSJob(metadata, stdout, stderr))

    print(f"Found {len(jobs)} jobs")
    return jobs


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Manual Graph500 parser (no sbatchman)")
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing nanjing-inter_* folders",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=64,
        help="Number of runs per job (default: 64)",
    )

    args = parser.parse_args()

    jobs = find_jobs(args.root)
    if not jobs:
        print("No jobs found.")
        return

    meta_df_pairs = [
        parse_job(j, run_indices=range(args.runs)) for j in jobs
    ]
    filtered_meta_df_pairs = []
    print('======= FAILED JOBS =======')
    for j, parsed in zip(jobs, meta_df_pairs):
        if parsed is None:
            print(j.variables)     
        else:
            filtered_meta_df_pairs.append(parsed)   
    print('======= END OF FAILED JOBS =======')


    cluster_name = jobs[0].cluster_name
    out_file = OUT_DIR / f"graph500_{cluster_name}_manual_data.parquet"

    import_export.describe_pairs_content(filtered_meta_df_pairs, verbose=False)
    import_export.write_multiple_to_parquet(filtered_meta_df_pairs, out_file)

    print(f"\n✅ Wrote: {out_file}")


if __name__ == "__main__":
    main()
