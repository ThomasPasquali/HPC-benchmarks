import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
import warnings
import numpy as np
import pandas as pd
import sbatchman as sbm
import graph500.ccutils.parser.ccutils_parser as ccutils_parser

sys.path.append(str(Path(__file__).parent.parent / "machines"))
from Leonardo.nodelists_generator import LeonardoNodelistGenerator
from HAICGU.nodes_map import HAICGUNodesMap

sys.path.append(str(Path(__file__).parent.parent))
import py_utils.import_export as import_export

OUT_DIR = Path('results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

NODES_MAP = None

def dict_get(d, key):
    r = d.get(key)
    if r is None:
        raise KeyError(f"{key} not found")
    return r


def raise_none(v, msg):
    if v is None:
        raise ValueError(f"{msg} not found")
    return v


def parse_job(j: sbm.Job, run_indices=range(64)) -> Tuple[Dict[Any, Any], pd.DataFrame]:
    """
    Returns: list of DataFrames with added jobid/run columns
    """
    stdout = raise_none(j.get_stdout(), "stdout")
    res = ccutils_parser.parse_ccutils_output(stdout)

    # Map rank → node number
    ranks_nodes_map = {}
    nodes = dict_get(res, "node_names")
    nodes = raise_none(nodes.get_mpi_print("node_names"), "node_names")

    for r in raise_none(nodes.get_all_ranks(), "nodes.get_all_ranks()"):
        node_str = raise_none(nodes.get_rank_output(r), f"node for rank {r}")
        if j.cluster_name == 'leonardo':
            ranks_nodes_map[r] = int(node_str.split(".")[0][4:])
        else:
            ranks_nodes_map[r] = node_str

    details = dict_get(res, "detailed_results")
    packet_bw = dict_get(details.mpi_all_prints, "packet_bandwidth")
    general = dict_get(res, "general_results").raw_text
    
    meta = {}
    vars = raise_none(j.variables, "job variables")
    for k in ['nodes', 'edgefactor', 'scale', 'partition']:
        meta[k] = vars[k]
    meta['buffer_size'] = vars['bin'].split('_')[-1]
    meta['cluster'] = j.cluster_name
    teps = -1
    for line in general.strip().splitlines():
        if 'harmonic_mean_TEPS' in line:
            line = re.subn(r'\s{2,}', ' ', line)[0]
            teps = float(line.split(' ')[-1])
            continue
    meta["teps"] = teps

    out_dfs = []
    for run_i in run_indices:
        rows = []
        if int(vars['nodes']) > 1:
            for dest in packet_bw.get_all_ranks():
                rank_output = packet_bw.get_rank_output(dest)
                if not rank_output:
                    continue

                for msg in rank_output.splitlines()[run_i].strip().split(" "):
                    if not msg: continue
                    src, size, t = msg.split(",")
                    distance = -1
                    if NODES_MAP:
                        distance = NODES_MAP.get_node_distance(
                            ranks_nodes_map[int(src)],
                            ranks_nodes_map[int(dest)],
                        )
                    rows.append([
                        int(src),
                        int(dest),
                        int(size),
                        float(t),
                        distance,
                    ])

        df = pd.DataFrame(rows, columns=["src", "dest", "size", "time", "distance"])
        df["distance"] = df["distance"].astype(np.int8)
        df["run"] = run_i

        # Clean negative times
        neg = df["time"] < 0
        df.loc[neg, "time"] = 0.0

        if not df.empty:
            out_dfs.append(df)

    return meta, pd.concat(out_dfs, ignore_index=True) if len(out_dfs) > 0 else pd.DataFrame()


def main():
    global NODES_MAP
    jobs = sbm.jobs_list(status=[sbm.Status.COMPLETED], from_active=True, from_archived=False)
    # FIXME this only checks job 0
    cluster_name = jobs[0].cluster_name
    
    if cluster_name == 'leonardo':
        NODES_MAP = LeonardoNodelistGenerator()
    if cluster_name == 'haicgu':
        NODES_MAP = HAICGUNodesMap()
    else:
        warnings.warn(f'No distance script found for cluster {cluster_name}')
        
    meta_df_pairs = [parse_job(j) for j in jobs]
    out_file = OUT_DIR / f'graph500_{cluster_name}_data.parquet'
    import_export.write_multiple_to_parquet(meta_df_pairs, out_file)

if __name__ == "__main__":
    main()
