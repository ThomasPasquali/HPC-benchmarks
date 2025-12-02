import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_multiple_to_parquet(pairs: List[Tuple[Dict[Any, Any], pd.DataFrame]], path: Path):
    """
    Write a list of (metadata_dict, dataframe) pairs to a single Parquet file.
    Adds an internal 'pair_id' to link metadata to rows.
    """
    df_list = []
    metadata_list = []

    for i, (meta, df) in enumerate(pairs):
        meta_copy = dict(meta)
        meta_copy["_pair_id"] = i
        df_copy = df.copy()
        df_copy["_pair_id"] = i
        metadata_list.append(meta_copy)
        df_list.append(df_copy)

    # Concatenate all dataframes
    full_df = pd.concat(df_list, ignore_index=True)

    table = pa.Table.from_pandas(full_df)

    # Encode metadata_list as JSON → bytes
    json_bytes = json.dumps(metadata_list).encode("utf-8")
    existing_meta = table.schema.metadata or {}
    merged_meta = {**existing_meta, b"user_metadata_list": json_bytes}

    table = table.replace_schema_metadata(merged_meta)
    pq.write_table(table, path)
    print(f'Written {len(pairs)} metadata/dataframe pairs to "{path.resolve().absolute()}"')


def read_multiple_from_parquet(paths: List[Path] | Path) -> Tuple[List[Tuple[Any, pd.DataFrame]], pd.DataFrame | None]:
    """
    Read one or multiple Parquet files written by write_multiple_to_parquet().
    Returns:
        mapping: dict of {metadata_dict: corresponding DataFrame}
        metadata_df: optional DataFrame with metadata values if all metadata dicts
                     share the same keys and contain only scalar values
    """
    if isinstance(paths, Path):
        paths = [paths]

    combined_result = []
    seen_keys = set()
    metadata_records = []

    for path in paths:
        table = pq.read_table(path)
        df = table.to_pandas()

        schema_meta = table.schema.metadata or {}
        json_bytes = schema_meta.get(b"user_metadata_list")
        if json_bytes is None:
            continue

        metadata_list = json.loads(json_bytes.decode("utf-8"))

        for meta in metadata_list:
            pair_id = meta.pop("_pair_id")
            subset_df = df[df["_pair_id"] == pair_id].drop(columns=["_pair_id"])

            # Use hashable tuple for duplicate checking
            key_tuple = tuple(sorted(meta.items()))
            if key_tuple in seen_keys:
                warnings.warn(f"Duplicate metadata found in {path}: {meta}. Keeping first occurrence.")
                continue
            seen_keys.add(key_tuple)

            combined_result.append((meta, subset_df))
            metadata_records.append(meta)

    # Try building metadata DataFrame if all dicts share keys & scalar values
    metadata_df = None
    if metadata_records:
        keys = set(metadata_records[0].keys())
        scalar_only = all(
            set(d.keys()) == keys and all(not isinstance(v, (dict, list)) for v in d.values())
            for d in metadata_records
        )
        if scalar_only:
            metadata_df = pd.DataFrame(metadata_records)

    return combined_result, metadata_df
