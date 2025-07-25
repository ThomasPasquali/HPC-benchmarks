import argparse
import pandas as pd

from pathlib import Path
from typing import List

from py_utils import OUT_DIR


def load_csv_files(filepaths: List[Path | str]) -> pd.DataFrame:
  dfs = []
  for file in filepaths:
    cluster = Path(file).stem.split("_")[0]
    df = pd.read_csv(file)
    df["cluster"] = cluster
    dfs.append(df)
  return pd.concat(dfs, ignore_index=True)


def get_basic_cli_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument('--output-dir', '-o', type=Path, default=OUT_DIR)
  parser.add_argument("--csv", '-f', nargs='+', type=Path, help="Optional CSV file(s) to read data from")
  
  return parser