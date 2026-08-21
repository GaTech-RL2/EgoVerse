"""Small helpers for inspecting nested datasets and parsing CLI booleans."""

from __future__ import annotations

import argparse
from numbers import Number

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


def is_key(value):
    return hasattr(value, "keys") and callable(value.keys)


def is_listy(value):
    return isinstance(value, list)


def nds_pq(file_path):
    """Print a parquet file's schema, columns, shape, and nested columns."""
    try:
        parquet_file = pq.ParquetFile(file_path)
        print(f"File Schema:\n{parquet_file.schema}\n")
        df = pd.read_parquet(file_path)
        print(f"Headers (Columns): {list(df.columns)}")
        print(f"Shape (Rows, Columns): {df.shape}")
        nested_columns = [
            column
            for column in df.columns
            if isinstance(df[column].iloc[0], (dict, list))
        ]
        if nested_columns:
            print(f"Nested Headers: {nested_columns}")
        else:
            print("No nested headers found.")
    except Exception as exc:
        print(f"An error occurred: {exc}")


nested_ds_pq = nds_pq
nds_parquet = nds_pq
nested_ds_parquet = nds_pq


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def nds(nested_ds, tab_level=0):
    """Print the structure of nested dictionaries, lists, and arrays."""
    if is_key(nested_ds):
        print("dict with keys: ", nested_ds.keys())
    elif is_listy(nested_ds):
        print("list of len: ", len(nested_ds))
    elif nested_ds is None:
        print("None")
    elif isinstance(nested_ds, Number):
        print("Number: ", nested_ds)
    elif isinstance(nested_ds, (np.ndarray, torch.Tensor)):
        print(nested_ds.shape)
    else:
        print("Type: ", type(nested_ds))

    if is_key(nested_ds):
        for key, value in nested_ds.items():
            print("\t" * tab_level, end="")
            print(f"{key}: ", end="")
            nds(value, tab_level + 1)
    elif is_listy(nested_ds):
        print("\t" * tab_level, end="")
        print("Index[0]", end="")
        nds(nested_ds[0], tab_level + 1)
