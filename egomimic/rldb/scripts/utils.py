"""
useful functions and features
"""

import pandas as pd
import pyarrow.parquet as pq


def nds_pq(file_path):
    """
    Open a .parquet file and explore its structure, including nested datasets.
    """
    try:
        parquet_file = pq.ParquetFile(file_path)
        print(f"File Schema:\n{parquet_file.schema}\n")

        df = pd.read_parquet(file_path)

        print(f"Headers (Columns): {list(df.columns)}")
        print(f"Shape (Rows, Columns): {df.shape}")

        nested_columns = []
        for column in df.columns:
            # Check for nested data
            if isinstance(df[column].iloc[0], (dict, list)):
                nested_columns.append(column)

        if nested_columns:
            print(f"Nested Headers: {nested_columns}")
        else:
            print("No nested headers found.")
    except Exception as e:
        print(f"An error occurred: {e}")


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


def is_key(x):
    return hasattr(x, "keys") and callable(x.keys)


def is_listy(x):
    return isinstance(x, list)


def nds(nested_ds, tab_level=0):
    """
    Print the structure of a nested dataset.
    nested_ds: a series of nested dictionaries and iterables.  If a dictionary, print the key and recurse on the value.  If a list, print the length of the list and recurse on just the first index.  For other types, just print the shape.
    """
    # print('--' * tab_level, end='')
    if is_key(nested_ds):
        print("dict with keys: ", nested_ds.keys())
    elif is_listy(nested_ds):
        print("list of len: ", len(nested_ds))
    elif nested_ds is None:
        print("None")
    else:
        # print('\t' * (tab_level), end='')
        print(nested_ds.shape)

    if is_key(nested_ds):
        for key, value in nested_ds.items():
            print("\t" * (tab_level), end="")
            print(f"{key}: ", end="")
            nds(value, tab_level + 1)
    elif isinstance(nested_ds, list):
        print("\t" * tab_level, end="")
        print("Index[0]", end="")
        nds(nested_ds[0], tab_level + 1)
