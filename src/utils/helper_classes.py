import pandas as pd
from typing import Optional
import os
import csv
from pandas.errors import EmptyDataError
import dask.dataframe as dd


def get_columns(fname: str) -> list:
    """Get the column names of a csv file. If the file does not exist, create it and return an empty list."""
    try:
        return pd.read_csv(fname, nrows=0).columns.tolist()
    except EmptyDataError:
        return []
    except FileNotFoundError:
        open(fname, "w").close()
        return []


class OutsideMemoryDF:
    """A class that allows you to read and write columns of a csv file without loading the entire file into memory.
    This is done by transposing the dataframe and writing the columns as rows."""

    def __init__(self, fname_write: str, fname_read: str):
        self.fname_write = fname_write
        self.fname_read = fname_read

    @property
    def columns_read(self):
        return get_columns(self.fname_read)

    @property
    def columns_write(self):
        return get_columns(self.fname_write)

    def write_column(self, df, col) -> None:
        with open(self.fname_write, "a") as f:
            writer = csv.writer(f)
            writer.writerow([col] + df[col].tolist())

    def read_col(
        self,
        colname: Optional[str],
        use_idx: Optional[bool] = False,
        idx_col: Optional[int] = None,
    ) -> pd.DataFrame:
        if use_idx:
            return pd.read_csv(self.fname_read, usecols=[idx_col])
        else:
            return pd.read_csv(self.fname_read, usecols=[colname])


if __name__ == "__main__":
    df = OutsideMemoryDF(fname_read="data/train.csv", fname_write="data/tryingoom2.csv")

    print("starting")
    a, b = df.read_col("price_usd"), df.read_col("srch_room_count")

    df.write_column(a, "price_usd")
