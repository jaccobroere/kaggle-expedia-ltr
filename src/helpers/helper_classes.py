import pandas as pd
from typing import Optional
import os
import csv
from pandas.errors import EmptyDataError


def get_columns(fname: str) -> list:
    """Get the columns of a csv file. If the file does not exist, create it and return an empty list."""
    try:
        return pd.read_csv(fname, nrows=0).columns.tolist()
    except EmptyDataError:
        return []
    except FileNotFoundError:
        open(fname, "w").close()
        return []


class OutsideMemoryDF:
    """A class that allows you to read and write columns of a csv file without loading the entire file into memory."""

    def __init__(self, fname_write: str, fname_read: str):
        super().__init__()
        self.fname_write = fname_write
        self.fname_read = fname_read

    @property
    def columns_read(self):
        return get_columns(self.fname_read)

    @property
    def columns_write(self):
        return get_columns(self.fname_write)

    # def write_col(self, col):

    #     self.to_csv(self.fname_write, columns=[colname])

    def read_col(
        self,
        colname: Optional[str],
        use_idx: Optional[bool] = False,
        idx_col: Optional[int] = None,
    ):
        if use_idx:
            return pd.read_csv(self.fname_read, usecols=[idx_col])
        else:
            return pd.read_csv(self.fname_read, usecols=[colname])


if __name__ == "__main__":
    df = OutsideMemoryDF(fname_read="data/train.csv", fname_write="data/tryingoom.csv")
