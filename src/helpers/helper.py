from typing import Generator, Optional, Union, List
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import timeit
import csv


def read_columns(
    path: str, col: Optional[Union[str, int, List[Union[str, int]]]] = 0, **kwargs
) -> Union[pd.DataFrame, pd.Series]:
    """Read one or more columns from a CSV file.

    Parameters:
        path (str): The path to the CSV file.
        col (Union[str, int, List[Union[str, int]]]): The name or index of the column(s) to read. If a string or an integer, a single column is returned as a Pandas Series. If a list, multiple columns are returned as a Pandas DataFrame. Defaults to 0 (the first column).
        **kwargs: Additional keyword arguments to pass to the `read_csv` function.

    Returns:
        Union[pd.DataFrame, pd.Series]: The requested column(s) of the CSV file.

    Raises:
        KeyError: If the specified column(s) do not exist in the CSV file.
    """
    if isinstance(col, str) or isinstance(col, int):
        return pd.read_csv(path, usecols=[col], **kwargs)
    elif isinstance(col, list):
        return pd.read_csv(path, usecols=col, **kwargs)
    else:
        raise TypeError("'col' must be a string, integer, or list")


def subset_index(
    df: Union[pd.DataFrame, pd.Series],
    col: Union[str, int, List[Union[str, int]]],
    size: Optional[float] = 0.2,
    val_size: Optional[float] = None,
) -> Union[pd.Index, pd.Index]:
    """
    Return a subset of the index of a Pandas DataFrame or Series.

    Parameters
    ----------
    df: Union[pd.DataFrame, pd.Series]
        The dataframe or series to subset.
    col: Union[str, int, List[Union[str, int]]]
        The column or columns to use for grouping.
    size: Optional[float]
        The size of the subset to return, as a fraction of the input data.
        Default is 0.2.
    val_size: Optional[float]
        The size of the validation subset to return, as a fraction of the
        input data. If provided, the validation subset will be taken from
        the rows that are not in the training subset.

    Returns
    -------
    Union[pd.Index, pd.Index]
        The subset of the index of the input data. If 'val_size' is provided,
        returns a tuple with the training and validation subsets of the index.
    """

    idx, _ = next(
        GroupShuffleSplit(n_splits=1, test_size=(1 - size)).split(
            df, groups=df.loc[:, col]
        )
    )

    if val_size:
        if size is None:
            raise ValueError("'size' must be provided if 'val_size' is specified")

        val_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=(1 - val_size)).split(
                df.loc[~df.index.isin(idx)]
            )
        )
        return idx, val_idx
    else:
        return idx


def row_generator_idx(
    reader: csv.reader, idx: List[int]
) -> Generator[List[str], None, None]:
    """
    A generator that reads specific rows from a CSV file one at a time.

    Parameters
    ----------
    reader: csv.reader
        A CSV reader object.
    idx: List[int]
        A list of row indices to read from the file.

    Yields
    ------
    List[str]
        A row from the CSV file.
    """
    while True:
        try:
            while reader.line_num not in idx:
                next(reader)

            yield next(reader)

        except StopIteration:
            break


def read_csv_rows(
    file_path: str, idx: List[int], header: Optional[bool] = True
) -> pd.DataFrame:
    """
    Read specific rows from a CSV file and return a Pandas DataFrame.

    Parameters
    ----------
    file_path: str
        The path to the CSV file.
    idx: List[int]
        A list of row indices to read from the file.
    header: bool, optional
        Whether the first row of the CSV file should be used as the column names for the DataFrame. Default is True.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with the specified rows from the CSV file.
    """
    with open(file_path, "r") as f:
        reader = csv.reader(f)

        if header:
            df = pd.DataFrame(columns=next(reader))

        for i, row in enumerate(row_generator_idx(reader, idx)):
            if i == 0 and not header:
                df = pd.DataFrame(data=row)
                continue

            df = pd.concat([df, pd.DataFrame(data=row)])

    return df


if __name__ == "__main__":

    idx = [i for i in range(1000) if i % 4 == 0]
    path = "data/playground.csv"
    # measure the time it takes to run function 2
    time_2 = timeit.timeit(lambda: read_csv_rows(path, idx), number=1)
    print(time_2)

    df = read_csv_rows(path, idx)
