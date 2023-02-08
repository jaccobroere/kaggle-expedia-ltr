from typing import List, Optional, Union

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


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
    if size == 1:
        return df.index

    df["aux_index"] = df.index

    idx, _ = next(
        GroupShuffleSplit(n_splits=1, test_size=(1 - size)).split(
            df, groups=df.loc[:, col]
        )
    )
    res_idx = df.loc[idx, "aux_index"]

    if not val_size:
        return res_idx

    df_val = df.loc[~df.index.isin(res_idx), :].reset_index()
    val_idx, _ = next(
        GroupShuffleSplit(n_splits=1, test_size=(1 - val_size / (1 - size))).split(
            df_val, groups=df_val.loc[:, col]
        )
    )
    res_val_idx = df_val.loc[val_idx, "aux_index"]

    return res_idx, res_val_idx


if __name__ == "__main__":
    pass
