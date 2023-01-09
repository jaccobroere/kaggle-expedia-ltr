from typing import Optional, Union, List
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


def subset_index(
    df: Union[pd.DataFrame, pd.Series],
    col: Union[str, int, List[Union[str, int]]],
    size: Optional[float] = 0.2,
    val_size: Optional[float] = None,
) -> Union[pd.DataFrame, pd.Series]:
    """Return a subset of the index of a Pandas DataFrame or Series."""

    idx, _ = next(
        GroupShuffleSplit(n_splits=1, test_size=(1 - size)).split(
            df, groups=df.loc[:, col]
        )
    )

    if val_size:
        val_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=(1 - val_size)).split(
                df.loc[~df.index.isin(idx)]
            )
        )

    return idx


if __name__ == "__main__":
    colname = "srch_id"
    df = read_columns("data/train.csv", colname)

    idx = subset_index(df, colname, size=0.2)
