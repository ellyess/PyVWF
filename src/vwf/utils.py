"""Common utility functions for PyVWF."""
import pandas as pd


def ensure_numeric(df: pd.DataFrame, cols: list[str], errors: str = 'coerce') -> pd.DataFrame:
    """Coerce specified columns to numeric type.

    Args:
        df: Input DataFrame.
        cols: List of column names to convert.
        errors: How to handle conversion errors. Options:
            - 'coerce': Invalid values become NaN (default)
            - 'raise': Raise exception on invalid values
            - 'ignore': Leave invalid values as-is

    Returns:
        DataFrame with specified columns converted to numeric.

    Note:
        This function modifies the DataFrame in place and also returns it.
    """
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors=errors)
    return df
