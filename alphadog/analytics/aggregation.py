import pandas as pd


def cross_sectional_mean(df, name='combined'):
    """
    Aggregate multiple columns by taking the cross-sectional mean.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to aggregate.
    name: str, optional.
        Column name for the result DataFrame.

    Returns
    -------
    pd.DataFrame
        Single column DataFrame which is the mean of the input DataFrame
    """
    df_res = df.copy()
    if isinstance(df, pd.Series):
        df_res = df.to_frame(name)
    return df_res.mean(axis=1).to_frame(name)
