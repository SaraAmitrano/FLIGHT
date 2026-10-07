# noqa: N999
import pandas as pd


def transform_function(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return train_df, val_df, test_df
