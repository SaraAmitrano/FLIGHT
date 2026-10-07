# noqa: N999
import pandas as pd


def transform_function(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: PLC0415

    primary_use_enc = LabelEncoder()
    primary_use_enc.fit(train_df["primary_use"])  # type: ignore
    train_df["primary_use"] = primary_use_enc.transform(train_df["primary_use"])  # type: ignore
    val_df["primary_use"] = primary_use_enc.transform(val_df["primary_use"])  # type: ignore
    test_df["primary_use"] = primary_use_enc.transform(test_df["primary_use"])  # type: ignore

    sea_level_pressure_scaler = StandardScaler()
    sea_level_pressure_scaler.fit(train_df[["sea_level_pressure"]])  # type: ignore
    train_df["sea_level_pressure"] = sea_level_pressure_scaler.transform(  # type: ignore
        train_df[["sea_level_pressure"]]
    )
    val_df["sea_level_pressure"] = sea_level_pressure_scaler.transform(  # type: ignore
        val_df[["sea_level_pressure"]]
    )
    test_df["sea_level_pressure"] = sea_level_pressure_scaler.transform(  # type: ignore
        test_df[["sea_level_pressure"]]
    )
    return train_df, val_df, test_df
