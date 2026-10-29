import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore

from flight_common.numpy_types import IntArray, IntType

from .utils import save_client_dataframe_data

logger = logging.getLogger(__name__)


class SplitData(TypedDict):
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


# Receives the splits of all clients at once (keyed by client index)
TransformFunction = Callable[[dict[int, SplitData]], dict[int, SplitData]]


def dirichlet_integer_partition(
    K: int,  # noqa: N803
    N: int,  # noqa: N803
    alpha: float,
    rng: int | np.random.Generator | None = None,
) -> IntArray:
    """
    Distribute K indistinguishable items among N distinguishable containers
    using a Dirichlet distribution.

    Parameters
    ----------
    K     : total number of items (integer, K > N)
    N     : total number of containers (integer, N > 0)
    alpha : list of Dirichlet parameters, each in (0,1), length N
    rng   : random number generator or seed (optional)

    """
    alphas = np.repeat([alpha], N, axis=0)

    # Step 1: Sample from Dirichlet
    if rng is None:
        rng_ = np.random.default_rng()
    elif isinstance(rng, int):
        rng_ = np.random.default_rng(rng)
    else:
        rng_ = rng

    p = rng_.dirichlet(alphas)  # p is an array of length N, sum(p)=1

    # Step 2: Distribute the extra (K-N)
    K_minus_N = K - N  # noqa: N806

    # 2a. Multiply p_i by (K-N)
    y = p * K_minus_N  # real-valued array

    # 2b. Base counts via floor
    x = np.floor(y).astype(IntType)

    # 2c. Leftover
    leftover = K_minus_N - np.sum(x)

    if leftover > 0:
        # 2d. Distribute leftover among largest fractional parts
        frac = y - x  # fractional parts
        # Sort indices by descending fractional part
        idx_sorted = np.argsort(-frac)  # negative for descending
        for i in range(leftover):
            x[idx_sorted[i]] += 1

    # Step 3: Add 1 to each container to avoid zeros,
    # so that it sums to K, each entry >= 1
    return x + 1


def train_test_split_time(
    data_df: pd.DataFrame, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamp_test: pd.Series[pd.Timestamp] = train_test_split(  # type: ignore
        data_df["timestamp"], test_size=test_size, shuffle=False
    )[1]

    timestamp_begin_test: pd.Timestamp = timestamp_test.min().replace(  # type: ignore
        hour=0
    ) + pd.to_timedelta(  # type: ignore
        "1 days"
    )

    train_df = data_df[data_df["timestamp"] < timestamp_begin_test].copy()
    test_df = data_df[data_df["timestamp"] >= timestamp_begin_test].copy()
    return train_df, test_df


def train_val_test_split(
    data_df: pd.DataFrame, val_set_fraction: float, test_set_fraction: float
) -> SplitData:
    """
    Split data into train, validation, and test sets.
    """
    train_val_df, test_df = train_test_split_time(data_df, test_set_fraction)
    train_df, val_df = train_test_split_time(train_val_df, val_set_fraction)

    return SplitData(
        train=train_df,
        val=val_df,
        test=test_df,
    )


def separate_data_class(
    data_df: pd.DataFrame,
    clients_list: list[int],
    alpha: float,
    group_col: str,
    rng_seed: int | None = None,
) -> tuple[list[pd.DataFrame], IntArray]:
    K = len(data_df[group_col].unique())  # type: ignore  # noqa: N806
    N = len(clients_list)  # noqa: N806
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = np.random.default_rng()

    counts = dirichlet_integer_partition(K, N, alpha, rng)

    data_groups_ids = np.sort(data_df[group_col].unique())  # type: ignore
    data_groups_ids = rng.permutation(data_groups_ids)  # type: ignore

    data_groupbys = data_df.groupby(group_col, observed=False)  # type: ignore
    data_groups = [data_groupbys.get_group(group_id) for group_id in data_groups_ids]  # type: ignore

    return data_groups, counts


def split_data(
    data_groups: list[pd.DataFrame],
    counts: IntArray,
    val_set_fraction: float,
    test_set_fraction: float,
    sort_columns: list[str],
    group_col: str,
    transform_function: TransformFunction | None = None,
) -> dict[int, SplitData]:
    clients_data: dict[int, SplitData] = {}
    for i, count in enumerate(counts):
        data_zones_client = data_groups[:count]
        data_groups = data_groups[count:]
        data_client = pd.concat(data_zones_client)

        data_client = data_client.sort_values(  # type: ignore
            sort_columns
        )

        out = train_val_test_split(data_client, val_set_fraction, test_set_fraction)

        train_df = out["train"]
        val_df = out["val"]
        test_df = out["test"]

        col_counts = {group_col: count}

        for col, count_col in col_counts.items():
            if (
                data_client[col].nunique() != count_col
                or train_df[col].nunique() != count_col
                or val_df[col].nunique() != count_col
                or test_df[col].nunique() != count_col
            ):
                raise ValueError(
                    f"Column '{col}' does not have the expected number of "
                    f"unique values ({count_col}) in client {i} data splits."
                )

        clients_data[i] = SplitData(train=train_df, val=val_df, test=test_df)

    # Applied once on all clients, so fitted encoders/scalers are shared
    if transform_function is not None:
        clients_data = transform_function(clients_data)

    return clients_data


def save_data(clients_data: dict[int, SplitData], preprocess_folder: Path) -> None:
    for client_id, data_df in clients_data.items():
        client_folder = preprocess_folder / f"{client_id}"
        client_folder.mkdir(parents=True, exist_ok=True)
        save_client_dataframe_data(data_df, client_folder)


def partition_and_split(
    data_df: pd.DataFrame,
    clients_list: list[int],
    alpha: float,
    rng_seed: int | None,
    test_set_fraction: float,
    val_set_fraction: float,
    preprocess_folder: Path | None,
    sort_columns: list[str],
    group_col: str,
    transform_function: TransformFunction | None = None,
) -> dict[int, SplitData]:
    logger.debug("Separating data by class")
    data_groups, counts = separate_data_class(
        data_df, clients_list, alpha, group_col, rng_seed
    )
    logger.debug("Splitting data into train, validation, and test sets")
    clients_data = split_data(
        data_groups,
        counts,
        val_set_fraction,
        test_set_fraction,
        sort_columns,
        group_col,
        transform_function,
    )
    if preprocess_folder is not None:
        logger.debug("Saving data ...")
        save_data(
            {clients_list[i]: data for i, data in clients_data.items()},
            preprocess_folder,
        )
    logger.debug("Data partitioning and splitting completed")
    return clients_data
