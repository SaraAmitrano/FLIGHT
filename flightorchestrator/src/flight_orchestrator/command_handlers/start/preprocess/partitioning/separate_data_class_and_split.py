import logging
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore

from flight_common.numpy_types import IntArray, IntType

from .utils import save_client_bytes_data

logger = logging.getLogger(__name__)

StatisticData = tuple[int, int]


class SplitData(TypedDict):
    train: tuple[list[bytes], IntArray]
    val: tuple[list[bytes], IntArray]
    test: tuple[list[bytes], IntArray]


def train_val_test_split(
    X_client: list[bytes],  # noqa: N803
    y_client: list[int],
    val_set_fraction: float,
    test_set_fraction: float,
    rng_seed: int | None = None,
) -> SplitData:
    X_train_val, X_test, y_train_val, y_test = train_test_split(  # type: ignore  # noqa: N806
        X_client, y_client, test_size=test_set_fraction, random_state=rng_seed
    )

    X_train_val = cast("list[bytes]", X_train_val)  # noqa: N806
    y_train_val = cast("list[int]", y_train_val)
    X_test = cast("list[bytes]", X_test)  # noqa: N806
    y_test = np.array(y_test, dtype=IntType)

    X_train, X_val, y_train, y_val = train_test_split(  # type: ignore  # noqa: N806
        X_train_val, y_train_val, test_size=val_set_fraction, random_state=rng_seed
    )

    X_train = cast("list[bytes]", X_train)  # noqa: N806
    y_train = np.array(y_train, dtype=IntType)

    X_val = cast("list[bytes]", X_val)  # noqa: N806
    y_val = np.array(y_val, dtype=IntType)

    return SplitData(
        train=(X_train, y_train),
        val=(X_val, y_val),
        test=(X_test, y_test),
    )


def separate_data_class_map(
    labels: list[int],
    clients_list: list[int],
    alpha: float,
    least_samples: int,
    *,
    shuffle: bool,
    rng_seed: int | None = None,
) -> dict[int, list[int]]:
    num_clients = len(clients_list)
    dataset_label = np.array(labels)

    data_idx_map: dict[int, list[int]] = {}

    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = np.random.default_rng()
    min_size = 0
    K = len(set(labels))  # noqa: N806
    N = len(dataset_label)  # noqa: N806

    idx_batch: list[list[int]] = [[] for _ in range(num_clients)]
    while min_size < least_samples:
        for k in range(K):
            idx_k = np.where(dataset_label == k)[0]
            if shuffle:
                rng.shuffle(idx_k)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [
                    p * (len(idx_j) < N / num_clients)
                    for p, idx_j in zip(proportions, idx_batch, strict=True)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(
                    idx_batch, np.split(idx_k, proportions), strict=True
                )
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])
        logger.debug(f"Minimum number of samples per client: {min_size}")

    for i, j in enumerate(clients_list):
        data_idx_map[j] = idx_batch[i]

    return data_idx_map


def get_data_from_map(
    dataset_content: list[bytes],
    labels: list[int],
    clients_list: list[int],
    data_idx_map: dict[int, list[int]],
) -> tuple[
    dict[int, list[tuple[bytes, int]]],
    dict[int, list[StatisticData]],
]:
    Xy: dict[int, list[tuple[bytes, int]]] = {}  # noqa: N806
    statistic: dict[int, list[StatisticData]] = {}
    dataset_label = np.array(labels)

    for client in clients_list:
        statistic.setdefault(client, [])
        idxs = data_idx_map[client]
        # X[client] = dataset_content[idxs].copy()
        # X[client] = [dataset_content[i] for i in idxs]
        # y[client] = dataset_label[idxs].copy()
        # y[client] = [int(dataset_label[i]) for i in idxs]

        Xy_client = [  # noqa: N806
            (dataset_content[i], int(dataset_label[i])) for i in idxs
        ]

        Xy[client] = Xy_client

        y = np.array([v[1] for v in Xy_client], dtype=IntType)

        for i in np.unique(y):
            statistic[client].append((int(i), int(sum(y == i))))

    return Xy, statistic


def split_data(
    Xy: dict[int, list[tuple[bytes, int]]],  # noqa: N803
    test_set_fraction: float,
    val_set_fraction: float,
    rng_seed: int | None,
) -> dict[int, SplitData]:
    clients_data: dict[int, SplitData] = {}
    for i, client_data in Xy.items():
        X_client_, y_client_ = zip(*client_data, strict=True)  # noqa: N806
        X_client = cast("list[bytes]", list(X_client_))  # noqa: N806
        y_client = cast("list[int]", list(y_client_))

        if len(X_client) != len(y_client):
            raise ValueError(
                f"Client {i} data size mismatch: "
                f"{len(X_client)} samples, {len(y_client)} labels."
            )

        idx_train_val, idx_test = train_test_split(  # type: ignore
            list(range(len(X_client))),
            test_size=test_set_fraction,
            random_state=rng_seed,
        )

        idx_train_val = cast("list[int]", idx_train_val)
        idx_test = cast("list[int]", idx_test)

        X_train_val = [X_client[i] for i in idx_train_val]  # noqa: N806
        y_train_val = [y_client[i] for i in idx_train_val]
        X_test = [X_client[i] for i in idx_test]  # noqa: N806
        y_test = np.array([y_client[i] for i in idx_test], dtype=IntType)

        idx_train, idx_val = train_test_split(  # type: ignore
            list(range(len(idx_train_val))),
            test_size=val_set_fraction,
            random_state=rng_seed,
        )

        idx_train = cast("list[int]", idx_train)
        idx_val = cast("list[int]", idx_val)

        X_train = [X_train_val[i] for i in idx_train]  # noqa: N806
        y_train = np.array([y_train_val[i] for i in idx_train], dtype=IntType)

        X_val = [X_train_val[i] for i in idx_val]  # noqa: N806
        y_val = np.array([y_train_val[i] for i in idx_val], dtype=IntType)

        clients_data[i] = SplitData(
            train=(X_train, y_train),
            val=(X_val, y_val),
            test=(X_test, y_test),
        )

    return clients_data


def save_data(clients_data: dict[int, SplitData], preprocess_folder: Path) -> None:
    for client_id, data in clients_data.items():
        client_folder = preprocess_folder / f"{client_id}"
        client_folder.mkdir(parents=True, exist_ok=True)
        save_client_bytes_data(data, client_folder)


def partition_and_split(
    dataset_content: list[bytes],
    labels: list[int],
    clients_list: list[int],
    alpha: float,
    rng_seed: int | None,
    least_samples: int,
    *,
    shuffle: bool,
    test_set_fraction: float,
    val_set_fraction: float,
    preprocess_folder: Path | None,
) -> dict[int, SplitData]:
    logger.debug("Separating data by class")

    data_idx_map = separate_data_class_map(
        labels, clients_list, alpha, least_samples, shuffle=shuffle, rng_seed=rng_seed
    )

    Xy, _ = get_data_from_map(  # noqa: N806
        dataset_content, labels, clients_list, data_idx_map
    )
    logger.debug("Splitting data into train, validation, and test sets")
    clients_data = split_data(Xy, test_set_fraction, val_set_fraction, rng_seed)
    if preprocess_folder is not None:
        logger.debug("Saving data ...")
        save_data(clients_data, preprocess_folder)
    logger.debug("Data partitioning and splitting completed")
    return clients_data
