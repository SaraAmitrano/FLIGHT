# noqa: N999
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from flight_orchestrator.command_handlers.start.preprocess.partitioning.dirichlet_and_split import (  # noqa: E501
        SplitData,
    )

SPLITS = ("train", "val", "test")


def transform_function(
    clients_data: "dict[int, SplitData]",
) -> "dict[int, SplitData]":
    """
    Encode and scale ASHRAE features consistently across all clients.

    The encoder and the scaler are fitted once on the union of the clients'
    train splits, so the same category code / scaled value has the same meaning
    on every client (required for a shared global model).
    """
    from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: PLC0415

    train_all = pd.concat(
        [data["train"][["primary_use", "sea_level_pressure"]] for data in clients_data.values()]
    )

    primary_use_enc = LabelEncoder()
    primary_use_enc.fit(train_all["primary_use"])  # type: ignore

    sea_level_pressure_scaler = StandardScaler()
    sea_level_pressure_scaler.fit(train_all[["sea_level_pressure"]])  # type: ignore

    for data in clients_data.values():
        for split in SPLITS:
            df = data[split]
            df["primary_use"] = primary_use_enc.transform(df["primary_use"])  # type: ignore
            df["sea_level_pressure"] = sea_level_pressure_scaler.transform(  # type: ignore
                df[["sea_level_pressure"]]
            )
    return clients_data
