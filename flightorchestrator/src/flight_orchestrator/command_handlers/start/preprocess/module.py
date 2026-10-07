from pathlib import Path

from flight_common.job.dataset import Dataset
from flight_orchestrator.data_load.module import (
    DataframeDatasetData,
    DatasetData,
    ImageDatasetData,
)

from .partitioning import dirichlet_and_split, separate_data_class_and_split


def preprocess_data(
    dataset_data: DatasetData,
    clients_list: list[int],
    alpha: float,
    rng_seed: int,
    test_set_fraction: float,
    val_set_fraction: float,
    save_data_path: Path,
    save_data_folder: str,
    *,
    include_config_str_folder: bool = False,
) -> None:
    num_clients = len(clients_list)

    dataset = dataset_data.dataset

    preprocess_folder = save_data_path / dataset.value / save_data_folder
    if include_config_str_folder:
        config_str = f"N_{num_clients}__alpha_{alpha}__seed_{rng_seed}"
        preprocess_folder /= config_str
    preprocess_folder.mkdir(parents=True, exist_ok=True)

    if dataset in (Dataset.gefcom_2012, Dataset.ashrae):
        if not isinstance(dataset_data, DataframeDatasetData):
            raise TypeError(
                f"Expected DataframeDatasetData for {dataset.value}, "
                f"got {type(dataset_data)}"
            )

        dirichlet_and_split.partition_and_split(
            data_df=dataset_data.data_df,
            clients_list=clients_list,
            alpha=alpha,
            rng_seed=rng_seed,
            sort_columns=dataset_data.sort_columns,
            group_col=dataset_data.group_col,
            test_set_fraction=test_set_fraction,
            val_set_fraction=val_set_fraction,
            transform_function=dataset_data.transform_function,
            preprocess_folder=preprocess_folder,
        )
        return

    # For image datasets
    if not isinstance(dataset_data, ImageDatasetData):
        raise TypeError(
            f"Expected ImageDatasetData for {dataset.value}, got {type(dataset_data)}"
        )

    separate_data_class_and_split.partition_and_split(
        dataset_content=dataset_data.dataset_content,
        labels=dataset_data.labels,
        clients_list=clients_list,
        alpha=alpha,
        rng_seed=rng_seed,
        least_samples=dataset_data.least_samples,
        shuffle=dataset_data.shuffle,
        test_set_fraction=test_set_fraction,
        val_set_fraction=val_set_fraction,
        preprocess_folder=preprocess_folder,
    )
    return


__all__ = ["preprocess_data"]
