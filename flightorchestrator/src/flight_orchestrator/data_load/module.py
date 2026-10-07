import logging
from pathlib import Path

import pandas as pd

from flight_common.job.dataset import Dataset
from flight_common.job.task import Task

from . import datasets as datasets_module

logger = logging.getLogger(__name__)


class DatasetData:
    def __init__(
        self,
        dataset: Dataset,
        task: Task,
        original_data_path: Path,
        input_data_folder: str,
    ) -> None:
        self.dataset = dataset
        self.task = task
        self.input_path = original_data_path / dataset.value / input_data_folder


DATASET_DATA: dict[tuple[Dataset, Task], DatasetData] = {}


class DataframeDatasetData(DatasetData):
    def __init__(
        self,
        dataset: Dataset,
        task: Task,
        original_data_path: Path,
        input_data_folder: str,
    ) -> None:
        super().__init__(dataset, task, original_data_path, input_data_folder)

        if self.dataset not in [Dataset.gefcom_2012, Dataset.ashrae]:
            raise ValueError(
                "Dataset must be 'GEFCom2012' or 'ASHRAE' for DataframeDatasetData."
            )

        if self.task != Task.forecasting:
            raise ValueError(
                "Task must be 'forecasting' for GEFCom2012 or ASHRAE dataset."
            )

        logger.debug(f"Loading {dataset.value} dataset for task {task.value} ...")

        import importlib  # noqa: PLC0415

        dataset_module = importlib.import_module(
            f".{self.dataset.value}", datasets_module.__name__
        )
        transform_function = getattr(dataset_module, "transform_function", None)

        data_df = pd.read_feather(self.input_path / "data_clean.feather")

        orig_target_col = (
            "load_scaled"
            if self.dataset == Dataset.gefcom_2012
            else "meter_reading_scaled"
        )

        data_df = data_df.rename(columns={orig_target_col: "target"})

        group_col = "zone_id"

        # TODO remove
        if self.dataset == Dataset.ashrae:
            data_df = data_df.rename(columns={"site_id": group_col})

        sort_columns = (
            ["timestamp", group_col]
            if self.dataset == Dataset.gefcom_2012
            else ["timestamp", group_col, "meter", "primary_use"]
        )

        self.data_df = data_df
        self.sort_columns = sort_columns
        self.transform_function = transform_function
        self.group_col = "zone_id"


class ImageDatasetData(DatasetData):
    def __init__(
        self,
        dataset: Dataset,
        task: Task,
        original_data_path: Path,
        input_data_folder: str,
    ) -> None:
        super().__init__(dataset, task, original_data_path, input_data_folder)

        if dataset != Dataset.plant_village:
            raise ValueError("Dataset must be 'PlantVillage' for ImageDatasetData.")

        if task != Task.classification:
            raise ValueError("Task must be 'classification' for PlantVillage dataset.")

        logger.debug(f"Loading {dataset.value} dataset for task {task.value} ...")

        from .datasets.PlantVillage import load_processed_data  # noqa: PLC0415

        data_paths, dataset_content, labels = load_processed_data(self.input_path)

        least_samples = 10

        self.data_paths = data_paths
        self.dataset_content = dataset_content
        self.labels = labels
        self.least_samples = least_samples
        self.shuffle = False


DEFAULT_DATASET_DATA: list[tuple[Dataset, Task, type[DatasetData]]] = [
    (Dataset.gefcom_2012, Task.forecasting, DataframeDatasetData),
    (Dataset.ashrae, Task.forecasting, DataframeDatasetData),
    (Dataset.plant_village, Task.classification, ImageDatasetData),
]


def init_dataset_data(
    original_data_path: Path,
    input_data_folder: str,
    dataset_task_datas: list[
        tuple[Dataset, Task, type[DatasetData]]
    ] = DEFAULT_DATASET_DATA,
) -> None:
    DATASET_DATA.update(
        {
            (dataset, task): DatasetDataCls(
                dataset=dataset,
                task=task,
                original_data_path=original_data_path,
                input_data_folder=input_data_folder,
            )
            for dataset, task, DatasetDataCls in dataset_task_datas
        }
    )
