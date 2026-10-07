from typing import Literal

from pydantic import BaseModel

CustomType = Literal["Custom"]

CommonLearningAlgorithm = Literal[
    "GEFCom2012Dense", "GEFCom2012LSTM", "ASHRAEDense", "ASHRAELSTM"
]

BaseLearningAlgorithm = Literal[CommonLearningAlgorithm, "PlantVillageVGG16"]

CustomizableLearningAlgorithm = CommonLearningAlgorithm

LearningAlgorithm = Literal[BaseLearningAlgorithm, CustomizableLearningAlgorithm]


class CustomLearningAlgorithmParams(BaseModel):
    base_model: CustomizableLearningAlgorithm
    model_code: str


class CustomLearningAlgorithm(BaseModel):
    type: CustomType = "Custom"
    params: CustomLearningAlgorithmParams
