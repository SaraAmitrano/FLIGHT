from enum import Enum


class LossFunction(Enum):
    mse = "MSE"
    mae = "MAE"
    cross_entropy = "CrossEntropy"
    focal_loss = "FocalLoss"
