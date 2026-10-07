from enum import Enum


class EvaluationMetrics(Enum):
    loss = "Loss"
    mse = "MSE"
    mae = "MAE"
    rmse = "RMSE"
    mape = "MAPE"
    r2 = "R2"
    cross_entropy = "CrossEntropy"
    focal = "Focal"
    accuracy = "Accuracy"
    f1score = "F1Score"
    precision = "Precision"
    recall = "Recall"
