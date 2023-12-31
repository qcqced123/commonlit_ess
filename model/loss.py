import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Dict
from torch import Tensor


class SmoothL1Loss(nn.Module):
    """ Smooth L1 Loss in Pytorch """
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true) -> Tensor:
        criterion = nn.SmoothL1Loss(reduction=self.reduction)
        return criterion(y_pred, y_true)


class MSELoss(nn.Module):
    """ Mean Squared Error Loss in Pytorch """
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true) -> Tensor:
        criterion = nn.MSELoss(reduction=self.reduction)
        return criterion(y_pred, y_true)


class RMSELoss(nn.Module):
    def __init__(self, reduction: str = 'mean', eps=1e-8) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.eps = eps # If MSE == 0, We need eps

    def forward(self, yhat, y) -> Tensor:
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class MCRMSELoss(nn.Module):
    """
    Mean Column-wise Root Mean Squared Error Loss, which is used in this competition
    Calculate RMSE per target values(columns), and then calculate mean of each column's RMSE Result
    Args:
        reduction: str, reduction method of loss
        num_scored: int, number of scored target values
    """
    def __init__(self, reduction: str, num_scored: int = 1) -> None:
        super().__init__()
        self.RMSE = RMSELoss(reduction=reduction)
        self.num_scored = num_scored

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score = score + (self.RMSE(yhat[:, i], y[:, i]) / self.num_scored)
        return score


# Weighted MCRMSE Loss => Apply different loss rate per target classes
class WeightMCRMSELoss(nn.Module):
    """
    Apply loss rate per target classes for using Meta Pseudo Labeling
    Weighted Loss can transfer original label data's distribution to pseudo label data
    0.36 vs 0.64
    [Reference]
    https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609
    """
    def __init__(self, reduction, num_scored=2):
        super().__init__()
        self.RMSE = RMSELoss(reduction=reduction)
        self.num_scored = num_scored
        self._loss_rate = torch.tensor([0.21, 0.16, 0.10, 0.16, 0.21, 0.16], dtype=torch.float32)

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score = score + torch.mul(self.RMSE(yhat[:, i], y[:, i]), self._loss_rate[i])
        return score


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss
    [Reference]
    https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369793
    """
    def __init__(self, reduction, task_num=1) -> None:
        super(WeightedMSELoss, self).__init__()
        self.task_num = task_num
        self.smoothl1loss = nn.SmoothL1Loss(reduction=reduction)

    def forward(self, y_pred, y_true, log_vars) -> float:
        loss = 0
        for i in range(self.task_num):
            precision = torch.exp(-log_vars[i])
            diff = self.smoothl1loss(y_pred[:, i], y_true[:, i])
            loss += torch.sum(precision * diff + log_vars[i], -1)
        loss = 0.5*loss
        return loss
