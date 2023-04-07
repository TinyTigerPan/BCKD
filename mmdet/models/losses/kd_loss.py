from re import S
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..builder import LOSSES
from .utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def novel_kd_loss(pred,
                  soft_label,
                  detach_target=True,
                  beta=2.0):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    label = soft_label[1]
    pos = ((label >= 0) & (label < 80)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    
    target = soft_label[0].sigmoid()
    score = pred.sigmoid()
    
    t_pred = torch.argmax(target, dim=1).unsqueeze(1)
    t_c = target.gather(1, t_pred)
    s_c = score.gather(1, t_pred)
    ratio_c = t_c / s_c
    ratio_nc = (1 - t_c) / (1 - s_c)
    scale_s = score.scatter(1, t_pred, s_c * ratio_c / ratio_nc) * ratio_nc
    scale_s[pos].data = score[pos].data.clone()
    
    # if detach_target is True:
    target = target.detach()
    
    scale_factor = target - scale_s
    kd_loss = F.binary_cross_entropy(scale_s, target, reduction='none') * scale_factor.abs().pow(2.0)
    kd_loss = kd_loss.sum(dim=1, keepdim=False)
    return kd_loss


@LOSSES.register_module()
class NovelKDLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10, threshold=0.05):
        super(NovelKDLoss, self).__init__()
        # assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T
        self.threshold = threshold

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                beta=1.0):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * novel_kd_loss(pred, soft_label, weight, reduction=reduction, avg_factor=avg_factor, beta=beta)

        return loss_kd


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def knowledge_distillation_kl_div_loss(pred,
                                       soft_label,
                                       T,
                                       detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(
        F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (
                      T * T)

    return kd_loss


@LOSSES.register_module()
class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * knowledge_distillation_kl_div_loss(
            pred,
            soft_label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            T=self.T)

        return loss_kd


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def im_loss(x, soft_target):
    # print(x.shape, soft_target.shape)
    # print(F.mse_loss(x, soft_target))
    return F.mse_loss(x, soft_target)


@LOSSES.register_module()
class IMLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                x,
                soft_target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_im = self.loss_weight * im_loss(
            x, soft_target, reduction=reduction)

        return loss_im
