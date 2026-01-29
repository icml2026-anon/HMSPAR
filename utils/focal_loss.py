import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Balancing factor for positive class (0-1)
            gamma (float): Focusing parameter (>= 0)
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits) [N]
            targets: Ground truth labels [N]
            
        Returns:
            loss: Focal loss value
        """
        
        p = torch.sigmoid(inputs)
        
        
        p_t = p * targets + (1 - p) * (1 - targets)
        
        
        focal_weight = (1 - p_t) ** self.gamma
        
        
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        
        focal_loss = alpha_t * focal_weight * bce
        
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
