"""
Feature-based Knowledge Distillation (Romero et al., 2015 - FitNets)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistiller(nn.Module):
    """Feature distillation con projection layer appresa"""
    
    def __init__(self, student_dim, teacher_dim, T=3.0, beta=0.1):
        super().__init__()
        self.T = T
        self.beta = beta
        
        # Projection layer (appresa durante training)
        self.projection = nn.Linear(student_dim, teacher_dim, bias=False)
    
    def forward(self, student_logits, teacher_logits, student_hidden, teacher_hidden):
        """
        Args:
            student_logits: [batch, seq_len, vocab]
            teacher_logits: [batch, seq_len, vocab]
            student_hidden: [batch, seq_len, student_dim]
            teacher_hidden: [batch, seq_len, teacher_dim]
        Returns:
            loss totale (KL + feature alignment)
        """
        # KL divergence sui logits
        p_s = F.log_softmax(student_logits / self.T, dim=-1)
        p_t = F.softmax(teacher_logits / self.T, dim=-1)
        kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        
        # Feature alignment
        student_projected = self.projection(student_hidden)
        feature_loss = F.mse_loss(student_projected, teacher_hidden)
        
        return kl_loss + self.beta * feature_loss