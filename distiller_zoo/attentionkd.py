"""
Attention-based Knowledge Distillation (Zagoruyko & Komodakis, 2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDistiller(nn.Module):
    """Attention transfer distillation"""
    
    def __init__(self, T=3.0, gamma=0.1):
        super().__init__()
        self.T = T
        self.gamma = gamma
    
    def forward(self, student_logits, teacher_logits, student_attn, teacher_attn):
        """
        Args:
            student_logits: [batch, seq_len, vocab]
            teacher_logits: [batch, seq_len, vocab]
            student_attn: [batch, num_heads, seq_len, seq_len]
            teacher_attn: [batch, num_heads, seq_len, seq_len]
        Returns:
            loss totale (KL + attention alignment)
        """
        # KL divergence
        p_s = F.log_softmax(student_logits / self.T, dim=-1)
        p_t = F.softmax(teacher_logits / self.T, dim=-1)
        kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        
        # Attention alignment (media su heads se diversi)
        if student_attn.size(1) != teacher_attn.size(1):
            student_attn = student_attn.mean(dim=1)
            teacher_attn = teacher_attn.mean(dim=1)
        
        attn_loss = F.mse_loss(student_attn, teacher_attn)
        
        return kl_loss + self.gamma * attn_loss