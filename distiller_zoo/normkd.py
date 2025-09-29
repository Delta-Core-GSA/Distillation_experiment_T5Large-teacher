"""
Norm-based Knowledge Distillation
Based on "DistiLLM: Towards Streamlined Distillation for Large Language Models" (2023)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NormDistiller(nn.Module):
    """Norm-based distillation: allinea le norme degli hidden states"""
    
    def __init__(self, student_dim, teacher_dim, T=3.0, gamma=0.1):
        super().__init__()
        self.T = T
        self.gamma = gamma
        
        # Projection layer per allineare dimensioni
        self.projection = nn.Linear(student_dim, teacher_dim, bias=False)
    
    def forward(self, student_logits, teacher_logits, student_hidden, teacher_hidden):
        """
        Args:
            student_logits: [valid_positions, vocab_size]
            teacher_logits: [valid_positions, vocab_size]
            student_hidden: [valid_positions, student_dim]
            teacher_hidden: [valid_positions, teacher_dim]
        Returns:
            loss totale (KL + norm alignment)
        """
        # KL divergence sui logits
        p_s = F.log_softmax(student_logits / self.T, dim=-1)
        p_t = F.softmax(teacher_logits / self.T, dim=-1)
        kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        
        # Project student hidden states
        student_projected = self.projection(student_hidden)
        
        # Calcola norme L2 per ogni posizione
        student_norms = torch.norm(student_projected, p=2, dim=-1)  # [valid_positions]
        teacher_norms = torch.norm(teacher_hidden, p=2, dim=-1)     # [valid_positions]
        
        # Norm alignment loss
        norm_loss = F.mse_loss(student_norms, teacher_norms)
        
        return kl_loss + self.gamma * norm_loss