import torch
import torch.nn as nn
from .attention import CausalSelfAttention
from .mlp import MLP

class TransformerBlock(nn.Module):
    """
    A single transformer block that combines:
    1. Multi-head self-attention
    2. MLP
    3. Layer normalization
    4. Residual connections
    
    The computation follows:
    x' = x + Attention(LayerNorm(x))
    output = x' + MLP(LayerNorm(x'))
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Layer normalization for attention
        self.norm1 = nn.LayerNorm(d_model)
        
        # Multi-head self-attention
        self.attention = CausalSelfAttention(d_model, num_heads, dropout)
        
        # Layer normalization for MLP
        self.norm2 = nn.LayerNorm(d_model)
        
        # MLP
        self.mlp = MLP(d_model, d_ff, dropout)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, dict]:
        """
        Forward pass of the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Transformed tensor of shape (batch_size, seq_len, d_model)
            block_info: Dictionary containing attention and MLP information
        """
        # Attention block
        residual = x
        x = self.norm1(x)
        attn_output, attn_info = self.attention(x, mask)
        x = residual + self.dropout(attn_output)
        
        # MLP block
        residual = x
        x = self.norm2(x)
        mlp_output, mlp_info = self.mlp(x)
        x = residual + self.dropout(mlp_output)
        
        # Combine information for analysis
        block_info = {
            'attention': attn_info,
            'mlp': mlp_info,
            'pre_norm1': residual,
            'pre_norm2': residual
        }
        
        return x, block_info 