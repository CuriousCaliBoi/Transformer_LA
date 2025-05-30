import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    """
    A causal self-attention mechanism that implements the core attention computation:
    Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
    
    This implementation includes:
    1. Query, Key, Value projections
    2. Scaled dot-product attention
    3. Causal masking to prevent attending to future tokens
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, dict]:
        """
        Forward pass of the attention mechanism.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Transformed tensor of shape (batch_size, seq_len, d_model)
            attention_info: Dictionary containing attention weights and other metrics
        """
        batch_size, seq_len, _ = x.shape
        
        # Project inputs to queries, keys, and values
        # Shape: (batch_size, seq_len, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        # Shape: (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask to prevent attending to future tokens
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.to(x.device)
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Apply softmax to get attention weights
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attn_weights, v)
        
        # Reshape and combine heads
        # Shape: (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.out_proj(context)
        
        # Store attention information for analysis
        attention_info = {
            'attention_weights': attn_weights,
            'query_key_scores': scores,
            'context': context
        }
        
        return output, attention_info 