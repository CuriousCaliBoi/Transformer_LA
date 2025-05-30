import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A two-layer MLP with GELU activation and dropout.
    This implements the feed-forward network in the transformer architecture.
    
    The computation follows:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Where:
    - W1: First linear projection (expansion)
    - W2: Second linear projection (contraction)
    - b1, b2: Bias terms
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # First linear projection (expansion)
        self.fc1 = nn.Linear(d_model, d_ff)
        
        # GELU activation
        self.activation = nn.GELU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Second linear projection (contraction)
        self.fc2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Forward pass of the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            output: Transformed tensor of shape (batch_size, seq_len, d_model)
            mlp_info: Dictionary containing intermediate activations for analysis
        """
        # First linear projection
        # Shape: (batch_size, seq_len, d_ff)
        hidden = self.fc1(x)
        
        # Store pre-activation values for analysis
        pre_activation = hidden.clone()
        
        # Apply GELU activation
        hidden = self.activation(hidden)
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Second linear projection
        # Shape: (batch_size, seq_len, d_model)
        output = self.fc2(hidden)
        
        # Store MLP information for analysis
        mlp_info = {
            'pre_activation': pre_activation,
            'post_activation': hidden,
            'output': output
        }
        
        return output, mlp_info 