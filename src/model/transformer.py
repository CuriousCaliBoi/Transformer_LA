import torch
import torch.nn as nn
from .transformer_block import TransformerBlock

class Transformer(nn.Module):
    """
    A minimal decoder-only transformer model (GPT-style) that focuses on
    understanding internal computations through linear algebra.
    
    The model consists of:
    1. Token embeddings
    2. Position embeddings
    3. Multiple transformer blocks
    4. Final layer normalization
    5. Output projection
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position embeddings
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights of the model."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Forward pass of the transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            model_info: Dictionary containing information about internal computations
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Create position indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        
        # Get embeddings
        token_emb = self.token_embedding(x)  # (batch_size, seq_len, d_model)
        pos_emb = self.position_embedding(pos)  # (seq_len, d_model)
        
        # Combine embeddings
        x = token_emb + pos_emb
        
        # Store information about each layer
        layer_info = []
        
        # Pass through transformer blocks
        for block in self.blocks:
            x, block_info = block(x)
            layer_info.append(block_info)
            
        # Final layer normalization
        x = self.norm(x)
        
        # Output projection
        logits = self.output(x)
        
        # Combine all information for analysis
        model_info = {
            'token_embeddings': token_emb,
            'position_embeddings': pos_emb,
            'layer_info': layer_info,
            'final_norm': x
        }
        
        return logits, model_info
    
    def generate(self, x: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate new tokens using the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated sequence of shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Get predictions
            logits, _ = self(x)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            x = torch.cat([x, next_token], dim=1)
            
        return x 