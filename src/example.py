import torch
from model import Transformer

def main():
    # Model hyperparameters
    vocab_size = 50257  # GPT-2 vocabulary size
    d_model = 256      # Embedding dimension
    num_heads = 8      # Number of attention heads
    num_layers = 6     # Number of transformer blocks
    d_ff = 1024       # Feed-forward dimension
    max_seq_len = 128  # Maximum sequence length
    
    # Create model
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    )
    
    # Create a sample input
    batch_size = 2
    seq_len = 16
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits, model_info = model(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Print some model information
    print("\nModel Information:")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of layers: {model.num_layers}")
    print(f"Embedding dimension: {model.d_model}")
    
    # Generate some text
    print("\nGenerating text...")
    generated = model.generate(x, max_new_tokens=10, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
    
    # Access attention information
    print("\nAttention Information:")
    for i, layer_info in enumerate(model_info['layer_info']):
        attn_weights = layer_info['attention']['attention_weights']
        print(f"Layer {i+1} attention weights shape: {attn_weights.shape}")

if __name__ == "__main__":
    main() 