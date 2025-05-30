# Linear Algebra Transformer

A minimal decoder-only Transformer model implementation focused on understanding internal computations through linear algebra. This project analyzes how the model generates text by examining attention mechanisms and MLP transformations.

## Features

- Minimal decoder-only Transformer architecture (GPT-style)
- Causal self-attention implementation
- Detailed analysis of:
  - Query-key dot products
  - Softmax attention weights
  - Value aggregation
  - MLP transformations
- Visualization tools for attention patterns and matrix operations

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `src/`
  - `model/` - Transformer model implementation
  - `utils/` - Utility functions and visualization tools
  - `analysis/` - Tools for analyzing model internals
- `notebooks/` - Jupyter notebooks for experiments and analysis
- `tests/` - Unit tests

## Usage

[Usage instructions will be added as the project develops]

## License

MIT License 