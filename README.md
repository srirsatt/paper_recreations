# Paper recreations

My implementations of ideas from machine learning papers, written to understand how they work.

## Transformer

[`transformers/decoder.py`](transformers/decoder.py) implements a GPT-style, character-level language model in PyTorch, using transformer components from *Attention Is All You Need*.

- **Model:** Six decoder blocks with masked multi-head self-attention, feed-forward layers, residual connections, and layer normalization. Token and learned position embeddings represent the input characters.
- **LoRA experiment:** Rank-4 adapters on the query and key projections, with their base weights frozen.
- **Training:** Downloads Tiny Shakespeare, splits it into training and validation data, and learns to predict the next character from a 64-character context. Uses Apple Silicon's MPS backend when available, otherwise CPU.
- **Generation:** Includes a sampling loop to generate 500 characters after training.

The reference paper is saved as [`transformers/aiayn.pdf`](transformers/aiayn.pdf).
