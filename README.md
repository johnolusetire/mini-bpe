# Mini BPE

A minimal implementation of Byte Pair Encoding (BPE) tokenization.

## Overview

This project provides a simple and educational implementation of the BPE algorithm commonly used in natural language processing and machine learning applications.

## Installation

```bash
git clone https://github.com/johnolusetire/mini-bpe.git
cd mini-bpe
```

## Usage

```python
# Basic usage example
from mini_bpe import BPE

# Initialize tokenizer
tokenizer = BPE()

# Train on text
tokenizer.train(text_data)

# Encode text
tokens = tokenizer.encode("Hello world!")

# Decode tokens
text = tokenizer.decode(tokens)
```

## Features

- Simple BPE implementation
- Easy to understand and modify
- Minimal dependencies

## License

MIT License