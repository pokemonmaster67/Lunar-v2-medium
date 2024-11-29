# Lunar-v2-medium

A powerful and efficient medium-sized language model with 120M parameters (expandable to 150M). Built with state-of-the-art transformer architecture and optimized for both performance and efficiency.

## Features
- 120M parameters (base model)
- Expandable to 150M parameters
- Advanced attention mechanisms
- Efficient training and inference
- Optimized model architecture

## Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA compatible GPU (recommended)

## Installation
1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Architecture
- 12 transformer layers (base model)
- 12 attention heads
- 768 hidden dimensions
- Rotary positional embeddings
- Flash Attention support
- Sliding Window Attention

## Training
To train the model:
```bash
python train.py --config configs/base_config.json
```

## Inference
To run inference:
```bash
python inference.py --model_path checkpoints/lunar_v2_medium.pt
```
