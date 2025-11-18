# CS182 Final Project

A comprehensive codebase for vision-language model experiments, supporting CLIP, Vision Transformers, and related architectures.

## Project Structure

```
182-Final-Project/
├── src/
│   ├── data/              # Data loading and preprocessing
│   │   ├── dataset.py     # Dataset classes
│   │   └── transforms.py  # Data augmentation
│   ├── models/            # Model architectures
│   │   ├── clip_model.py  # CLIP implementation
│   │   ├── vit_model.py   # Vision Transformer
│   │   └── model_factory.py
│   ├── training/          # Training utilities
│   │   ├── trainer.py     # Main trainer class
│   │   └── losses.py      # Loss functions
│   ├── evaluation/        # Evaluation utilities
│   │   ├── evaluator.py
│   │   ├── zero_shot.py   # Zero-shot evaluation
│   │   └── linear_probe.py
│   └── utils/             # Utilities
│       ├── config.py      # Config loading
│       ├── metrics.py     # Evaluation metrics
│       ├── tokenizer.py   # Text tokenization
│       └── visualization.py
├── config.yaml            # Main configuration file
├── experiments.yaml       # Experiment grid configuration
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── run_experiments.py    # Batch experiment runner
└── requirements.txt      # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Your Experiment

Edit `config.yaml` to set your experiment parameters:

```yaml
experiment:
  name: "my_experiment"
  seed: 42
  device: "cuda"  # or "cpu"

data:
  dataset: "cifar10"  # Options: cifar10, imagenet, custom
  data_dir: "./data"
  batch_size: 32
  image_size: 224

model:
  type: "clip"  # Options: clip, vit
  vision_model: "ViT-B/32"
  embed_dim: 512

training:
  epochs: 10
  learning_rate: 1e-4
  weight_decay: 0.01
```

## Usage

### Training

Train a model with the default configuration:

```bash
python train.py --config config.yaml
```

Resume training from a checkpoint:

```bash
python train.py --config config.yaml --resume checkpoints/checkpoint_epoch_5.pth
```

### Evaluation

Evaluate a trained model:

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pth
```

Run zero-shot evaluation:

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pth --zero_shot
```

Run linear probe evaluation:

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pth --linear_probe
```

### Running Multiple Experiments

1. Define your experiment grid in `experiments.yaml`:

```yaml
grid:
  "training.learning_rate": [1e-4, 5e-5, 1e-5]
  "data.batch_size": [32, 64]
  "model.temperature": [0.07, 0.1]
```

2. Run all experiments:

```bash
python run_experiments.py --config config.yaml --experiments experiments.yaml
```

## Model Architectures

### CLIP (Contrastive Language-Image Pre-training)

The CLIP model learns visual representations by contrasting image-text pairs. Supports:
- Pretrained OpenAI CLIP models
- Custom CLIP training from scratch
- Configurable vision and text encoders

### Vision Transformer (ViT)

A pure transformer architecture for image classification:
- Patch-based image encoding
- Multi-head self-attention
- Configurable depth and width

## Datasets

### Supported Datasets

- **CIFAR-10**: Automatically downloaded on first use
- **ImageNet**: Requires manual download and setup
- **Custom**: Provide a JSON file with image paths and text descriptions

### Custom Dataset Format

Create a JSON file with the following structure:

```json
{
  "images": ["path/to/image1.jpg", "path/to/image2.jpg", ...],
  "texts": ["description 1", "description 2", ...]
}
```

## Configuration

The configuration system supports:
- YAML-based configuration files
- Nested parameter organization
- Easy experiment parameterization
- Configurable model, data, and training settings

Key configuration sections:
- `experiment`: General experiment settings
- `data`: Dataset and data loading settings
- `model`: Model architecture and hyperparameters
- `training`: Training hyperparameters and settings
- `evaluation`: Evaluation metrics and methods
- `logging`: Logging and visualization settings

## Features

- **Flexible Model Architecture**: Support for CLIP, ViT, and custom models
- **Multiple Evaluation Modes**: Standard evaluation, zero-shot, and linear probe
- **Comprehensive Metrics**: Accuracy, top-k accuracy, F1 score
- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Checkpointing**: Save and resume training from checkpoints
- **Experiment Management**: Run multiple experiments with different configurations
- **Visualization**: Plot training curves and confusion matrices

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` in `config.yaml`
- Enable gradient accumulation: set `gradient_accumulation_steps > 1`
- Use a smaller model (e.g., `ViT-B/16` instead of `ViT-L/14`)

### Slow Training

- Enable mixed precision: set `mixed_precision: true`
- Increase `num_workers` for data loading
- Use a GPU if available

### Import Errors

- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're using the correct Python version (3.8+)
- Check that you're in the project root directory

## Citation

If you use this codebase in your research, please cite the relevant papers:

- CLIP: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021)
- Vision Transformer: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2020)

## License

This project is for educational purposes as part of CS182 coursework.
