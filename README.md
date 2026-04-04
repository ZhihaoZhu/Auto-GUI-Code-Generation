# Auto-GUI-Code-Generation

Automatic GUI code generation for web, Android, and iOS, leveraging machine learning and template-based compilers. Based on the [pix2code](https://github.com/tonybeltramelli/pix2code) research by Tony Beltramelli.

The pipeline is: **Screenshot Image --> ResNet50 + Transformer --> DSL tokens --> Platform Compiler --> Native Code**

## Features

- **ResNet50 encoder + Transformer decoder** for screenshot-to-DSL generation (PyTorch)
- **Template-based code generation** for multiple platforms: Web (HTML/Bootstrap), Android (XML), iOS (Storyboard)
- PyTorch `Dataset` and `DataLoader` for efficient data loading and batching

## Directory Overview

```
compiler/               # Platform-specific code generators
  ├── android-compiler.py
  ├── ios-compiler.py
  ├── web-compiler.py
  ├── assets/           # DSL-to-platform mapping JSON files
  └── classes/          # Compiler internals (Compiler, Node, Utils)
model/                  # ML pipeline (PyTorch)
  ├── dataset.py        # Pix2CodeDataset + Vocabulary + DataLoaders
  ├── model.py          # Pix2CodeModel (ResNet50 encoder + Transformer decoder)
  ├── train.py          # Training loop with validation and checkpointing
  ├── generate.py       # Inference: screenshot -> .gui file
  └── classes/
      ├── Utils.py      # Image preprocessing with torchvision
      └── model/
          └── Config.py # Hyperparameters (d_model, n_heads, etc.)
tests/                  # Unit tests
  ├── test_dataset.py
  └── test_model.py
requirements.txt        # Python dependencies
```

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare dataset:** Place `.gui` + `.png` file pairs in a data directory (e.g., `datasets/web/all_data/`). Each `.gui` file contains DSL tokens and its matching `.png` is the corresponding screenshot.

3. **Train the model:**
   ```bash
   python -m model.train --data_dir datasets/web/all_data --epochs 10 --batch_size 64
   ```
   This saves the best checkpoint to `checkpoints/best_model.pt`.

4. **Generate DSL from a screenshot:**
   ```bash
   python -m model.generate --image screenshot.png --checkpoint checkpoints/best_model.pt
   ```
   This outputs DSL code to stdout, or to a file with `--output output.gui`.

5. **Compile DSL to platform code** (run from the `compiler/` directory):
   ```bash
   cd compiler
   python web-compiler.py <path_to_gui_file>
   python android-compiler.py <path_to_gui_file>
   python ios-compiler.py <path_to_gui_file>
   ```

## Model Architecture

```
Screenshot (256x256x3)
    |
    v
ResNet50 (frozen, ImageNet pretrained) -> 8x8x2048 spatial features
    |
    v
Linear projection -> 64 image tokens x 256 dims
    |
    v
Transformer Decoder (3 layers, 8 heads, d_model=256)
    |- Token Embedding + Sinusoidal Positional Encoding
    |- Causal Self-Attention
    |- Cross-Attention to image tokens
    |- Feed-Forward
    |
    v
Linear -> vocab_size logits -> next DSL token
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## License

[Apache 2.0](LICENSE)

## References

- Tony Beltramelli, [pix2code: Generating Code from a Graphical User Interface Screenshot](https://github.com/tonybeltramelli/pix2code), 2018.
