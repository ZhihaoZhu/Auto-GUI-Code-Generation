# PyTorch Rewrite: ResNet50 + Transformer pix2code Pipeline

## Context

The Auto-GUI-Code-Generation project converts GUI screenshots into platform-specific code (HTML, Android XML, iOS Storyboard) via a DSL intermediate representation. The original ML pipeline used TensorFlow 1.x and was never completed — only feature extraction and data preprocessing scaffolding existed. This rewrite builds the complete pipeline in PyTorch with a ResNet50 encoder and Transformer decoder architecture.

## Architecture

```
Screenshot (256x256x3)
    |
    v
ResNet50 (frozen, pretrained ImageNet) -> (8x8x2048) spatial features
    |
    v
Linear projection -> (64 tokens x d_model=256) "image tokens"
    |
    v
Transformer Decoder (3 layers, 8 heads, d_model=256)
    |- Token Embedding + Sinusoidal Positional Encoding
    |- Masked Self-Attention (causal)
    |- Cross-Attention to image tokens
    |- Feed-Forward
    |
    v
Linear -> vocab_size logits -> next token prediction
```

## File Structure

```
model/
  dataset.py          # PyTorch Dataset + DataLoader + vocabulary
  model.py            # Pix2CodeModel (ResNet50 encoder + Transformer decoder)
  train.py            # Training loop with validation
  generate.py         # Inference: screenshot -> .gui file
  classes/
    Utils.py          # Rewritten with torchvision transforms
    model/
      Config.py       # Updated hyperparameters
requirements.txt      # Modern PyTorch dependencies
```

Files removed: `Extract_features.py` (replaced by encoder in `model.py`).

## Components

### 1. Dataset (`dataset.py`)

**`Pix2CodeDataset(torch.utils.data.Dataset)`**:
- Scans a directory for `.gui` + `.png` file pairs
- Builds vocabulary from all DSL tokens plus special tokens (`<START>`, `<END>`, `<PAD>`)
- Images: resize to 256x256, normalize with ImageNet mean/std via `torchvision.transforms`
- Token sequences: encode to integer IDs, pad to `CONTEXT_LENGTH`
- Returns `(image, input_ids, target_ids)` where target is input shifted by 1

**`get_dataloaders(data_dir, platform, batch_size)`**:
- Splits data into train/val/test (ratio controlled by config)
- Returns three `DataLoader` instances with appropriate shuffling

### 2. Model (`model.py`)

**`ImageEncoder(nn.Module)`**:
- `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)` with final avgpool + fc removed
- Outputs 8x8x2048 spatial feature map
- `nn.Linear(2048, d_model)` projection to get 64 image tokens of dimension `d_model=256`
- Frozen (no gradient computation)

**`Pix2CodeModel(nn.Module)`**:
- Combines `ImageEncoder` + `nn.TransformerDecoder`
- Token embedding: `nn.Embedding(vocab_size, d_model)`
- Sinusoidal positional encoding (fixed, not learned)
- `nn.TransformerDecoderLayer(d_model=256, nhead=8, dim_feedforward=512)` x 3 layers
- Output projection: `nn.Linear(d_model, vocab_size)`
- `forward(images, token_ids)`: returns logits `(batch, seq_len, vocab_size)`
- `generate(image, max_len, start_token_id)`: autoregressive greedy decoding

### 3. Training (`train.py`)

- Loss: `nn.CrossEntropyLoss(ignore_index=PAD_ID)`
- Optimizer: `torch.optim.Adam(lr=1e-4)`
- Scheduler: `CosineAnnealingLR`
- Teacher forcing during training
- Validation loss computed after each epoch
- Best model saved as `checkpoint.pt`
- CLI: `python train.py --platform web --data_dir ../datasets --epochs 10 --batch_size 64`

### 4. Inference (`generate.py`)

- Loads trained checkpoint
- Greedy decoding: start with `<START>`, predict next token iteratively until `<END>` or max length
- Optional beam search (width=3)
- Outputs `.gui` file compatible with existing compilers
- CLI: `python generate.py --image screenshot.png --checkpoint checkpoint.pt --platform web`

### 5. Config (`Config.py`)

Updated hyperparameters:
- `CONTEXT_LENGTH = 48`
- `IMAGE_SIZE = 256`
- `BATCH_SIZE = 64`
- `EPOCHS = 10`
- `D_MODEL = 256`
- `N_HEADS = 8`
- `N_LAYERS = 3`
- `D_FF = 512`
- `LEARNING_RATE = 1e-4`

### 6. Utils (`model/classes/Utils.py`)

Rewritten to use PyTorch/torchvision:
- `get_image_transform()`: returns `torchvision.transforms.Compose` with resize, normalize
- `get_preprocessed_img()`: uses PIL + transforms instead of cv2
- `show()`: uses matplotlib instead of cv2 window
- Remove `sparsify()` (PyTorch's CrossEntropyLoss takes class indices directly)

### 7. Dependencies (`requirements.txt`)

```
torch>=2.0
torchvision>=0.15
numpy>=1.24
opencv-python>=4.8
Pillow>=10.0
matplotlib>=3.7
```

TensorFlow, Keras, and h5py are removed entirely.

## What Does Not Change

- The entire `compiler/` directory is untouched (no ML framework dependency)
- The DSL format and `.gui` file format remain the same
- The compiler DSL mapping JSON files are unchanged
- The `compiler/classes/` modules (Node, Compiler, Utils) are unchanged

## Verification

1. `python -m py_compile` on all new/modified `.py` files
2. Instantiate `Pix2CodeDataset` and verify it loads/pads correctly (mock data or real dataset)
3. Instantiate `Pix2CodeModel`, run a forward pass with random tensors, verify output shape
4. Run `train.py` for 1 epoch on a small dataset subset to verify training loop works end-to-end
5. Run `generate.py` with a checkpoint to verify it produces valid `.gui` output
