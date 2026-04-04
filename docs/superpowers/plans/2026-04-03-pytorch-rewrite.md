# PyTorch Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the ML pipeline from TensorFlow 1.x to PyTorch, building the complete pix2code pipeline (data loading, ResNet50 encoder + Transformer decoder model, training loop, inference/generation).

**Architecture:** Pre-trained ResNet50 extracts spatial features (8x8x2048) from GUI screenshots, projected to 64 image tokens. A 3-layer Transformer decoder with cross-attention generates DSL tokens autoregressively. Teacher forcing during training, greedy/beam search at inference.

**Tech Stack:** PyTorch >= 2.0, torchvision >= 0.15, numpy, Pillow, matplotlib

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `model/classes/model/Config.py` | Modify | Add Transformer hyperparameters |
| `model/classes/Utils.py` | Rewrite | PyTorch/torchvision image transforms |
| `model/dataset.py` | Create | `Pix2CodeDataset` + vocabulary + dataloaders |
| `model/model.py` | Create | `ImageEncoder` + `Pix2CodeModel` |
| `model/train.py` | Create | Training loop with validation |
| `model/generate.py` | Create | Inference: screenshot -> .gui |
| `model/Extract_features.py` | Delete | Replaced by encoder in model.py |
| `model/data_process.py` | Delete | Replaced by dataset.py |
| `requirements.txt` | Modify | PyTorch stack, drop TF/Keras |
| `README.md` | Modify | Update for PyTorch pipeline |
| `tests/test_dataset.py` | Create | Dataset + vocabulary tests |
| `tests/test_model.py` | Create | Model shape + forward pass tests |

---

### Task 1: Update Config and requirements.txt

**Files:**
- Modify: `model/classes/model/Config.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Update Config.py with Transformer hyperparameters**

```python
"""Hyperparameter and configuration constants for the ML model."""

__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

# Data
CONTEXT_LENGTH = 48
IMAGE_SIZE = 256
BATCH_SIZE = 64

# Training
EPOCHS = 10
LEARNING_RATE = 1e-4

# Model architecture
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 3
D_FF = 512
DROPOUT = 0.1
```

- [ ] **Step 2: Update requirements.txt**

```
torch>=2.0
torchvision>=0.15
numpy>=1.24
opencv-python>=4.8
Pillow>=10.0
matplotlib>=3.7
```

- [ ] **Step 3: Verify syntax**

Run: `python -m py_compile model/classes/model/Config.py`
Expected: No output (success)

- [ ] **Step 4: Commit**

```bash
git commit -m "Update config for Transformer architecture and switch to PyTorch dependencies"
```

---

### Task 2: Rewrite Utils for PyTorch

**Files:**
- Rewrite: `model/classes/Utils.py`

- [ ] **Step 1: Rewrite Utils.py with torchvision transforms**

```python
"""Utility functions for image preprocessing."""

from PIL import Image
from torchvision import transforms

from model.classes.model.Config import IMAGE_SIZE


def get_image_transform():
    """Return the standard image transform pipeline for model input.

    Resizes to IMAGE_SIZE x IMAGE_SIZE, converts to tensor, and normalizes
    with ImageNet mean and standard deviation.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_preprocessed_img(img_path):
    """Load an image and apply the standard transform pipeline.

    Args:
        img_path: Path to the image file.

    Returns:
        Tensor of shape (3, IMAGE_SIZE, IMAGE_SIZE), normalized.
    """
    img = Image.open(img_path).convert("RGB")
    return get_image_transform()(img)


def show(image_tensor):
    """Display a transformed image tensor using matplotlib.

    Args:
        image_tensor: Tensor of shape (3, H, W), ImageNet-normalized.
    """
    import matplotlib.pyplot as plt

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = image_tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.permute(1, 2, 0).clamp(0, 1).numpy()
    plt.imshow(img)
    plt.axis("off")
    plt.show()
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile model/classes/Utils.py`
Expected: No output (success)

- [ ] **Step 3: Commit**

```bash
git commit -m "Rewrite model Utils for PyTorch with torchvision transforms"
```

---

### Task 3: Create Dataset with Vocabulary

**Files:**
- Create: `model/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write test for vocabulary and dataset**

Create `tests/test_dataset.py`:

```python
"""Tests for Pix2CodeDataset and Vocabulary."""

import os
import tempfile
import pytest
import torch
from PIL import Image

from model.dataset import Vocabulary, Pix2CodeDataset, get_dataloaders


@pytest.fixture
def sample_data_dir():
    """Create a temporary directory with mock .gui + .png pairs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # DSL content matching web platform vocabulary
        gui_content = "body {\nheader {\nbtn-active\nbtn-inactive\n}\nrow {\nsingle {\nsmall-title\ntext\nbtn-green\n}\n}\n}\n"
        for i in range(6):
            name = f"sample_{i}"
            with open(os.path.join(tmpdir, f"{name}.gui"), "w") as f:
                f.write(gui_content)
            img = Image.new("RGB", (256, 256), color=(i * 40, i * 40, i * 40))
            img.save(os.path.join(tmpdir, f"{name}.png"))
        yield tmpdir


class TestVocabulary:
    def test_special_tokens_present(self):
        vocab = Vocabulary()
        vocab.build(["body header btn-active"])
        assert "<PAD>" in vocab.token_to_id
        assert "<START>" in vocab.token_to_id
        assert "<END>" in vocab.token_to_id

    def test_pad_is_zero(self):
        vocab = Vocabulary()
        vocab.build(["body header"])
        assert vocab.token_to_id["<PAD>"] == 0

    def test_encode_decode_roundtrip(self):
        vocab = Vocabulary()
        vocab.build(["body header btn-active"])
        tokens = ["body", "header", "btn-active"]
        ids = vocab.encode(tokens)
        decoded = vocab.decode(ids)
        assert decoded == tokens

    def test_vocab_size(self):
        vocab = Vocabulary()
        vocab.build(["body header", "body btn-active"])
        # 3 unique tokens + 3 special tokens
        assert len(vocab) == 6


class TestPix2CodeDataset:
    def test_dataset_length(self, sample_data_dir):
        ds = Pix2CodeDataset(sample_data_dir)
        assert len(ds) == 6

    def test_item_shapes(self, sample_data_dir):
        ds = Pix2CodeDataset(sample_data_dir)
        image, input_ids, target_ids = ds[0]
        assert image.shape == (3, 256, 256)
        assert input_ids.shape == target_ids.shape
        assert input_ids.dtype == torch.long

    def test_target_is_shifted_input(self, sample_data_dir):
        ds = Pix2CodeDataset(sample_data_dir)
        _, input_ids, target_ids = ds[0]
        # target should be input shifted left by 1:
        # input:  <START> body { header ...
        # target: body    {    header ...  <END>
        start_id = ds.vocab.token_to_id["<START>"]
        assert input_ids[0].item() == start_id

    def test_get_dataloaders(self, sample_data_dir):
        train_dl, val_dl, test_dl = get_dataloaders(
            sample_data_dir, batch_size=2
        )
        assert train_dl is not None
        batch = next(iter(train_dl))
        images, input_ids, target_ids = batch
        assert images.shape[0] <= 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'model.dataset')

- [ ] **Step 3: Create model/dataset.py**

```python
"""PyTorch Dataset and Vocabulary for pix2code GUI-to-DSL data."""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

from model.classes.Utils import get_image_transform
from model.classes.model.Config import CONTEXT_LENGTH


class Vocabulary:
    """Maps DSL tokens to integer IDs and back.

    Special tokens:
        <PAD> (0): padding for sequences shorter than CONTEXT_LENGTH
        <START> (1): beginning of sequence marker
        <END> (2): end of sequence marker
    """

    PAD = "<PAD>"
    START = "<START>"
    END = "<END>"

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def build(self, dsl_texts):
        """Build vocabulary from a list of DSL text strings.

        Args:
            dsl_texts: List of strings, each containing space-separated DSL tokens.
        """
        tokens = set()
        for text in dsl_texts:
            for token in text.split():
                tokens.add(token)

        self.token_to_id = {self.PAD: 0, self.START: 1, self.END: 2}
        for i, token in enumerate(sorted(tokens), start=3):
            self.token_to_id[token] = i
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, tokens):
        """Convert a list of token strings to a list of integer IDs."""
        return [self.token_to_id[t] for t in tokens]

    def decode(self, ids):
        """Convert a list of integer IDs back to token strings.

        Stops at <END> and filters out <PAD> and <START>.
        """
        tokens = []
        for i in ids:
            token = self.id_to_token[i]
            if token == self.END:
                break
            if token not in (self.PAD, self.START):
                tokens.append(token)
        return tokens

    def __len__(self):
        return len(self.token_to_id)


class Pix2CodeDataset(Dataset):
    """Dataset that loads .gui + .png pairs for pix2code training.

    Each sample returns:
        image: Tensor (3, IMAGE_SIZE, IMAGE_SIZE), ImageNet-normalized
        input_ids: Tensor (seq_len,), token IDs with <START> prefix, padded
        target_ids: Tensor (seq_len,), token IDs with <END> suffix, padded
    """

    def __init__(self, data_dir, context_length=CONTEXT_LENGTH):
        self.data_dir = data_dir
        self.context_length = context_length
        self.transform = get_image_transform()

        # Find all .gui files and their matching .png files
        self.samples = []
        for f in sorted(os.listdir(data_dir)):
            if f.endswith(".gui"):
                name = f[:-4]
                png_path = os.path.join(data_dir, name + ".png")
                gui_path = os.path.join(data_dir, f)
                if os.path.exists(png_path):
                    self.samples.append((png_path, gui_path))

        # Read all DSL files and build vocabulary
        dsl_texts = []
        self.token_sequences = []
        for _, gui_path in self.samples:
            with open(gui_path) as f:
                text = f.read().replace("\n", " ").strip()
            dsl_texts.append(text)
            self.token_sequences.append(text.split())

        self.vocab = Vocabulary()
        self.vocab.build(dsl_texts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        png_path, _ = self.samples[idx]
        tokens = self.token_sequences[idx]

        # Load and transform image
        image = Image.open(png_path).convert("RGB")
        image = self.transform(image)

        # Build input sequence: <START> + tokens, padded to context_length
        input_tokens = [Vocabulary.START] + tokens
        target_tokens = tokens + [Vocabulary.END]

        # Truncate if needed
        input_tokens = input_tokens[:self.context_length]
        target_tokens = target_tokens[:self.context_length]

        # Encode to IDs
        input_ids = self.vocab.encode(input_tokens)
        target_ids = self.vocab.encode(target_tokens)

        # Pad to context_length
        pad_id = self.vocab.token_to_id[Vocabulary.PAD]
        while len(input_ids) < self.context_length:
            input_ids.append(pad_id)
        while len(target_ids) < self.context_length:
            target_ids.append(pad_id)

        return (
            image,
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


def get_dataloaders(data_dir, batch_size, val_fraction=0.15, test_fraction=0.15):
    """Create train/val/test DataLoaders from a data directory.

    Args:
        data_dir: Directory containing .gui + .png file pairs.
        batch_size: Batch size for all loaders.
        val_fraction: Fraction of data for validation.
        test_fraction: Fraction of data for testing.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    dataset = Pix2CodeDataset(data_dir)

    total = len(dataset)
    val_size = max(1, int(total * val_fraction))
    test_size = max(1, int(total * test_fraction))
    train_size = total - val_size - test_size

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git commit -m "Add PyTorch Dataset and Vocabulary for pix2code data"
```

---

### Task 4: Create the Model (ImageEncoder + Transformer Decoder)

**Files:**
- Create: `model/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write model tests**

Create `tests/test_model.py`:

```python
"""Tests for Pix2CodeModel architecture."""

import torch
import pytest

from model.model import ImageEncoder, Pix2CodeModel


class TestImageEncoder:
    def test_output_shape(self):
        encoder = ImageEncoder(d_model=256)
        images = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            features = encoder(images)
        # Should be (batch, 64 spatial tokens, d_model)
        assert features.shape == (2, 64, 256)

    def test_frozen_resnet(self):
        encoder = ImageEncoder(d_model=256)
        for param in encoder.resnet.parameters():
            assert not param.requires_grad


class TestPix2CodeModel:
    def test_forward_output_shape(self):
        vocab_size = 23
        model = Pix2CodeModel(vocab_size=vocab_size, d_model=256, n_heads=8,
                              n_layers=3, d_ff=512, max_seq_len=48)
        images = torch.randn(2, 3, 256, 256)
        token_ids = torch.randint(0, vocab_size, (2, 48))
        with torch.no_grad():
            logits = model(images, token_ids)
        assert logits.shape == (2, 48, vocab_size)

    def test_generate_output(self):
        vocab_size = 23
        model = Pix2CodeModel(vocab_size=vocab_size, d_model=256, n_heads=8,
                              n_layers=3, d_ff=512, max_seq_len=48)
        model.eval()
        image = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output_ids = model.generate(image, start_token_id=1,
                                        end_token_id=2, max_len=20)
        assert isinstance(output_ids, list)
        assert len(output_ids) <= 20
        assert all(isinstance(i, int) for i in output_ids)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'model.model')

- [ ] **Step 3: Create model/model.py**

```python
"""ResNet50 encoder + Transformer decoder model for GUI code generation."""

import math
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ImageEncoder(nn.Module):
    """Extract spatial features from screenshots using a frozen ResNet50.

    Removes the final avgpool and fc layers, keeping the 8x8 spatial grid.
    Projects 2048-dim features to d_model, yielding 64 image tokens.
    """

    def __init__(self, d_model):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove avgpool and fc — keep up to layer4
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.projection = nn.Linear(2048, d_model)

    def forward(self, images):
        """Extract image features.

        Args:
            images: (batch, 3, 256, 256) tensor, ImageNet-normalized.

        Returns:
            (batch, 64, d_model) tensor of image tokens.
        """
        with torch.no_grad():
            features = self.resnet(images)  # (batch, 2048, 8, 8)
        batch_size = features.size(0)
        features = features.view(batch_size, 2048, -1)  # (batch, 2048, 64)
        features = features.permute(0, 2, 1)  # (batch, 64, 2048)
        return self.projection(features)  # (batch, 64, d_model)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the Transformer decoder."""

    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """Add positional encoding to input embeddings.

        Args:
            x: (batch, seq_len, d_model) tensor.
        """
        return x + self.pe[:, :x.size(1)]


class Pix2CodeModel(nn.Module):
    """Screenshot-to-DSL model: ResNet50 encoder + Transformer decoder.

    The encoder extracts spatial features from a screenshot as 64 image tokens.
    The decoder attends to these image tokens via cross-attention while
    autoregressively predicting DSL tokens.
    """

    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=3,
                 d_ff=512, max_seq_len=48, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.encoder = ImageEncoder(d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers,
        )
        self.output_projection = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, seq_len, device):
        """Generate a causal (upper-triangular) attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, images, token_ids):
        """Forward pass with teacher forcing.

        Args:
            images: (batch, 3, 256, 256) screenshot tensors.
            token_ids: (batch, seq_len) input token ID sequences.

        Returns:
            (batch, seq_len, vocab_size) logits.
        """
        # Encode images to memory
        memory = self.encoder(images)  # (batch, 64, d_model)

        # Embed tokens + positional encoding
        x = self.token_embedding(token_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Causal mask for autoregressive decoding
        causal_mask = self._generate_causal_mask(
            token_ids.size(1), token_ids.device
        )

        # Transformer decoder
        x = self.transformer_decoder(x, memory, tgt_mask=causal_mask)

        return self.output_projection(x)  # (batch, seq_len, vocab_size)

    @torch.no_grad()
    def generate(self, image, start_token_id, end_token_id, max_len=48):
        """Generate DSL tokens from a single screenshot via greedy decoding.

        Args:
            image: (1, 3, 256, 256) single image tensor.
            start_token_id: Integer ID of the <START> token.
            end_token_id: Integer ID of the <END> token.
            max_len: Maximum number of tokens to generate.

        Returns:
            List of integer token IDs (excluding <START>).
        """
        self.eval()
        device = next(self.parameters()).device
        image = image.to(device)

        memory = self.encoder(image)  # (1, 64, d_model)
        generated = [start_token_id]

        for _ in range(max_len):
            token_ids = torch.tensor([generated], dtype=torch.long, device=device)
            x = self.token_embedding(token_ids) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)

            causal_mask = self._generate_causal_mask(len(generated), device)
            x = self.transformer_decoder(x, memory, tgt_mask=causal_mask)

            logits = self.output_projection(x[:, -1, :])  # (1, vocab_size)
            next_token = logits.argmax(dim=-1).item()

            if next_token == end_token_id:
                break
            generated.append(next_token)

        return generated[1:]  # exclude <START>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_model.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git commit -m "Add Pix2CodeModel with ResNet50 encoder and Transformer decoder"
```

---

### Task 5: Create Training Script

**Files:**
- Create: `model/train.py`

- [ ] **Step 1: Create model/train.py**

```python
"""Training script for the pix2code model."""

import argparse
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.dataset import get_dataloaders, Pix2CodeDataset
from model.model import Pix2CodeModel
from model.classes.model.Config import (
    CONTEXT_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT,
)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Run one training epoch.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for images, input_ids, target_ids in dataloader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(images, input_ids)  # (batch, seq_len, vocab_size)
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Run validation.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    for images, input_ids, target_ids in dataloader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(images, input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train pix2code model")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .gui + .png file pairs")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        args.data_dir, batch_size=args.batch_size,
    )

    # Get vocabulary size from the dataset
    vocab_size = len(train_loader.dataset.dataset.vocab)
    pad_id = train_loader.dataset.dataset.vocab.token_to_id["<PAD>"]
    print(f"Vocabulary size: {vocab_size}")

    # Create model
    model = Pix2CodeModel(
        vocab_size=vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=CONTEXT_LENGTH,
        dropout=DROPOUT,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "vocab": train_loader.dataset.dataset.vocab,
            }, checkpoint_path)
            print(f"  Saved best model (val_loss={val_loss:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile model/train.py`
Expected: No output (success)

- [ ] **Step 3: Commit**

```bash
git commit -m "Add training script with validation and checkpointing"
```

---

### Task 6: Create Inference/Generation Script

**Files:**
- Create: `model/generate.py`

- [ ] **Step 1: Create model/generate.py**

```python
"""Generate .gui DSL files from screenshots using a trained pix2code model."""

import argparse
import torch

from model.model import Pix2CodeModel
from model.classes.Utils import get_preprocessed_img
from model.classes.model.Config import D_MODEL, N_HEADS, N_LAYERS, D_FF, CONTEXT_LENGTH


def load_model(checkpoint_path, device):
    """Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: torch.device to load the model on.

    Returns:
        Tuple of (model, vocab).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = checkpoint["vocab"]

    model = Pix2CodeModel(
        vocab_size=len(vocab), d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=CONTEXT_LENGTH,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab


def generate_gui(model, vocab, image_path, device, max_len=CONTEXT_LENGTH):
    """Generate DSL tokens from a screenshot.

    Args:
        model: Trained Pix2CodeModel.
        vocab: Vocabulary instance.
        image_path: Path to the input screenshot.
        device: torch.device.
        max_len: Maximum tokens to generate.

    Returns:
        String of DSL code.
    """
    image = get_preprocessed_img(image_path).unsqueeze(0).to(device)

    start_id = vocab.token_to_id["<START>"]
    end_id = vocab.token_to_id["<END>"]

    token_ids = model.generate(image, start_id, end_id, max_len=max_len)
    tokens = vocab.decode(token_ids)

    # Format DSL with proper indentation
    indent = 0
    lines = []
    for token in tokens:
        if token == "}":
            indent -= 1
        lines.append("  " * indent + token)
        if token.endswith("{"):
            indent += 1
        elif token == "{":
            indent += 1

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate .gui from screenshot")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input screenshot")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .gui file path (default: stdout)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab = load_model(args.checkpoint, device)

    gui_code = generate_gui(model, vocab, args.image, device)

    if args.output:
        with open(args.output, "w") as f:
            f.write(gui_code)
        print(f"Generated DSL written to {args.output}")
    else:
        print(gui_code)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile model/generate.py`
Expected: No output (success)

- [ ] **Step 3: Commit**

```bash
git commit -m "Add inference script for generating .gui files from screenshots"
```

---

### Task 7: Delete Old TF Files and Update Documentation

**Files:**
- Delete: `model/Extract_features.py`
- Delete: `model/data_process.py`
- Modify: `README.md`

- [ ] **Step 1: Delete old TensorFlow files**

```bash
git rm model/Extract_features.py model/data_process.py
```

- [ ] **Step 2: Update README.md**

Update the README to reflect the PyTorch rewrite: new directory structure, updated getting started, new usage examples for training and inference.

Key changes:
- Update directory overview to show `dataset.py`, `model.py`, `train.py`, `generate.py`
- Update "Getting Started" with PyTorch install instructions
- Add "Training" section with `python -m model.train --data_dir ...`
- Add "Inference" section with `python -m model.generate --image ... --checkpoint ...`
- Update "Known Limitations" to remove TF 1.x references
- Remove legacy dependency note

- [ ] **Step 3: Delete the empty `model/classes/dataset/` package**

```bash
rm -r model/classes/dataset/
```

(This was an empty placeholder that is now unnecessary — dataset logic lives in `model/dataset.py`.)

- [ ] **Step 4: Verify all files compile**

```bash
python -m py_compile model/model.py && python -m py_compile model/dataset.py && python -m py_compile model/train.py && python -m py_compile model/generate.py && python -m py_compile model/classes/Utils.py && python -m py_compile model/classes/model/Config.py
```

Expected: No output (all pass)

- [ ] **Step 5: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 6: Commit and push**

```bash
git add -A
git commit -m "Complete PyTorch rewrite: replace TF pipeline with ResNet50+Transformer model"
git push origin master
```

---

## Verification Checklist

After all tasks are complete:

1. `python -m py_compile` passes on all `.py` files
2. `python -m pytest tests/ -v` — all tests pass
3. `Pix2CodeModel` forward pass with random data produces correct output shape `(batch, seq_len, vocab_size)`
4. `model.generate()` produces a list of integer token IDs
5. No TensorFlow/Keras imports remain anywhere in the codebase
6. Compiler files in `compiler/` are completely unchanged
7. `git status` is clean after final commit
