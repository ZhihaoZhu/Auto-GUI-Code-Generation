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
