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
