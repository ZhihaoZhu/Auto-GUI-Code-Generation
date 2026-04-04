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
