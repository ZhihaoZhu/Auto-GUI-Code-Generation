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
