"""Training script for the pix2code model."""

import argparse
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.dataset import get_dataloaders
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
    vocab = train_loader.dataset.dataset.vocab
    vocab_size = len(vocab)
    pad_id = vocab.token_to_id["<PAD>"]
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
                "vocab": vocab,
            }, checkpoint_path)
            print(f"  Saved best model (val_loss={val_loss:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    main()
