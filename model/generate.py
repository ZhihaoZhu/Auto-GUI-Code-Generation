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
