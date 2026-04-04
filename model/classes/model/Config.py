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
