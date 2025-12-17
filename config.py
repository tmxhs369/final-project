# config.py

from dataclasses import dataclass

@dataclass
class Config:
    # Data
    midi_dir: str = "data/midi"
    processed_dir: str = "outputs/processed"
    checkpoints_dir: str = "outputs/checkpoints"
    generated_dir: str = "outputs/generated"

    # Tokenization / time quantization
    # We'll quantize time shifts in fixed steps (seconds).
    time_step: float = 0.05  # 50ms
    max_time_shift_steps: int = 100  # max TIME_SHIFT_100 => 5s

    # Model
    emb_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.2

    # Training
    seed: int = 42
    batch_size: int = 16
    lr: float = 3e-4
    num_epochs: int = 10
    seq_len: int = 256  # chunk length for training
    grad_clip: float = 1.0
    val_ratio: float = 0.1

    # Generation
    temperature: float = 1.0
    top_k: int = 50
    steps: int = 4000

CFG = Config()
