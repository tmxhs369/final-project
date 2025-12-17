# train.py

import os
import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import CFG
from utils import (
    set_seed, ensure_dir, load_json
)


class SeqDataset(Dataset):
    def __init__(self, sequences: List[List[int]], seq_len: int):
        self.sequences = sequences
        self.seq_len = seq_len

        # Flatten all sequences into one long stream
        self.stream = []
        for s in sequences:
            self.stream.extend(s)
        self.stream = torch.tensor(self.stream, dtype=torch.long)

    def __len__(self):
        # number of chunks
        return max(1, (len(self.stream) - 1) // self.seq_len)

    def __getitem__(self, idx):
        i = idx * self.seq_len
        x = self.stream[i:i+self.seq_len]
        y = self.stream[i+1:i+self.seq_len+1]
        return x, y


class LSTMLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.drop(out)
        logits = self.fc(out)
        return logits, hidden


def split_train_val(sequences: List[List[int]], val_ratio: float) -> Tuple[List[List[int]], List[List[int]]]:
    n = len(sequences)
    n_val = max(1, int(n * val_ratio))
    train = sequences[:-n_val]
    val = sequences[-n_val:]
    if len(train) == 0:
        train = val
    return train, val


def run_epoch(model, loader, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0

    crit = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)
        # logits: (B, T, V), y: (B, T)
        loss = crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20, avg_loss))
    return avg_loss, ppl


def main():
    set_seed(CFG.seed)
    ensure_dir(CFG.checkpoints_dir)

    # Load processed
    vocab_path = os.path.join(CFG.processed_dir, "vocab.json")
    seq_path = os.path.join(CFG.processed_dir, "sequences.json")
    if not os.path.exists(vocab_path) or not os.path.exists(seq_path):
        print("[ERROR] Processed files not found. Run preprocess.py first.")
        return

    vocab_obj = load_json(vocab_path)
    vocab = vocab_obj["vocab"]
    vocab_size = len(vocab)

    sequences = load_json(seq_path)["sequences"]
    train_seq, val_seq = split_train_val(sequences, CFG.val_ratio)

    train_ds = SeqDataset(train_seq, CFG.seq_len)
    val_ds = SeqDataset(val_seq, CFG.seq_len)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, drop_last=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")
    print(f"[INFO] vocab_size = {vocab_size}, train_chunks={len(train_ds)}, val_chunks={len(val_ds)}")

    model = LSTMLM(vocab_size, CFG.emb_dim, CFG.hidden_dim, CFG.num_layers, CFG.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)

    best_val = float("inf")
    best_path = os.path.join(CFG.checkpoints_dir, "best.pt")
    last_path = os.path.join(CFG.checkpoints_dir, "last.pt")

    for epoch in range(1, CFG.num_epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, optimizer, device, train=True)
        va_loss, va_ppl = run_epoch(model, val_loader, optimizer, device, train=False)

        print(f"[Epoch {epoch:02d}] train loss={tr_loss:.4f} ppl={tr_ppl:.2f} | val loss={va_loss:.4f} ppl={va_ppl:.2f}")

        # Save last
        torch.save({"model": model.state_dict(), "vocab_size": vocab_size}, last_path)

        # Save best
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "vocab_size": vocab_size}, best_path)
            print(f"  -> saved best checkpoint: {best_path}")

    print("[INFO] Training finished.")
    print(f"  best val loss = {best_val:.4f}")
    print(f"  checkpoints in {CFG.checkpoints_dir}")


if __name__ == "__main__":
    main()
