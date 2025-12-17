# generate.py

import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn

from config import CFG
from utils import (
    ensure_dir, load_json, decode_ids, tokens_to_midi,
    sample_next_token, pitch_class_entropy, note_density, ngram_repetition_rate
)


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


@torch.no_grad()
def generate_sequence(model, start_id: int, steps: int, temperature: float, top_k: int, device: str):
    model.eval()
    ids = [start_id]
    x = torch.tensor([[start_id]], dtype=torch.long, device=device)
    hidden = None

    for _ in range(steps):
        logits, hidden = model(x, hidden)
        next_id = sample_next_token(logits[0, -1], temperature=temperature, top_k=top_k)
        ids.append(next_id)
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="path to .pt checkpoint")
    ap.add_argument("--num_files", type=int, default=3)
    ap.add_argument("--steps", type=int, default=CFG.steps)
    ap.add_argument("--temperature", type=float, default=CFG.temperature)
    ap.add_argument("--top_k", type=int, default=CFG.top_k)
    args = ap.parse_args()

    vocab_path = os.path.join(CFG.processed_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        print("[ERROR] vocab not found. Run preprocess.py first.")
        return

    vocab_obj = load_json(vocab_path)
    vocab = vocab_obj["vocab"]
    id2tok = {int(k): v for k, v in vocab_obj["id2tok"].items()}

    bos_id = vocab["<BOS>"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    vocab_size = ckpt["vocab_size"]

    model = LSTMLM(vocab_size, CFG.emb_dim, CFG.hidden_dim, CFG.num_layers, CFG.dropout).to(device)
    model.load_state_dict(ckpt["model"])

    ensure_dir(CFG.generated_dir)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[INFO] Generating {args.num_files} files...")

    for i in range(args.num_files):
        ids = generate_sequence(
            model=model,
            start_id=bos_id,
            steps=args.steps,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        toks = decode_ids(ids, id2tok)

        # Stop at EOS if appears
        if "<EOS>" in toks:
            toks = toks[:toks.index("<EOS>") + 1]

        pm = tokens_to_midi(toks, CFG.time_step)
        out_path = os.path.join(CFG.generated_dir, f"gen_{stamp}_{i:02d}.mid")
        pm.write(out_path)

        # Metrics
        ent = pitch_class_entropy(pm)
        dens = note_density(pm)
        rep = ngram_repetition_rate(toks, n=4)

        print(f"  saved: {out_path}")
        print(f"    pitch-class entropy: {ent:.3f}")
        print(f"    note density       : {dens:.3f} notes/sec")
        print(f"    4-gram repetition  : {rep:.3f}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
