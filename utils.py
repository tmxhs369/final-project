# utils.py

import os
import json
import math
import random
from typing import List, Dict, Tuple

import numpy as np
import pretty_midi

import torch
import torch.nn.functional as F


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Tokenization: MIDI <-> Tokens
# -----------------------------

def build_vocab(tokens_list: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab = {}
    for tokens in tokens_list:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)

    # Add special tokens
    for sp in ["<PAD>", "<BOS>", "<EOS>"]:
        if sp not in vocab:
            vocab[sp] = len(vocab)

    id2tok = {i: t for t, i in vocab.items()}
    return vocab, id2tok


def midi_to_tokens(pm: pretty_midi.PrettyMIDI, time_step: float, max_shift_steps: int) -> List[str]:
    """
    Convert MIDI to a simple event-based token stream:
    - NOTE_ON_{pitch}
    - NOTE_OFF_{pitch}
    - TIME_SHIFT_{k} where k is number of time steps
    We flatten all instruments into a single timeline.
    """
    events = []

    # gather note on/off events
    for inst in pm.instruments:
        for note in inst.notes:
            events.append((note.start, f"NOTE_ON_{note.pitch}"))
            events.append((note.end, f"NOTE_OFF_{note.pitch}"))

    if not events:
        return []

    events.sort(key=lambda x: x[0])

    tokens = ["<BOS>"]
    prev_t = events[0][0]

    for t, ev in events:
        dt = max(0.0, t - prev_t)
        steps = int(round(dt / time_step))

        # emit time shifts in chunks
        while steps > 0:
            k = min(steps, max_shift_steps)
            tokens.append(f"TIME_SHIFT_{k}")
            steps -= k

        tokens.append(ev)
        prev_t = t

    tokens.append("<EOS>")
    return tokens


def tokens_to_midi(tokens: List[str], time_step: float, default_tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
    """
    Convert token stream back to MIDI.
    We use one instrument (Acoustic Grand Piano) by default.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=default_tempo)
    inst = pretty_midi.Instrument(program=0)

    t = 0.0
    active_notes: Dict[int, float] = {}  # pitch -> start time

    for tok in tokens:
        if tok in ("<PAD>", "<BOS>", "<EOS>"):
            continue

        if tok.startswith("TIME_SHIFT_"):
            k = int(tok.split("_")[-1])
            t += k * time_step
        elif tok.startswith("NOTE_ON_"):
            pitch = int(tok.split("_")[-1])
            # if already active, close it first to avoid overlaps
            if pitch in active_notes:
                start = active_notes[pitch]
                end = max(start + time_step, t)
                inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))
            active_notes[pitch] = t
        elif tok.startswith("NOTE_OFF_"):
            pitch = int(tok.split("_")[-1])
            if pitch in active_notes:
                start = active_notes[pitch]
                end = max(start + time_step, t)
                inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))
                del active_notes[pitch]

    # Close remaining active notes
    for pitch, start in active_notes.items():
        end = start + time_step
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))

    pm.instruments.append(inst)
    return pm


def encode_tokens(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    unk_id = vocab.get("<PAD>", 0)
    return [vocab.get(t, unk_id) for t in tokens]


def decode_ids(ids: List[int], id2tok: Dict[int, str]) -> List[str]:
    return [id2tok[i] for i in ids]


# -----------------------------
# Sampling
# -----------------------------

@torch.no_grad()
def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0) -> int:
    """
    logits: (vocab,)
    """
    if temperature <= 0:
        temperature = 1e-6
    logits = logits / temperature

    if top_k and top_k > 0:
        v, ix = torch.topk(logits, k=min(top_k, logits.size(-1)))
        probs = F.softmax(v, dim=-1)
        next_local = torch.multinomial(probs, num_samples=1).item()
        return ix[next_local].item()
    else:
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()


# -----------------------------
# Simple evaluation metrics
# -----------------------------

def pitch_class_entropy(pm: pretty_midi.PrettyMIDI) -> float:
    """
    Entropy of pitch classes (0..11). Higher => more diverse pitch usage.
    """
    counts = np.zeros(12, dtype=np.float64)
    for inst in pm.instruments:
        for note in inst.notes:
            counts[note.pitch % 12] += 1.0
    s = counts.sum()
    if s == 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def note_density(pm: pretty_midi.PrettyMIDI) -> float:
    """
    Notes per second.
    """
    total_notes = sum(len(inst.notes) for inst in pm.instruments)
    end_time = pm.get_end_time()
    if end_time <= 0:
        return 0.0
    return float(total_notes / end_time)


def ngram_repetition_rate(tokens: List[str], n: int = 4) -> float:
    """
    Rough repetition: fraction of repeated n-grams among all n-grams.
    """
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    if total == 0:
        return 0.0
    return float(1.0 - unique / total)


# -----------------------------
# I/O helpers
# -----------------------------

def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
