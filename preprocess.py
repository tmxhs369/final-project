# preprocess.py

import os
from glob import glob
from typing import List

import pretty_midi
from tqdm import tqdm

from config import CFG
from utils import ensure_dir, midi_to_tokens, build_vocab, encode_tokens, save_json


def list_midi_files(midi_dir: str) -> List[str]:
    exts = ["*.mid", "*.midi"]
    files = []
    for e in exts:
        files.extend(glob(os.path.join(midi_dir, e)))
    return sorted(files)


def main():
    ensure_dir(CFG.processed_dir)

    midi_files = list_midi_files(CFG.midi_dir)
    if len(midi_files) == 0:
        print(f"[ERROR] No MIDI files found in: {CFG.midi_dir}")
        print("Put some .mid/.midi files i적
