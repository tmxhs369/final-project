# final-project
MIDI 기반 음악 생성 모델 (LSTM)
---

###  **목표**
MIDI 파일 데이터셋을 기반으로 새로운 음악을 생성하는 딥러닝 모델을 설계하고 구현한다.

이를 위해 시퀀스 데이터 처리에 적합한 LSTM(Long Short-Term Memory) 모델을 활용하여,
이전 음표 시퀀스를 바탕으로 다음 음표를 예측하고 이를 반복적으로 생성함으로써 새로운 음악을 만들어내는 모델을 구현한다.

---

###  **접근 방식**

1. 데이터 수집
 - 특정 장르(예: 피아노, 재즈, 게임 BGM 등)의 MIDI 파일 데이터셋을 수집한다.
 - MIDI 파일은 음높이(pitch), 시작 시점, 종료 시점 등의 정보를 포함한다.
2. 데이터 전처리
 - pretty_midi 라이브러리를 활용하여 MIDI 파일을 파싱한다.
 - 음악을 다음과 같은 이벤트 기반 토큰 시퀀스로 변환한다.
  - NOTE_ON_{pitch}: 특정 음표의 시작
  - NOTE_OFF_{pitch}: 특정 음표의 종료
  - TIME_SHIFT_{n}: 일정 시간 간격 이동
 - 이를 통해 음악을 자연어 처리 문제와 유사한 시퀀스 예측 문제로 변환한다.
3. 모델 설계
 - Embedding Layer + LSTM + Fully Connected Layer 구조의 Sequence Language Model을 구성한다.
 - 모델은 이전 토큰들을 입력으로 받아 다음 토큰의 확률 분포를 예측한다.
4. 모델 학습
 - Cross-Entropy Loss를 사용하여 다음 토큰 예측 성능을 최적화한다.
 - AdamW 옵티마이저를 사용하여 학습을 수행한다.
 - Validation Loss 기준으로 최적의 모델을 저장한다.
5. 음악 생성
 - 학습된 모델을 사용하여 <BOS> 토큰부터 시작해 순차적으로 토큰을 샘플링한다.
 - Temperature 및 Top-k Sampling을 적용하여 생성 다양성을 조절한다.
 - 생성된 토큰 시퀀스를 다시 MIDI 파일로 변환한다.
6. 평가 방법
 - 정량적 평가
  - Pitch Class Entropy (음높이 다양성)
  - Note Density (초당 음표 수)
  - N-gram 반복률 (패턴 반복 정도)
 - 정성적 평가
  - 사람이 직접 청취하여 음악적 자연스러움, 일관성, 장르 적합성 평가

---

##  1. `config.py`

```python
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

```

###  설명

1. 역할
- 프로젝트 전반에서 사용되는 하이퍼파라미터 및 경로 설정 파일
2. 주요 내용
- 데이터 경로 
- 토큰화 관련 설정 
- LSTM 모델 구조 파라미터 
- 학습 관련 설정 
- 음악 생성 시 사용되는 sampling 파라미터
3. 목적
- 모든 설정 값을 한 곳에서 관리하여 실험 재현성 및 코드 가독성 향상

---

##  2. `utils.py`

```python
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

```

###  설명

1. 역할
- 프로젝트 전반에서 공통적으로 사용되는 유틸리티 함수 모음
2. 주요 기능
- MIDI ↔ 토큰 변환
- Sampling 함수
- 평가 지표 계산
3. 목적
- 음악 표현, 샘플링, 평가 로직을 분리하여 핵심 모델 코드 단순화

---

##  3. `preprocess.py`

```python
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
- 음악 데이터를 자연어 처리 문제와 유사한 시퀀스 예측 문제로 변환
---

##  4. `train.py`

```python
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

```
---

##  5. `generate.py`


```python
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
```
---

##  5. `generate.py`




###  설명

1. **데이터 로드**: Hugging Face `datasets` 라이브러리를 사용해 CNN/DailyMail 데이터셋을 불러온다.
2. **전처리**: 기사와 요약을 BART 모델 입력 형식에 맞게 토크나이징
3. **모델 로드**: BART 모델을 로드하여 fine-tuning을 준비한다.
4. **학습 인자 설정**: 학습에 필요한 설정(배치 사이즈, 학습률, 평가 전략 등)을 정의
5. **평가지표 설정**: ROUGE 점수 계산을 위한 함수를 정의
6. **Trainer 구성**: Hugging Face의 `Trainer` 클래스를 활용해 학습을 구성
7. **모델 학습**: fine-tuning 시작.
8. **테스트셋 평가**: 학습이 끝난 모델을 테스트셋에 적용해 성능을 평가한다.

---

##  4. 요약 결과 확인

```python
article = dataset["test"][0]["article"]
ref = dataset["test"][0]["highlights"]

input_ids = tokenizer(article, return_tensors="pt", truncation=True).input_ids
summary_ids = model.generate(input_ids, max_length=MAX_TARGET_LENGTH)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(" 원문:\n", article[:500])
print("\n 모델 요약:\n", summary)
print("\n 참조 요약:\n", ref)
```

###  설명

* 학습된 모델을 사용해 하나의 기사에 대해 요약을 생성하고, 사람이 직접 비교할 수 있도록 참조 요약과 함께 출력

---

##  5. 추출적 요약

```python
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

def extractive_summary(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)
```

###  설명

* Sumy의 LSA 방식으로 간단한 **추출적 요약**을 구현한다.
* 이는 문장에서 중요한 문장을 선택해 요약을 만드는 방식입니다. 별도의 학습 없이 사용 가능하다.

---

##  전체 요약

| 단계           | 설명                                  |
| ------------ | ----------------------------------- |
| 1. 설정 파일 작성  | 모델, 데이터셋, 하이퍼파라미터 설정                |
| 2. 전처리 함수 구현 | 기사와 요약을 모델 입력에 맞게 변환                |
| 3. 메인 코드 실행  | 데이터 로드 → 모델 준비 → 학습 → 평가까지 전체 파이프라인 |
| 4. 결과 확인     | 실제 기사에 대해 생성된 요약을 수동으로 비교           |
| 5. 추가 기법     | Sumy 등으로 추출적 요약도 테스트 가능             |

---
