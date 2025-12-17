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

* 사용할 사전학습 모델(BART)을 지정한다.
* CNN/Daily Mail 데이터셋 버전 3.0.0을 사용함
* 입력 텍스트와 요약 길이, 학습 관련 하이퍼파라미터(Batch Size, Epoch 수 등)를 설정

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

* 기사 본문(`article`)과 요약문(`highlights`)을 토크나이징
* 입력과 출력 모두 정해진 길이로 자르고 패딩을 적용해 모델이 학습 가능한 형태로 변환

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
        print("Put some .mid/.midi files into data/midi/ and retry.")
        return

    all_tokens = []
    kept_files = []

    print(f"[INFO] Found {len(midi_files)} MIDI files. Tokenizing...")
    for fp in tqdm(midi_files):
        try:
            pm = pretty_midi.PrettyMIDI(fp)
            tokens = midi_to_tokens(pm, CFG.time_step, CFG.max_time_shift_steps)
            if len(tokens) < 10:
                continue
            all_tokens.append(tokens)
            kept_files.append(fp)
        except Exception:
            continue

    if len(all_tokens) == 0:
        print("[ERROR] Tokenization produced no usable sequences.")
        return

    vocab, id2tok = build_vocab(all_tokens)

    encoded = [encode_tokens(toks, vocab) for toks in all_tokens]

    # Save
    save_json({"vocab": vocab, "id2tok": {str(k): v for k, v in id2tok.items()}},
              os.path.join(CFG.processed_dir, "vocab.json"))
    save_json({"files": kept_files}, os.path.join(CFG.processed_dir, "files.json"))
    save_json({"sequences": encoded}, os.path.join(CFG.processed_dir, "sequences.json"))

    print("[INFO] Done.")
    print(f"  usable files : {len(kept_files)}")
    print(f"  vocab size   : {len(vocab)}")
    print(f"  saved to     : {CFG.processed_dir}")


if __name__ == "__main__":
    main()

```

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
