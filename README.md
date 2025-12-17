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
MODEL_NAME = "facebook/bart-base"
DATASET_NAME = "cnn_dailymail"
DATASET_CONFIG = "3.0.0"
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
```

###  설명

* 사용할 사전학습 모델(BART)을 지정한다.
* CNN/Daily Mail 데이터셋 버전 3.0.0을 사용함
* 입력 텍스트와 요약 길이, 학습 관련 하이퍼파라미터(Batch Size, Epoch 수 등)를 설정

---

##  2. `utils.py`

```python
# utils.py
from transformers import AutoTokenizer

def preprocess_function(examples, tokenizer, max_input_len, max_target_len):
    inputs = examples["article"]
    targets = examples["highlights"]
    model_inputs = tokenizer(
        inputs, max_length=max_input_len, truncation=True, padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_len, truncation=True, padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

###  설명

* 기사 본문(`article`)과 요약문(`highlights`)을 토크나이징
* 입력과 출력 모두 정해진 길이로 자르고 패딩을 적용해 모델이 학습 가능한 형태로 변환

---

##  3. `main.py`

```python
# main.py
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq)
import evaluate
from config import *
from utils import preprocess_function

dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenized_datasets = dataset.map(
    lambda x: preprocess_function(x, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH),
    batched=True,
    remove_columns=dataset["train"].column_names,
)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    save_total_limit=2,
    predict_with_generate=True,
    logging_dir='./logs',
)

rouge = evaluate.load("rouge")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return rouge.compute(predictions=decoded_preds, references=decoded_labels)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

test_results = trainer.predict(tokenized_datasets["test"])
print("ROUGE 결과:", compute_metrics(test_results))
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
