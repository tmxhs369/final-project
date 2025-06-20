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
