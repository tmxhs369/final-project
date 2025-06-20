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
