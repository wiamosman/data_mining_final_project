import time
import json
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    Trainer, TrainingArguments,
    # === MODEL/TOKENIZER IMPORTS ===
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    XLNetTokenizer, XLNetForSequenceClassification,
    BartTokenizer, BartForSequenceClassification,
    
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from functools import partial
import pandas as pd
import numpy as np

# 🚨 LINE TO CHANGE FOR DIFFERENT MODELS 🚨
MODEL_CHOICE = "BART" # "BERT", "ROBERTA", "XLNET", "BART"


TRAIN_SIZE = 22500
DEV_SIZE = 2500
TEST_SIZE = 25000 
DEV_SPLIT_RATIO = DEV_SIZE / (TRAIN_SIZE + DEV_SIZE) # 0.1

# Model-specific parameters lookup
MODEL_SPECS = {
    "BERT": {
        "tokenizer_class": BertTokenizer,
        "model_class": BertForSequenceClassification,
        "pretrained_name": "bert-base-uncased",
    },
    "ROBERTA": {
        "tokenizer_class": RobertaTokenizer,
        "model_class": RobertaForSequenceClassification,
        "pretrained_name": "roberta-base",
    },
    "XLNET": {
        "tokenizer_class": XLNetTokenizer,
        "model_class": XLNetForSequenceClassification,
        "pretrained_name": "xlnet-base-cased",
    },
    "BART": {
        "tokenizer_class": BartTokenizer,
        "model_class": BartForSequenceClassification,
        "pretrained_name": "facebook/bart-base",
    }
}

config = MODEL_SPECS[MODEL_CHOICE]
PRETRAINED_NAME = config["pretrained_name"]
OUTPUT_DIR = f"./{MODEL_CHOICE.lower()}-finetuned-imdb-custom-split"
MAX_LENGTH = 256
NUM_LABELS = 2
NUM_EPOCHS = 2
BATCH_SIZE = 8



# 2. Data Loading, Custom Splitting, and Preprocessing (FIXED)
print(f"--- Loading IMDb dataset and applying custom split ---")
raw_datasets = load_dataset("imdb")

# Split the original 'train' set into new 'train' and 'dev' sets
train_dev_split = raw_datasets["train"].train_test_split(
    test_size=DEV_SPLIT_RATIO, 
    seed=42, 
    stratify_by_column="label" 
)

# Create the final DatasetDict
custom_datasets = DatasetDict({
    'train': train_dev_split['train'],
    'dev': train_dev_split['test'],
    'test': raw_datasets['test']
})

print("\n--- Initial Dataset Split Sizes ---")
print(f"Train: {len(custom_datasets['train'])}")
print(f"Dev (Validation): {len(custom_datasets['dev'])}")
print(f"Test: {len(custom_datasets['test'])}")
print("-" * 30)

# Initialize tokenizer with model-specific adjustments
tokenizer = config["tokenizer_class"].from_pretrained(PRETRAINED_NAME)

# Adjust padding side for XLNet
if MODEL_CHOICE == "XLNET":
    tokenizer.padding_side = "left"

def preprocess_and_tokenize(examples, tokenizer, max_len):
    """
    Tokenizes the text and explicitly passes the labels to the output dictionary.
    Handles a batch of examples (`batched=True` on outer .map).
    """
    
    # 1. Tokenize the text
    tokenized_batch = tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=max_len
    )
    
    # 2. Add the original labels, renamed to 'labels' for the Trainer
    tokenized_batch["labels"] = examples["label"]
    
    return tokenized_batch

# Apply tokenization and label preparation to all three custom splits
tokenized_datasets = custom_datasets.map(
    partial(preprocess_and_tokenize, tokenizer=tokenizer, max_len=MAX_LENGTH),
    batched=True, 
    
    remove_columns=custom_datasets['train'].column_names 
)

def set_tensor_format(ds):
    """Sets the format to PyTorch tensors and selects the required columns."""
    
    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in ds.features:
        columns_to_keep.append("token_type_ids")
        
    # set_format modifies the dataset IN-PLACE
    ds.set_format(type="torch", columns=columns_to_keep)
    return ds 

for split in tokenized_datasets:
    set_tensor_format(tokenized_datasets[split])

# Final dataset assignments
train_dataset = tokenized_datasets["train"]
dev_dataset = tokenized_datasets["dev"]
test_dataset = tokenized_datasets["test"]

print(f"\n--- Final Tokenized Dataset Split Sizes ---")
print(f"Train: {len(train_dataset)}")
print(f"Dev (Validation): {len(dev_dataset)}")
print(f"Test: {len(test_dataset)}")
print("-" * 30)

# 3. Metrics and Trainer Setup
def compute_metrics(eval_pred):
    # eval_pred is an EvalPrediction object containing predictions and label_ids
    logits, labels = eval_pred 
    
    if isinstance(logits, tuple):
        logits = logits[0]
        
    preds = np.argmax(logits, axis=-1)
    
    # Return the dictionary of metrics
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average='binary'),
        "recall": recall_score(labels, preds, average='binary'),
        "f1": f1_score(labels, preds, average='binary'),
    }

# 4. Core Fine-Tuning Function with Time Logging
def run_fine_tuning():
    """Initializes, trains, evaluates, and saves results with detailed timing."""
    
    # 1. Initialize Model
    model = config["model_class"].from_pretrained(PRETRAINED_NAME, num_labels=NUM_LABELS)


    strategy = "epoch" 

    # 2. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        eval_strategy=strategy, 
        logging_dir=f"./logs/{MODEL_CHOICE.lower()}",
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to="none",
        load_best_model_at_end=False,
    )

    # 3. Trainer Setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics
    )

    # 4. Train and Capture Runtime
    print(f"\n--- Starting {MODEL_CHOICE} Training ---")
    
    train_output = trainer.train()
    
    train_time = train_output.metrics.get("train_runtime", 0.0)

    # 5. Save Model and Tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel and tokenizer saved to {OUTPUT_DIR}")

    # 6. Final Evaluation on TEST Set and Capture Runtime (Inference Time)
    print(f"\n--- Starting Final Evaluation on TEST Set ({TEST_SIZE} samples) ---")
    start_eval = time.time()
    final_metrics = trainer.evaluate(eval_dataset=test_dataset)
    end_eval = time.time()
    eval_time = end_eval - start_eval

    # 7. Calculate Total Time and Save Results
    total_time = train_time + eval_time
    
    # Add time metrics to the results dictionary
    final_metrics['runtime_train_with_validation_seconds'] = round(train_time, 2)
    final_metrics['runtime_test_evaluation_seconds'] = round(eval_time, 2)
    final_metrics['runtime_total_seconds'] = round(total_time, 2)
    
    # Save the full metrics to a model-specific JSON file
    metrics_filename = f"{MODEL_CHOICE.lower()}_final_test_metrics.json"
    with open(metrics_filename, "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"\n{MODEL_CHOICE} Final Test Evaluation Metrics (including runtimes) saved to {metrics_filename}:")
    print("-" * 70)
    print(json.dumps(final_metrics, indent=4))
    print("-" * 70)
    
    return final_metrics

# 5. Main Execution
if __name__ == "__main__":
    results = run_fine_tuning()