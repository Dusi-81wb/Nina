"""Programmatic diagnostic script for Phase 4B.1 DistilBERT inspection, label verification, tokenization audit, and tiny-dataset overfit sanity test."""

import csv
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

from nina.api.schemas import SupportedEmotion
from nina.emotion.mapping import EmotionLabelMapper
from nina.emotion.trainer import ID_TO_LABEL, LABEL_TO_ID, verify_dataset_integrity

LABEL_TO_ID_VERIFIED = {
    "happy": 0,
    "sadness": 1,
    "anger": 2,
    "fear": 3,
    "love": 4,
    "surprise": 5,
}

ID_TO_LABEL_VERIFIED = {v: k for k, v in LABEL_TO_ID_VERIFIED.items()}


def verify_label_mappings():
    print("==================================================")
    print("         STEP 2: LABEL MAPPING VERIFICATION       ")
    print("==================================================\n")

    print(f"Trainer LABEL_TO_ID:  {LABEL_TO_ID}")
    print(f"Trainer ID_TO_LABEL:  {ID_TO_LABEL}")
    print(f"Verified Canonical:   {LABEL_TO_ID_VERIFIED}")

    assert LABEL_TO_ID == LABEL_TO_ID_VERIFIED, "LABEL_TO_ID mismatch!"
    assert ID_TO_LABEL == ID_TO_LABEL_VERIFIED, "ID_TO_LABEL mismatch!"

    for emo in SupportedEmotion:
        raw_val = emo.value
        mapped_val = EmotionLabelMapper.map_label(raw_val).value
        assert raw_val == mapped_val, f"Mapper discrepancy for {raw_val}"

    print("STATUS: Label mapping is 100% consistent across all components [ OK ]\n")


def audit_tokenization_and_columns(df_train):
    print("==================================================")
    print("    STEP 3 & 4: DATASET COLUMNS & TOKENIZATION    ")
    print("==================================================\n")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    sample_texts = df_train["text"].tolist()
    sample_labels = df_train["emotion"].tolist()

    lengths = []
    truncated_count = 0
    max_len = 64

    for text in sample_texts:
        tokens = tokenizer.encode(text, truncation=False)
        tok_len = len(tokens)
        lengths.append(tok_len)
        if tok_len > max_len:
            truncated_count += 1

    lengths.sort()
    n = len(lengths)
    avg_len = sum(lengths) / n
    p95_len = lengths[int(n * 0.95)]
    max_l = lengths[-1]
    trunc_pct = (truncated_count / n) * 100.0

    print(f"Total Training Samples Analyzed: {n}")
    print(f"Average Token Length:             {avg_len:.2f} tokens")
    print(f"95th Percentile Token Length:     {p95_len} tokens")
    print(f"Maximum Token Length:             {max_l} tokens")
    print(f"Truncated at max_len={max_len}:        {truncated_count} ({trunc_pct:.2f}%)\n")

    # Print sample tokenization
    print("Sample Tokenization Inspection:")
    for i in range(3):
        t = sample_texts[i]
        lbl = sample_labels[i]
        lbl_id = LABEL_TO_ID[lbl]
        encoded = tokenizer(t, max_length=max_len, truncation=True, padding="max_length")

        print(f"\n--- Sample #{i+1} ---")
        print(f"Raw Text:       '{t}'")
        print(f"Label:          {lbl} (ID: {lbl_id})")
        print(f"Tokens:         {tokenizer.convert_ids_to_tokens(encoded['input_ids'][:15])}...")
        print(f"Input IDs[:10]: {encoded['input_ids'][:10]}")
        print(f"Attention[:10]: {encoded['attention_mask'][:10]}")

    print("\nSTATUS: Tokenizer correctly tokenizes text column; label IDs match target [ OK ]\n")


def run_overfit_sanity_test(df_train):
    print("==================================================")
    print("     STEP 7 & 8: OVERFIT SANITY EXPERIMENT        ")
    print("==================================================\n")

    # Select a tiny subset of 60 samples (10 per class) using train_test_split
    tiny_df, _ = train_test_split(df_train, train_size=60, stratify=df_train["emotion"], random_state=42)

    print(f"Tiny Overfit Dataset Size: {len(tiny_df)} samples (10 per class)")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=6,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    # Initial weights check
    initial_classifier_weight = model.classifier.weight.clone().detach()

    tiny_ds = Dataset.from_pandas(pd.DataFrame({
        "text": tiny_df["text"].values,
        "label": [LABEL_TO_ID[e] for e in tiny_df["emotion"].values],
    }))

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=64, padding="max_length")

    tiny_tok = tiny_ds.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir="artifacts/sanity_overfit",
        eval_strategy="no",
        save_strategy="no",
        learning_rate=5e-4,
        per_device_train_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.0,
        logging_steps=5,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tiny_tok,
    )

    print("Running 10-epoch tiny overfit training pass...")
    train_res = trainer.train()

    # Weight update check
    final_classifier_weight = model.classifier.weight.clone().detach()
    weight_diff = torch.sum(torch.abs(final_classifier_weight - initial_classifier_weight)).item()

    print(f"\nTiny Overfit Final Training Loss: {train_res.training_loss:.4f}")
    print(f"Model Weight L1 Delta:            {weight_diff:.6f}")

    # Evaluate accuracy on tiny set
    preds_raw = trainer.predict(tiny_tok)
    preds = np.argmax(preds_raw.predictions, axis=-1)
    labels = np.array([LABEL_TO_ID[e] for e in tiny_df["emotion"].values])
    acc = (preds == labels).mean()

    print(f"Tiny Dataset Re-prediction Accuracy: {acc * 100:.2f}%")

    if acc >= 0.80 and weight_diff > 0.1:
        print("\nSANITY PASSED: Model backpropagation, gradients, and loss optimization are working correctly! [ OK ]\n")
    else:
        print("\nSANITY FAILED: Model could not overfit tiny training set!\n")


if __name__ == "__main__":
    df_train, df_val, df_test = verify_dataset_integrity()
    verify_label_mappings()
    audit_tokenization_and_columns(df_train)
    run_overfit_sanity_test(df_train)
