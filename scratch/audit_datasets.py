"""Programmatic dataset audit, label mapping, conflict analysis, and candidate dataset builder for Nina Phase 4."""

import csv
import hashlib
import json
from pathlib import Path
import re

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Nina Canonical Emotion Taxonomy
CANONICAL_EMOTIONS = {"happy", "sadness", "anger", "fear", "love", "surprise"}

# Direct label mappings
DIRECT_LABEL_MAP = {
    # Joy / Happy
    "joy": "happy",
    "happy": "happy",
    "happiness": "happy",
    # Sadness
    "sadness": "sadness",
    "sad": "sadness",
    # Anger
    "anger": "anger",
    "angry": "anger",
    # Fear
    "fear": "fear",
    "fearful": "fear",
    "scared": "fear",
    # Love
    "love": "love",
    "loving": "love",
    # Surprise
    "surprise": "surprise",
    "surprised": "surprise",
}

# Ambiguous / Discarded labels requiring explicit documentation
AMBIGUOUS_LABELS = {
    "neutral",
    "worry",
    "hate",
    "disgust",
    "shame",
    "fun",
    "enthusiasm",
    "relief",
    "boredom",
    "empty",
    "enthusiasm",
    "pity",
}


def normalize_text(text: str) -> str:
    """Normalize text for duplicate checking without destroying content."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def generate_text_id(text: str) -> str:
    """Generate deterministic MD5 hash string for text sample."""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()[:12]


def audit_raw_datasets():
    print("==================================================")
    print("         NINA DATASET INTEGRATION AUDIT           ")
    print("==================================================\n")

    dataset_files = [
        "emotion_dataset.csv",
        "final_dataset.csv",
        "Emotion_Sentiment_DataSet.csv",
    ]

    all_records = []
    dataset_summaries = {}

    for file_name in dataset_files:
        file_path = RAW_DIR / file_name
        if not file_path.exists():
            print(f"ERROR: {file_name} not found in {RAW_DIR}")
            continue

        print(f"--- Auditing: {file_name} ---")
        rows_count = 0
        missing_count = 0
        duplicate_rows_count = 0

        raw_label_counts = {}
        text_lengths_char = []
        text_lengths_word = []

        seen_raw_rows = set()
        seen_raw_texts = set()
        unique_texts_count = 0
        duplicate_texts_count = 0

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            # Identify text and emotion column indices
            text_idx = -1
            emotion_idx = -1

            if header:
                for idx, col in enumerate(header):
                    col_clean = col.strip().lower()
                    if col_clean in ["text", "clean_text"] and text_idx == -1:
                        text_idx = idx
                    elif col_clean in ["emotion", "sentiment", "label"] and emotion_idx == -1:
                        emotion_idx = idx

            # Fallback column index defaults if header is non-standard
            if text_idx == -1:
                text_idx = 2 if len(header) > 2 else 0
            if emotion_idx == -1:
                emotion_idx = 1 if len(header) > 1 else 1

            print(f"Header: {header}")
            print(f"Detected text col idx: {text_idx}, emotion col idx: {emotion_idx}")

            for row_id, row in enumerate(reader):
                rows_count += 1
                row_tuple = tuple(row)

                if row_tuple in seen_raw_rows:
                    duplicate_rows_count += 1
                seen_raw_rows.add(row_tuple)

                if len(row) <= max(text_idx, emotion_idx):
                    missing_count += 1
                    continue

                raw_text = row[text_idx].strip()
                raw_emotion = row[emotion_idx].strip().lower()

                if not raw_text or not raw_emotion:
                    missing_count += 1
                    continue

                if raw_text in seen_raw_texts:
                    duplicate_texts_count += 1
                else:
                    unique_texts_count += 1
                    seen_raw_texts.add(raw_text)

                raw_label_counts[raw_emotion] = raw_label_counts.get(raw_emotion, 0) + 1
                text_lengths_char.append(len(raw_text))
                text_lengths_word.append(len(raw_text.split()))

                all_records.append({
                    "source_dataset": file_name,
                    "source_row_id": row_id,
                    "raw_text": raw_text,
                    "raw_label": raw_emotion,
                })

        avg_char = sum(text_lengths_char) / len(text_lengths_char) if text_lengths_char else 0
        avg_word = sum(text_lengths_word) / len(text_lengths_word) if text_lengths_word else 0

        dataset_summaries[file_name] = {
            "total_rows": rows_count,
            "columns": header,
            "missing_records": missing_count,
            "duplicate_rows": duplicate_rows_count,
            "unique_texts": unique_texts_count,
            "duplicate_texts": duplicate_texts_count,
            "avg_char_length": round(avg_char, 1),
            "avg_word_length": round(avg_word, 1),
            "raw_labels": raw_label_counts,
        }

        print(f"Total Rows:            {rows_count}")
        print(f"Missing/Malformed:     {missing_count}")
        print(f"Exact Duplicate Rows:  {duplicate_rows_count}")
        print(f"Unique Texts:          {unique_texts_count}")
        print(f"Avg Text Word Length:  {avg_word:.1f} words")
        print(f"Raw Label Counts:      {raw_label_counts}\n")

    return all_records, dataset_summaries


def process_and_deduplicate(all_records):
    print("==================================================")
    print("        LABEL MAPPING & CONFLICT ANALYSIS         ")
    print("==================================================\n")

    mapped_records = []
    discarded_records = []
    unmapped_labels = set()

    for rec in all_records:
        raw_label = rec["raw_label"]
        mapped = DIRECT_LABEL_MAP.get(raw_label)

        if mapped:
            rec["canonical_emotion"] = mapped
            mapped_records.append(rec)
        else:
            unmapped_labels.add(raw_label)
            discarded_records.append(rec)

    print(f"Total Raw Records Evaluated: {len(all_records)}")
    print(f"Successfully Mapped:        {len(mapped_records)}")
    print(f"Discarded / Unmapped:       {len(discarded_records)}")
    print(f"Discarded Raw Labels Found: {sorted(list(unmapped_labels))}\n")

    # Conflict Analysis
    norm_text_to_emotions = {}
    norm_text_to_records = {}

    for rec in mapped_records:
        norm_t = normalize_text(rec["raw_text"])
        emo = rec["canonical_emotion"]

        if norm_t not in norm_text_to_emotions:
            norm_text_to_emotions[norm_t] = set()
            norm_text_to_records[norm_t] = []

        norm_text_to_emotions[norm_t].add(emo)
        norm_text_to_records[norm_t].append(rec)

    conflicting_texts = {t: emos for t, emos in norm_text_to_emotions.items() if len(emos) > 1}
    print(f"Total Unique Normalized Texts: {len(norm_text_to_emotions)}")
    print(f"Label Conflict Instances:      {len(conflicting_texts)}")

    # Build Candidate Dataset (removing conflicts & exact duplicate texts)
    candidate_records = []
    seen_candidate_norm_texts = set()
    conflict_details = []

    for norm_t, rec_list in norm_text_to_records.items():
        emotions = norm_text_to_emotions[norm_t]
        if len(emotions) > 1:
            # Record conflict detail
            conflict_details.append({
                "normalized_text": norm_t,
                "conflicting_emotions": list(emotions),
                "sample_count": len(rec_list),
                "sources": list({r["source_dataset"] for r in rec_list}),
            })
            continue

        # Single consistent emotion - select the best representative record
        primary_rec = rec_list[0]
        rec_id = generate_text_id(primary_rec["raw_text"])

        candidate_records.append({
            "id": rec_id,
            "text": primary_rec["raw_text"],
            "emotion": primary_rec["canonical_emotion"],
            "source_dataset": primary_rec["source_dataset"],
            "source_label": primary_rec["raw_label"],
            "source_row_id": primary_rec["source_row_id"],
        })
        seen_candidate_norm_texts.add(norm_t)

    print(f"Clean Candidate Dataset Size:  {len(candidate_records)}\n")

    # Candidate Class Distribution
    candidate_class_counts = {}
    for r in candidate_records:
        emo = r["emotion"]
        candidate_class_counts[emo] = candidate_class_counts.get(emo, 0) + 1

    print("Candidate 6-Class Distribution:")
    total_cand = len(candidate_records)
    for emo in sorted(CANONICAL_EMOTIONS):
        cnt = candidate_class_counts.get(emo, 0)
        pct = (cnt / total_cand * 100) if total_cand > 0 else 0
        print(f"  {emo:<10} {cnt:>6} ({pct:.2f}%)")

    return candidate_records, conflict_details, candidate_class_counts


def create_splits_and_save(candidate_records):
    import random
    random.seed(42)

    shuffled = list(candidate_records)
    random.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * 0.80)
    val_end = int(total * 0.90)

    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]

    print(f"\nCreated Reproducible 80/10/10 Splits:")
    print(f"  Train Set:      {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"  Validation Set: {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    print(f"  Test Set:       {len(test_data)} ({len(test_data)/total*100:.1f}%)")

    # Verify Leakage
    train_texts = {normalize_text(r["text"]) for r in train_data}
    val_texts = {normalize_text(r["text"]) for r in val_data}
    test_texts = {normalize_text(r["text"]) for r in test_data}

    val_leak = train_texts.intersection(val_texts)
    test_leak = train_texts.intersection(test_texts).union(val_texts.intersection(test_texts))

    print(f"\nData Leakage Check:")
    print(f"  Train <-> Val Overlap:  {len(val_leak)}")
    print(f"  Train/Val <-> Test Overlap: {len(test_leak)}")

    # Save to data/processed/
    fieldnames = ["id", "text", "emotion", "source_dataset", "source_label", "source_row_id"]

    for name, data_set in [("candidate_all.csv", candidate_records), ("train.csv", train_data), ("val.csv", val_data), ("test.csv", test_data)]:
        out_path = PROCESSED_DIR / name
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_set)

    print(f"Successfully saved clean splits to {PROCESSED_DIR.resolve()}\n")


if __name__ == "__main__":
    records, summaries = audit_raw_datasets()
    candidate_recs, conflicts, class_counts = process_and_deduplicate(records)
    create_splits_and_save(candidate_recs)
