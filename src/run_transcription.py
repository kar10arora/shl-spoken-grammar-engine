"""
Batch Transcription Runner for Train and Test Datasets.
Uses faster-whisper on CPU with int8 quantization.
Saves checkpoints periodically to prevent any data loss.
"""

import os
import time
import pandas as pd
from faster_whisper import WhisperModel
from tqdm import tqdm

DATASET_DIR = r"c:\Users\user\Desktop\SHL\shl-hiring-assessment-2026\Dataset_Final"
OUTPUT_DIR = r"c:\Users\user\Desktop\SHL\data"


def transcribe_file(model, wav_path):
    if not os.path.exists(wav_path):
        return ""
    try:
        segments, info = model.transcribe(
            wav_path,
            language="en",
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
            beam_size=1,
        )
        texts = [seg.text.strip() for seg in segments if seg.text.strip()]
        return " ".join(texts)
    except Exception as e:
        print(f"Error on {wav_path}: {e}")
        return ""


def process_dataset(model, audio_dir, metadata_df, output_csv, name="Dataset"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    existing = {}
    if os.path.exists(output_csv):
        prev = pd.read_csv(output_csv)
        for _, row in prev.iterrows():
            if pd.notna(row.get("transcript")):
                existing[row["filename"]] = str(row["transcript"])
        print(f"[{name}] Resuming from existing cache: {len(existing)} / {len(metadata_df)} processed.")

    records = []
    start_t = time.time()
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc=f"Transcribing {name}"):
        fname = row["filename"]
        if fname in existing and existing[fname] != "":
            transcript = existing[fname]
        else:
            wav_path = os.path.join(audio_dir, fname)
            transcript = transcribe_file(model, wav_path)
            existing[fname] = transcript

        rec = {"filename": fname, "transcript": transcript}
        if "label" in row:
            rec["label"] = row["label"]
        records.append(rec)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(metadata_df):
            pd.DataFrame(records).to_csv(output_csv, index=False)

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False)
    print(f"[{name}] Finished {len(df_out)} samples in {time.time() - start_t:.1f}s -> {output_csv}")
    return df_out


def main():
    print("Initializing faster-whisper tiny.en on CPU int8...")
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=4)

    # 1. First process TEST set (216 samples)
    test_audio_dir = os.path.join(DATASET_DIR, "test")
    test_files = sorted(os.listdir(test_audio_dir))
    test_meta = pd.DataFrame({"filename": test_files})
    test_out_csv = os.path.join(OUTPUT_DIR, "test_transcripts.csv")
    process_dataset(model, test_audio_dir, test_meta, test_out_csv, name="Test Set")

    # 2. Process TRAIN set (769 samples)
    train_audio_dir = os.path.join(DATASET_DIR, "train")
    train_meta = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
    train_out_csv = os.path.join(OUTPUT_DIR, "train_transcripts.csv")
    process_dataset(model, train_audio_dir, train_meta, train_out_csv, name="Train Set")

    print("\nAll transcriptions completed successfully!")


if __name__ == "__main__":
    main()
