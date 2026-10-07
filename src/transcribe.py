"""
Audio Transcription Module using faster-whisper.
Runs optimized CPU inference (int8 quantized) with verbatim settings:
- condition_on_previous_text=False (prevents hallucinations & repetitive loops)
- temperature=0.0 (deterministic output)
- Preserves spoken disfluencies and grammar slips critical for scoring.
"""

import os
import time
import pandas as pd
from typing import Optional
from tqdm import tqdm


def get_whisper_model(model_size: str = "base.en", device: str = "cpu", compute_type: str = "int8"):
    """Loads faster-whisper WhisperModel."""
    from faster_whisper import WhisperModel
    print(f"Loading faster-whisper model: '{model_size}' on {device} ({compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return model


def transcribe_single_audio(
    model,
    wav_path: str,
    condition_on_prev: bool = False,
    temperature: float = 0.0,
) -> str:
    """Transcribes a single audio file, returning the full text transcript."""
    if not os.path.exists(wav_path):
        return ""

    try:
        segments, info = model.transcribe(
            wav_path,
            beam_size=1,
            language="en",
            temperature=temperature,
            condition_on_previous_text=condition_on_prev,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        text_segments = [seg.text.strip() for seg in segments if seg.text.strip()]
        full_text = " ".join(text_segments)
        return full_text
    except Exception as e:
        print(f"Error transcribing {wav_path}: {e}")
        return ""


def transcribe_dataset(
    model,
    audio_dir: str,
    metadata_df: pd.DataFrame,
    filename_col: str = "filename",
    output_cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Transcribes all audio files in a dataset dataframe, supporting resume from cache.
    """
    results = {}
    if output_cache_path and os.path.exists(output_cache_path):
        cached_df = pd.read_csv(output_cache_path)
        for _, row in cached_df.iterrows():
            results[row[filename_col]] = str(row.get("transcript", ""))
        print(f"Loaded {len(results)} existing transcripts from cache: {output_cache_path}")

    records = []
    start_time = time.time()
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Transcribing"):
        fname = row[filename_col]
        if fname in results and results[fname] != "":
            transcript = results[fname]
        else:
            wav_path = os.path.join(audio_dir, fname)
            transcript = transcribe_single_audio(model, wav_path)
            results[fname] = transcript

        rec = {filename_col: fname, "transcript": transcript}
        if "label" in row:
            rec["label"] = row["label"]
        records.append(rec)

        # Periodic checkpoint
        if output_cache_path and (idx + 1) % 50 == 0:
            pd.DataFrame(records).to_csv(output_cache_path, index=False)

    out_df = pd.DataFrame(records)
    if output_cache_path:
        out_df.to_csv(output_cache_path, index=False)
        print(f"Saved {len(out_df)} transcripts to {output_cache_path} (elapsed: {time.time() - start_time:.1f}s)")

    return out_df
