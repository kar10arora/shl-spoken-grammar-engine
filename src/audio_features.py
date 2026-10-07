"""
Audio Signal and Acoustic Feature Extraction Module for SHL Grammar Assessment.
Extracts physically grounded acoustic properties from raw 16kHz WAV files:
- Energy metrics: RMS amplitude, peak amplitude, dynamic range.
- Temporal / Silence metrics: Active speech ratio, silence percentage (-40dB threshold).
- Non-speech / Anomaly detection: Low-energy flags to reliably isolate the 0.0 MOS class.
"""

import os
import wave
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm


def extract_single_audio_features(wav_path: str) -> Dict[str, float]:
    """
    Extracts acoustic and temporal features from a single WAV audio file using standard library and numpy.
    No heavy audio dependencies required; fast, reproducible, and robust.
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    # Convert to 16-bit signed PCM numpy array
    if sampwidth == 2:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float64)
    elif sampwidth == 1:
        audio = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float64) - 128.0) * 256.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float64) / 65536.0
    else:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float64)

    # If stereo, average channels
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    duration = n_frames / float(framerate) if framerate > 0 else 0.0

    # Basic amplitude stats
    abs_audio = np.abs(audio)
    max_amp = float(np.max(abs_audio)) if len(audio) > 0 else 0.0
    mean_amp = float(np.mean(abs_audio)) if len(audio) > 0 else 0.0
    rms_energy = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0

    # Frame-level energy analysis (25ms window, 10ms hop)
    frame_size = int(0.025 * framerate)
    hop_size = int(0.010 * framerate)

    if len(audio) >= frame_size and hop_size > 0:
        num_frames = 1 + int((len(audio) - frame_size) / hop_size)
        # Vectorized frame slicing
        indices = np.arange(frame_size)[None, :] + np.arange(num_frames)[:, None] * hop_size
        frames = audio[indices]
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))

        # Peak normalized dB
        peak_rms = np.max(frame_rms) if len(frame_rms) > 0 else 1.0
        peak_rms = max(peak_rms, 1e-6)
        frame_db = 20.0 * np.log10(np.maximum(frame_rms / peak_rms, 1e-5))

        # Silence threshold: frames with energy lower than -35 dB from peak
        silence_mask = frame_db < -35.0
        silence_ratio = float(np.mean(silence_mask))
        speech_ratio = 1.0 - silence_ratio
        speech_duration = duration * speech_ratio

        # Energy dynamics
        energy_std = float(np.std(frame_rms))
        energy_skew = float(
            np.mean(((frame_rms - np.mean(frame_rms)) / (energy_std + 1e-6)) ** 3)
        )
        # Zero-crossing rate on speech
        sign_changes = np.diff(np.signbit(audio))
        zcr = float(np.mean(sign_changes))
    else:
        silence_ratio = 1.0
        speech_ratio = 0.0
        speech_duration = 0.0
        energy_std = 0.0
        energy_skew = 0.0
        zcr = 0.0

    # Near-zero energy detection (isolates 0.0 ground truth samples)
    is_silent_or_corrupt = 1.0 if (rms_energy < 50.0 or speech_ratio < 0.05 or max_amp < 200.0) else 0.0

    return {
        "duration_sec": duration,
        "sample_rate": framerate,
        "rms_energy": rms_energy,
        "max_amplitude": max_amp,
        "mean_amplitude": mean_amp,
        "silence_ratio": silence_ratio,
        "speech_ratio": speech_ratio,
        "speech_duration_sec": speech_duration,
        "energy_std": energy_std,
        "energy_skew": energy_skew,
        "zero_crossing_rate": zcr,
        "is_silent_or_corrupt": is_silent_or_corrupt,
    }


def extract_dataset_audio_features(
    audio_dir: str, metadata_df: pd.DataFrame, filename_col: str = "filename"
) -> pd.DataFrame:
    """
    Extracts acoustic features for all audio files referenced in a metadata dataframe.
    """
    features_list = []
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Extracting Audio Features"):
        fname = row[filename_col]
        wav_path = os.path.join(audio_dir, fname)
        feat = extract_single_audio_features(wav_path)
        feat[filename_col] = fname
        features_list.append(feat)

    feat_df = pd.DataFrame(features_list)
    return feat_df
