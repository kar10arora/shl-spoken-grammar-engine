"""
Linguistic and Grammatical Feature Extraction Module for SHL Grammar Assessment.
Engineered specifically against the competition rubric:
- Syntactic Complexity: Mean clause length, sentence length variance, subordination ratio.
- Grammatical Control: Subordinating conjunctions, relative pronouns, complex modals, passive markers.
- Disfluency & Self-Correction: Spoken fillers, word repetitions, self-repair markers.
- Lexical Diversity: Type-Token Ratio (TTR), Guiraud's Index, polysyllabic word density.
- Sentence Completeness: Incomplete sentence ratio, dangling conjunctions.
"""

import re
import math
from typing import Dict, List
import pandas as pd
from tqdm import tqdm


SUBORDINATING_CONJUNCTIONS = {
    "although", "because", "since", "unless", "while", "whereas", "whether",
    "even though", "in order that", "provided that", "as long as", "so that",
    "whenever", "wherever", "after", "before", "once", "though", "until"
}

COORDINATING_CONJUNCTIONS = {"and", "but", "or", "so", "for", "nor", "yet"}

RELATIVE_PRONOUNS = {"which", "who", "whom", "whose", "that", "whereby", "wherein"}

COMPLEX_MODALS = {
    "would have", "could have", "should have", "might have", "must have",
    "would", "could", "should", "might", "ought to"
}

SELF_CORRECTION_MARKERS = {
    "i mean", "sorry", "rather", "or rather", "i should say", "let me correct",
    "actually no", "excuse me"
}

FILLER_WORDS = {"um", "uh", "er", "ah", "hmm", "like", "you know"}


def count_syllables(word: str) -> int:
    """Estimates the number of syllables in an English word."""
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    # Remove silent trailing e
    word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
    word = re.sub(r'^y', '', word)
    matches = re.findall(r'[aeiouy]{1,2}', word)
    return max(1, len(matches))


def extract_single_linguistic_features(
    text: str, duration_sec: float = 60.0, speech_duration_sec: float = 60.0
) -> Dict[str, float]:
    """
    Extracts comprehensive grammatical, syntactic, and fluency features from transcribed text.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return {
            "num_words": 0.0,
            "num_unique_words": 0.0,
            "ttr": 0.0,
            "guiraud_index": 0.0,
            "mean_word_length": 0.0,
            "polysyllable_ratio": 0.0,
            "num_sentences": 0.0,
            "mean_sentence_length": 0.0,
            "std_sentence_length": 0.0,
            "subordination_count": 0.0,
            "subordination_ratio": 0.0,
            "relative_pronoun_count": 0.0,
            "complex_modal_count": 0.0,
            "filler_count": 0.0,
            "filler_ratio": 0.0,
            "repetition_count": 0.0,
            "repetition_ratio": 0.0,
            "self_correction_count": 0.0,
            "incomplete_sentence_ratio": 1.0,
            "words_per_minute": 0.0,
            "speech_rate": 0.0,
        }

    # Tokenize words
    words = re.findall(r"\b[A-Za-z0-9'-]+\b", cleaned_text.lower())
    num_words = len(words)
    if num_words == 0:
        return extract_single_linguistic_features("", duration_sec, speech_duration_sec)

    unique_words = set(words)
    num_unique = len(unique_words)
    ttr = num_unique / num_words
    guiraud = num_unique / math.sqrt(num_words)

    # Word length and syllables
    word_lengths = [len(w) for w in words]
    mean_word_len = sum(word_lengths) / num_words
    syllable_counts = [count_syllables(w) for w in words]
    polysyllables = sum(1 for s in syllable_counts if s >= 3)
    polysyllable_ratio = polysyllables / num_words

    # Sentence segmentation
    sentences = [s.strip() for s in re.split(r"[.!?]+", cleaned_text) if s.strip()]
    num_sentences = max(1, len(sentences))

    sentence_lens = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    mean_sent_len = sum(sentence_lens) / num_sentences
    var_sent_len = sum((l - mean_sent_len) ** 2 for l in sentence_lens) / num_sentences
    std_sent_len = math.sqrt(var_sent_len)

    # Syntactic complexity: Subordinations & Relative Clauses
    lower_text = cleaned_text.lower()
    subordination_count = 0
    for conj in SUBORDINATING_CONJUNCTIONS:
        subordination_count += len(re.findall(rf"\b{re.escape(conj)}\b", lower_text))
    subordination_ratio = subordination_count / max(1, num_sentences)

    relative_pronoun_count = 0
    for rel in RELATIVE_PRONOUNS:
        relative_pronoun_count += len(re.findall(rf"\b{re.escape(rel)}\b", lower_text))

    complex_modal_count = 0
    for modal in COMPLEX_MODALS:
        complex_modal_count += len(re.findall(rf"\b{re.escape(modal)}\b", lower_text))

    # Disfluency & repetitions
    filler_count = 0
    for filler in FILLER_WORDS:
        filler_count += len(re.findall(rf"\b{re.escape(filler)}\b", lower_text))
    filler_ratio = filler_count / num_words

    # Consecutive word repetitions (e.g. "I I went", "the the")
    repetition_count = sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])
    repetition_ratio = repetition_count / num_words

    # Self corrections
    self_correction_count = 0
    for marker in SELF_CORRECTION_MARKERS:
        self_correction_count += len(re.findall(rf"\b{re.escape(marker)}\b", lower_text))

    # Incomplete sentences: ending with conjunction or preposition or lacking finite verb
    incomplete_count = 0
    for s in sentences:
        s_words = s.lower().split()
        if s_words and s_words[-1] in {"and", "because", "but", "so", "or", "that", "which", "to", "with"}:
            incomplete_count += 1
    incomplete_ratio = incomplete_count / num_sentences

    # Speaking rates
    active_time = max(1.0, speech_duration_sec)
    total_time = max(1.0, duration_sec)
    words_per_minute = (num_words / total_time) * 60.0
    speech_rate = (num_words / active_time) * 60.0

    return {
        "num_words": float(num_words),
        "num_unique_words": float(num_unique),
        "ttr": float(ttr),
        "guiraud_index": float(guiraud),
        "mean_word_length": float(mean_word_len),
        "polysyllable_ratio": float(polysyllable_ratio),
        "num_sentences": float(num_sentences),
        "mean_sentence_length": float(mean_sent_len),
        "std_sentence_length": float(std_sent_len),
        "subordination_count": float(subordination_count),
        "subordination_ratio": float(subordination_ratio),
        "relative_pronoun_count": float(relative_pronoun_count),
        "complex_modal_count": float(complex_modal_count),
        "filler_count": float(filler_count),
        "filler_ratio": float(filler_ratio),
        "repetition_count": float(repetition_count),
        "repetition_ratio": float(repetition_ratio),
        "self_correction_count": float(self_correction_count),
        "incomplete_sentence_ratio": float(incomplete_ratio),
        "words_per_minute": float(words_per_minute),
        "speech_rate": float(speech_rate),
    }


def extract_dataset_linguistic_features(
    transcripts_df: pd.DataFrame,
    text_col: str = "transcript",
    duration_col: str = "duration_sec",
    speech_duration_col: str = "speech_duration_sec",
    filename_col: str = "filename",
) -> pd.DataFrame:
    """Extracts linguistic features for all transcripts."""
    features_list = []
    for _, row in tqdm(transcripts_df.iterrows(), total=len(transcripts_df), desc="Extracting Linguistic Features"):
        text = str(row.get(text_col, ""))
        dur = float(row.get(duration_col, 60.0))
        speech_dur = float(row.get(speech_duration_col, 60.0))
        fname = row[filename_col]

        feat = extract_single_linguistic_features(text, dur, speech_dur)
        feat[filename_col] = fname
        features_list.append(feat)

    return pd.DataFrame(features_list)
