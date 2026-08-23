#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %% [markdown]
# # Album Intelligence Tool
# A Colab-ready notebook-style Python script for analyzing a small group of WAV
# files from the same project or album. The goal is to extract both technical
# signals and artist-friendly creative insights about individual songs, the
# collection as a whole, and how cohesive or distinctive the project feels.

# %%
# ## 1. Dependency Install Cell
# In Google Colab, run this once at the top. The check avoids reinstalling
# packages that are already available in the environment.

RUN_PIP_INSTALL = True

if RUN_PIP_INSTALL:
    import importlib.util
    import subprocess
    import sys

    REQUIRED_PACKAGES = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "librosa": "librosa",
        "seaborn": "seaborn",
        "soundfile": "soundfile",
    }

    missing_packages = [
        package_name
        for import_name, package_name in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(import_name) is None
    ]

    if missing_packages:
        print("Installing missing packages:", ", ".join(sorted(set(missing_packages))))
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *sorted(set(missing_packages))]
        )
    else:
        print("Required packages already available.")

# %%
# ## 2. Imports

import json
import math
import os
import re
import warnings
from collections import Counter
from datetime import datetime

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# %%
# ## 3. User Input Cell
# Define your WAV file paths here. These should point to files accessible from
# the Colab runtime, such as paths in `/content/`.

project_name = "Untitled Album Project"

wav_files = [
    "/content/song1.wav",
    "/content/song2.wav",
    "/content/song3.wav",
]

# Optional manual notes for each song. These are used only as supporting
# context when shaping the creative interpretation.
song_notes = {
    # "/content/song1.wav": "dark acoustic opener, spiritual tension, lonely but hopeful",
    # "/content/song2.wav": "more aggressive, tribal rhythm, wider emotional lift",
}

output_dir = "/content/album_intelligence_outputs"
export_results = True
create_plots = True
max_waveform_points = 5000

# %%
# ## 4. Helper Functions

KEY_NAMES = np.array(["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"])
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_STOPWORDS = {
    "and", "the", "with", "into", "from", "more", "less", "very", "dark", "light",
    "song", "track", "feel", "feels", "vibe", "vibes", "this", "that", "like",
    "but", "for", "its", "it's", "too", "not", "just", "still", "then", "than",
    "wide", "wider", "about", "across", "around", "over", "under", "hopeful",
}


def clamp(value, minimum=0.0, maximum=1.0):
    return float(max(minimum, min(maximum, value)))


def safe_mean(values, default=0.0):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float(default)
    return float(np.nanmean(values))


def seconds_to_mmss(seconds):
    seconds = float(max(0.0, seconds))
    minutes = int(seconds // 60)
    remaining = int(round(seconds % 60))
    return f"{minutes}:{remaining:02d}"


def normalize_vector(values):
    values = np.asarray(values, dtype=float)
    denominator = np.linalg.norm(values) + 1e-8
    return values / denominator


def summarize_value(value, thresholds, labels):
    for threshold, label in zip(thresholds, labels):
        if value <= threshold:
            return label
    return labels[-1]


def short_track_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0] or file_path


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def to_serializable(value):
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        if np.isnan(value):
            return None
        return value.item()
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def get_note_keywords(song_results, top_n=8):
    counter = Counter()
    for song in song_results:
        note = song.get("manual_note", "")
        if not note:
            continue
        tokens = re.findall(r"[a-zA-Z']+", note.lower())
        counter.update(token for token in tokens if len(token) >= 4 and token not in NOTE_STOPWORDS)
    return counter.most_common(top_n)


def preview_waveform(y, sr, max_points=5000):
    if len(y) <= max_points:
        indices = np.arange(len(y))
    else:
        indices = np.linspace(0, len(y) - 1, max_points).astype(int)
    return {
        "time_sec": (indices / sr).astype(float).tolist(),
        "amplitude": y[indices].astype(float).tolist(),
    }


def estimate_key_from_chroma(chroma_mean):
    chroma_mean = np.asarray(chroma_mean, dtype=float)
    chroma_norm = chroma_mean / (np.sum(chroma_mean) + 1e-8)
    candidates = []
    for i, key_name in enumerate(KEY_NAMES):
        major_score = np.corrcoef(chroma_norm, np.roll(MAJOR_PROFILE, i))[0, 1]
        minor_score = np.corrcoef(chroma_norm, np.roll(MINOR_PROFILE, i))[0, 1]
        candidates.append((major_score, key_name, "major"))
        candidates.append((minor_score, key_name, "minor"))
    score, root, mode = max(candidates, key=lambda item: np.nan_to_num(item[0], nan=-np.inf))
    top_pitch_classes = np.argsort(chroma_norm)[::-1][:3]
    top_pitch_names = [KEY_NAMES[idx] for idx in top_pitch_classes]
    return {
        "key": f"{root} {mode}",
        "root": root,
        "mode": mode,
        "confidence": float(np.nan_to_num(score, nan=0.0)),
        "top_pitch_classes": top_pitch_names,
        "chroma_mean": chroma_norm.astype(float).tolist(),
    }


def compute_beat_stability(beat_times):
    beat_times = np.asarray(beat_times, dtype=float)
    if beat_times.size < 3:
        return {"beat_count": int(beat_times.size), "beat_interval_cv": None, "beat_stability": 0.5}
    intervals = np.diff(beat_times)
    if np.mean(intervals) <= 0:
        return {"beat_count": int(beat_times.size), "beat_interval_cv": None, "beat_stability": 0.5}
    interval_cv = float(np.nan_to_num(stats.variation(intervals), nan=1.0))
    stability = clamp(1.0 - interval_cv, 0.0, 1.0)
    return {
        "beat_count": int(beat_times.size),
        "beat_interval_cv": interval_cv,
        "beat_stability": stability,
    }


def compute_instrumentation_density(mel_db):
    # A simple full-mix density heuristic: count how many mel bands are active
    # within 20 dB of the frame's local peak. It is not literal source counting,
    # but it gives a practical proxy for sparse vs. crowded arrangements.
    frame_peak = np.max(mel_db, axis=0, keepdims=True)
    relative_db = mel_db - frame_peak
    active_band_ratio = (relative_db > -20.0).mean(axis=0)
    smoothed_curve = gaussian_filter1d(active_band_ratio.astype(float), sigma=2)
    return {
        "density_curve": smoothed_curve.astype(float).tolist(),
        "density_mean": float(np.mean(active_band_ratio)),
        "density_peak": float(np.max(active_band_ratio)),
        "density_label": summarize_value(
            float(np.mean(active_band_ratio)),
            [0.18, 0.32, 0.46],
            ["sparse", "moderately layered", "dense", "crowded"],
        ),
    }


def approximate_sections(y, sr, hop_length=512):
    # Agglomerative clustering on MFCC frames gives a usable approximation of
    # section changes without forcing heavy segmentation dependencies.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    frame_count = mfcc.shape[1]
    if frame_count < 6:
        return {
            "section_count": 1,
            "sections": [{"label": "A", "start_sec": 0.0, "end_sec": float(len(y) / sr), "duration_sec": float(len(y) / sr)}],
        }
    rough_section_count = int(np.clip(round((len(y) / sr) / 45.0) + 3, 3, 8))
    try:
        boundaries = librosa.segment.agglomerative(mfcc, k=min(rough_section_count, frame_count - 1))
        boundaries = np.asarray(boundaries, dtype=int)
    except Exception:
        boundaries = np.linspace(0, frame_count - 1, min(rough_section_count, frame_count), dtype=int)
    boundaries = np.unique(np.concatenate(([0], boundaries, [frame_count - 1])))
    section_times = librosa.frames_to_time(boundaries, sr=sr, hop_length=hop_length)
    if section_times[-1] < (len(y) / sr):
        section_times = np.append(section_times, len(y) / sr)
    sections = []
    for idx in range(len(section_times) - 1):
        start_sec = float(section_times[idx])
        end_sec = float(section_times[idx + 1])
        sections.append(
            {
                "label": chr(65 + idx) if idx < 26 else f"S{idx + 1}",
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": max(0.0, end_sec - start_sec),
            }
        )
    return {"section_count": len(sections), "sections": sections}


def compute_repetition_metrics(chroma, beat_frames):
    if chroma.shape[1] < 8:
        return {
            "repetition_score": 0.0,
            "strong_recurrence_ratio": 0.0,
            "repetition_label": "too short to judge repetition",
        }
    if len(beat_frames) >= 6:
        try:
            sync_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        except Exception:
            sync_chroma = chroma
    else:
        stride = max(1, chroma.shape[1] // 64)
        sync_chroma = chroma[:, ::stride]
    recurrence = librosa.segment.recurrence_matrix(sync_chroma, mode="affinity", metric="cosine", sym=True)
    recurrence = np.asarray(recurrence, dtype=float)
    np.fill_diagonal(recurrence, 0.0)
    upper = recurrence[np.triu_indices_from(recurrence, k=1)]
    if upper.size == 0:
        repetition_score = 0.0
        strong_recurrence_ratio = 0.0
    else:
        repetition_score = float(np.nanmean(upper))
        strong_recurrence_ratio = float(np.mean(upper > 0.6))
    repetition_label = summarize_value(
        repetition_score,
        [0.20, 0.38, 0.55],
        [
            "more through-composed than loop-driven",
            "balanced between repetition and development",
            "motif-led with clear recurring ideas",
            "strongly cyclical and repetition-heavy",
        ],
    )
    return {
        "repetition_score": repetition_score,
        "strong_recurrence_ratio": strong_recurrence_ratio,
        "repetition_label": repetition_label,
    }


def compute_energy_arc(rms, times_sec):
    energy = gaussian_filter1d(np.asarray(rms, dtype=float), sigma=2)
    energy_norm = energy / (np.max(energy) + 1e-8)
    third = max(1, len(energy_norm) // 3)
    start_mean = float(np.mean(energy_norm[:third]))
    middle_mean = float(np.mean(energy_norm[third: 2 * third])) if len(energy_norm) >= 2 * third else float(np.mean(energy_norm))
    end_mean = float(np.mean(energy_norm[2 * third:])) if len(energy_norm) > 2 * third else float(np.mean(energy_norm[-third:]))
    peak_idx = int(np.argmax(energy_norm))
    peak_position_ratio = float(peak_idx / max(1, len(energy_norm) - 1))
    peak_time_sec = float(times_sec[peak_idx]) if len(times_sec) else 0.0

    if peak_position_ratio < 0.33:
        peak_label = "front-loaded"
    elif peak_position_ratio < 0.66:
        peak_label = "center-peaking"
    else:
        peak_label = "late-blooming"

    if end_mean > start_mean + 0.12 and peak_position_ratio >= 0.55:
        arc_summary = "builds from a leaner opening into a later, denser emotional peak"
    elif middle_mean > start_mean + 0.10 and end_mean < middle_mean - 0.08:
        arc_summary = "rises to a mid-song crest and then lets some pressure out"
    elif abs(end_mean - start_mean) <= 0.08:
        arc_summary = "holds a fairly even intensity across most of its runtime"
    elif end_mean < start_mean - 0.10:
        arc_summary = "starts with more force than it keeps by the end"
    else:
        arc_summary = "gradually accumulates pressure without relying on one huge payoff"

    return {
        "energy_curve": energy_norm.astype(float).tolist(),
        "start_mean": start_mean,
        "middle_mean": middle_mean,
        "end_mean": end_mean,
        "peak_position_ratio": peak_position_ratio,
        "peak_time_sec": peak_time_sec,
        "peak_label": peak_label,
        "arc_summary": arc_summary,
    }


def infer_mood_descriptors(features):
    descriptors = []

    tempo = features["tempo_bpm"]
    mode = features["key"]["mode"]
    brightness = features["spectral_centroid_mean"]
    dynamics = features["dynamic_range_db"]
    density = features["instrumentation_density"]["density_mean"]
    onset_density = features["onset_density_per_sec"]

    if mode == "minor":
        descriptors.append("melancholic")
    else:
        descriptors.append("resolving")

    if tempo < 78 and density < 0.25:
        descriptors.append("meditative")
    elif tempo < 105:
        descriptors.append("steady")
    elif tempo < 130:
        descriptors.append("driving")
    else:
        descriptors.append("urgent")

    if brightness < 1500:
        descriptors.append("dusky")
    elif brightness < 2500:
        descriptors.append("textured")
    else:
        descriptors.append("bright-edged")

    if dynamics > 18:
        descriptors.append("cinematic")
    elif dynamics < 10:
        descriptors.append("insistent")
    else:
        descriptors.append("breathing")

    if onset_density > 4.5:
        descriptors.append("kinetic")
    elif onset_density < 2.0:
        descriptors.append("floating")

    seen = set()
    filtered = []
    for descriptor in descriptors:
        if descriptor not in seen:
            filtered.append(descriptor)
            seen.add(descriptor)
    return filtered


def summarize_rhythmic_profile(features):
    tempo = features["tempo_bpm"]
    onset_density = features["onset_density_per_sec"]
    beat_stability = features["beat_metrics"]["beat_stability"]
    tempo_label = summarize_value(
        tempo,
        [75, 105, 130],
        ["slow-moving", "mid-tempo", "up-tempo", "fast-moving"],
    )
    onset_label = summarize_value(
        onset_density,
        [2.0, 4.5],
        ["low", "moderate", "high"],
    )
    if beat_stability >= 0.8:
        pulse_label = "a very steady pulse"
    elif beat_stability >= 0.6:
        pulse_label = "a reasonably stable groove"
    else:
        pulse_label = "an elastic or less settled pulse"
    return f"{tempo_label} pacing with {pulse_label} and {onset_label} onset activity"


def summarize_harmonic_profile(features):
    key = features["key"]["key"]
    tonal_motion_mean = features["tonal_motion_mean"]
    top_pitch_classes = ", ".join(features["key"]["top_pitch_classes"])
    motion_label = summarize_value(
        tonal_motion_mean,
        [0.08, 0.16, 0.26],
        [
            "harmonically settled",
            "gently shifting",
            "restless",
            "quite mobile",
        ],
    )
    return f"centered on {key}, {motion_label}, with pitch emphasis around {top_pitch_classes}"


def summarize_arrangement_arc(features):
    density_label = features["instrumentation_density"]["density_label"]
    repetition_label = features["repetition"]["repetition_label"]
    return f"{features['energy_arc']['arc_summary']}; overall texture feels {density_label} and {repetition_label}"


def summarize_song_plain_english(song_result):
    features = song_result["features"]
    name = song_result["track_name"]
    duration = seconds_to_mmss(features["duration_sec"])
    tempo = round(features["tempo_bpm"], 1)
    key = features["key"]["key"]
    mood = ", ".join(song_result["mood_descriptors"][:4])
    rhythmic = song_result["rhythmic_profile_summary"]
    harmonic = song_result["melodic_harmonic_profile_summary"]
    arrangement = song_result["arrangement_arc_summary"]

    summary = (
        f"{name} runs {duration} at about {tempo} BPM and points toward {key}. "
        f"It reads as {mood}. Rhythmically it has {rhythmic}. "
        f"Harmonically it feels {harmonic}. "
        f"In arrangement terms it {arrangement}."
    )

    note = song_result.get("manual_note", "")
    if note:
        summary += f" The manual note about '{note}' generally aligns with those signal-level traits."
    return summary


def track_identity_sentence(song_result):
    features = song_result["features"]
    identity_bits = []

    if features["energy_arc"]["peak_position_ratio"] >= 0.6:
        identity_bits.append("its late lift is a major part of the drama")
    elif features["energy_arc"]["peak_position_ratio"] <= 0.35:
        identity_bits.append("it makes its point early rather than saving everything for the end")

    if features["instrumentation_density"]["density_mean"] < 0.2:
        identity_bits.append("negative space is doing a lot of the emotional work")
    elif features["instrumentation_density"]["density_mean"] > 0.38:
        identity_bits.append("its impact depends on sustained layer density")

    if features["repetition"]["repetition_score"] > 0.45:
        identity_bits.append("recurring motifs seem central to its identity")
    else:
        identity_bits.append("it feels more developmental than loop-anchored")

    if features["tonal_motion_mean"] > 0.18:
        identity_bits.append("harmonic movement adds instability and forward pull")
    else:
        identity_bits.append("texture and contour matter more than harmonic volatility")

    return "What makes it feel like itself: " + "; ".join(identity_bits) + "."


def feature_vector_from_song(song_result):
    features = song_result["features"]
    vector = {
        "duration_sec": features["duration_sec"],
        "tempo_bpm": features["tempo_bpm"],
        "onset_density_per_sec": features["onset_density_per_sec"],
        "mean_rms_db": features["mean_rms_db"],
        "rms_std_db": features["rms_std_db"],
        "dynamic_range_db": features["dynamic_range_db"],
        "spectral_centroid_mean": features["spectral_centroid_mean"],
        "spectral_centroid_std": features["spectral_centroid_std"],
        "spectral_bandwidth_mean": features["spectral_bandwidth_mean"],
        "zcr_mean": features["zcr_mean"],
        "spectral_flatness_mean": features["spectral_flatness_mean"],
        "tonal_motion_mean": features["tonal_motion_mean"],
        "repetition_score": features["repetition"]["repetition_score"],
        "instrumentation_density_mean": features["instrumentation_density"]["density_mean"],
        "instrumentation_density_peak": features["instrumentation_density"]["density_peak"],
        "beat_stability": features["beat_metrics"]["beat_stability"],
        "section_count": features["sections"]["section_count"],
        "peak_position_ratio": features["energy_arc"]["peak_position_ratio"],
        "end_minus_start_energy": features["energy_arc"]["end_mean"] - features["energy_arc"]["start_mean"],
        "mode_binary": 1.0 if features["key"]["mode"] == "major" else 0.0,
    }
    chroma_mean = np.asarray(features["key"]["chroma_mean"], dtype=float)
    for idx, key_name in enumerate(KEY_NAMES):
        safe_name = key_name.lower().replace("#", "s").replace("b", "b")
        vector[f"chroma_{safe_name}"] = float(chroma_mean[idx])
    return vector


def pairwise_average(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] <= 1:
        return 1.0
    upper = values[np.triu_indices_from(values, k=1)]
    if upper.size == 0:
        return 1.0
    return float(np.mean(upper))


def subset_similarity(feature_df, columns):
    if len(feature_df) <= 1:
        return 1.0
    subset = feature_df[columns].astype(float).values
    scaled = StandardScaler().fit_transform(subset)
    similarity = (cosine_similarity(scaled) + 1.0) / 2.0
    return pairwise_average(similarity)


def build_project_identity_insights(album_profile):
    tempo_mean = album_profile["tempo_mean"]
    dominant_mode = album_profile["dominant_mode"]
    brightness_mean = album_profile["spectral_centroid_mean"]
    density_mean = album_profile["density_mean"]
    late_peak_ratio = album_profile["late_peak_ratio"]

    tempo_phrase = summarize_value(
        tempo_mean,
        [75, 105, 130],
        ["slow and spacious", "moderate-tempo", "energetic mid-fast", "fast-moving"],
    )
    brightness_phrase = summarize_value(
        brightness_mean,
        [1500, 2500],
        ["dark-leaning", "balanced-textural", "bright-edged"],
    )
    density_phrase = summarize_value(
        density_mean,
        [0.18, 0.32, 0.46],
        ["spare", "moderately layered", "dense", "thickly packed"],
    )

    arc_phrase = (
        "often building toward later peaks"
        if late_peak_ratio >= 0.6
        else "more evenly distributed in intensity"
        if late_peak_ratio <= 0.3
        else "balancing early statements with later lift"
    )

    return (
        f"The project identity is anchored in {tempo_phrase} motion, a {dominant_mode}-leaning tonal world, "
        f"{brightness_phrase} timbre, and arrangements that are generally {density_phrase}, {arc_phrase}."
    )


def infer_missing_element(album_profile):
    missing_ideas = []

    if album_profile["tempo_std"] < 8:
        missing_ideas.append("a stronger tempo contrast piece")
    if album_profile["dynamic_range_std"] < 3:
        missing_ideas.append("a track with more extreme dynamic breathing")
    if album_profile["mode_concentration"] > 0.8:
        missing_ideas.append("a clearer modal or emotional counterweight")
    if album_profile["late_peak_ratio"] > 0.75:
        missing_ideas.append("one song that states its thesis immediately instead of waiting to bloom")
    if album_profile["density_std"] < 0.05:
        missing_ideas.append("more contrast between stripped and fully loaded arrangements")

    if not missing_ideas:
        return "The set already has enough variance to feel intentional; the main need is sharper sequencing rather than a missing trait."

    if len(missing_ideas) == 1:
        return f"What may be missing is {missing_ideas[0]}."
    return "What may be missing is " + ", ".join(missing_ideas[:-1]) + f", and {missing_ideas[-1]}."


def normalize_across_tracks(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if math.isclose(minimum, maximum):
        return np.full_like(values, 0.5, dtype=float)
    return (values - minimum) / (maximum - minimum)


def choose_sequence_roles(song_results, average_similarity_scores):
    if not song_results:
        return {}

    tempos = np.array([song["features"]["tempo_bpm"] for song in song_results], dtype=float)
    late_peaks = np.array([song["features"]["energy_arc"]["peak_position_ratio"] for song in song_results], dtype=float)
    end_minus_start = np.array(
        [song["features"]["energy_arc"]["end_mean"] - song["features"]["energy_arc"]["start_mean"] for song in song_results],
        dtype=float,
    )
    dynamic_ranges = np.array([song["features"]["dynamic_range_db"] for song in song_results], dtype=float)
    brightness = np.array([song["features"]["spectral_centroid_mean"] for song in song_results], dtype=float)

    cohesion = normalize_across_tracks(average_similarity_scores)
    build = normalize_across_tracks(end_minus_start)
    dynamic_norm = normalize_across_tracks(dynamic_ranges)
    late_norm = normalize_across_tracks(late_peaks)
    dark_norm = 1.0 - normalize_across_tracks(brightness)
    moderate_tempo = 1.0 - np.abs(normalize_across_tracks(tempos) - 0.5)

    opener_score = 0.40 * cohesion + 0.35 * build + 0.25 * moderate_tempo
    centerpiece_score = 0.40 * dynamic_norm + 0.35 * (1.0 - cohesion) + 0.25 * late_norm
    closer_score = 0.40 * late_norm + 0.35 * build + 0.25 * dark_norm

    opener_idx = int(np.argmax(opener_score))
    centerpiece_idx = int(np.argmax(centerpiece_score))
    closer_idx = int(np.argmax(closer_score))

    return {
        "opener": song_results[opener_idx]["track_name"],
        "centerpiece": song_results[centerpiece_idx]["track_name"],
        "closer": song_results[closer_idx]["track_name"],
    }


def analyze_song(file_path, manual_note=""):
    track_name = short_track_name(file_path)

    if not os.path.isfile(file_path):
        return {
            "file_path": file_path,
            "track_name": track_name,
            "manual_note": manual_note,
            "status": "error",
            "error": "File not found.",
        }

    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        y = np.nan_to_num(y)
        if y.size == 0:
            raise ValueError("Audio file appears empty.")

        duration_sec = float(len(y) / sr)
        hop_length = 512
        frame_length = 2048

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
        rms_db = 20.0 * np.log10(np.maximum(rms, 1e-8))

        onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=sr,
            hop_length=hop_length,
            backtrack=False,
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        onset_density_per_sec = float(len(onset_times) / max(duration_sec, 1e-8))

        tempo_bpm, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sr,
            hop_length=hop_length,
            trim=False,
        )
        tempo_bpm = float(np.atleast_1d(tempo_bpm)[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        beat_metrics = compute_beat_stability(beat_times)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]

        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        except Exception:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

        chroma_mean = np.mean(chroma, axis=1)
        key_data = estimate_key_from_chroma(chroma_mean)
        tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sr)
        tonal_motion = np.linalg.norm(np.diff(tonnetz, axis=1), axis=0) if tonnetz.shape[1] > 1 else np.array([0.0])

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel_spec + 1e-10, ref=np.max)
        instrumentation_density = compute_instrumentation_density(mel_db)

        sections = approximate_sections(y, sr, hop_length=hop_length)
        repetition = compute_repetition_metrics(chroma, beat_frames)
        energy_arc = compute_energy_arc(rms, rms_times)

        dynamic_range_db = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
        mean_rms_db = float(np.mean(rms_db))
        rms_std_db = float(np.std(rms_db))

        features = {
            "sample_rate": int(sr),
            "duration_sec": duration_sec,
            "tempo_bpm": tempo_bpm,
            "beat_times_sec": beat_times.astype(float).tolist(),
            "onset_times_sec": onset_times.astype(float).tolist(),
            "onset_density_per_sec": onset_density_per_sec,
            "rms_time_sec": rms_times.astype(float).tolist(),
            "rms_curve": rms.astype(float).tolist(),
            "rms_db_curve": rms_db.astype(float).tolist(),
            "mean_rms_db": mean_rms_db,
            "rms_std_db": rms_std_db,
            "dynamic_range_db": dynamic_range_db,
            "spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "spectral_centroid_std": float(np.std(spectral_centroid)),
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
            "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "spectral_flatness_mean": float(np.mean(spectral_flatness)),
            "key": key_data,
            "tonal_motion_mean": float(np.mean(tonal_motion)),
            "tonal_motion_std": float(np.std(tonal_motion)),
            "sections": sections,
            "repetition": repetition,
            "energy_arc": energy_arc,
            "instrumentation_density": instrumentation_density,
            "waveform_preview": preview_waveform(y, sr, max_points=max_waveform_points),
            "waveform_peak_amplitude": float(np.max(np.abs(y))),
            "beat_metrics": beat_metrics,
        }

        mood_descriptors = infer_mood_descriptors(features)
        rhythmic_profile_summary = summarize_rhythmic_profile(features)
        melodic_harmonic_profile_summary = summarize_harmonic_profile(features)
        arrangement_arc_summary = summarize_arrangement_arc(features)

        song_result = {
            "file_path": file_path,
            "track_name": track_name,
            "manual_note": manual_note,
            "status": "ok",
            "features": features,
            "mood_descriptors": mood_descriptors,
            "rhythmic_profile_summary": rhythmic_profile_summary,
            "melodic_harmonic_profile_summary": melodic_harmonic_profile_summary,
            "arrangement_arc_summary": arrangement_arc_summary,
        }
        song_result["identity_summary"] = track_identity_sentence(song_result)
        song_result["plain_english_summary"] = summarize_song_plain_english(song_result)
        song_result["feature_vector"] = feature_vector_from_song(song_result)
        return song_result
    except Exception as exc:
        return {
            "file_path": file_path,
            "track_name": track_name,
            "manual_note": manual_note,
            "status": "error",
            "error": str(exc),
        }


def generate_coolness_suggestions(album_analysis):
    profile = album_analysis["album_profile"]
    suggestions = []

    identity_sentence = build_project_identity_insights(profile)
    suggestions.append(
        f"Lean harder into the clearest identity markers already present: {identity_sentence.lower()}"
    )

    if profile["tempo_std"] < 8:
        suggestions.append(
            "Introduce one sharper tempo departure so the project has a stronger contour; a single slower or more aggressive pulse could make the surrounding tracks feel more intentional."
        )

    if profile["mode_concentration"] > 0.8:
        suggestions.append(
            "Most of the material points toward the same tonal color. Keeping that core is good, but one track could provide a useful emotional reframing through a modal shift, borrowed-color harmony, or a more ambiguous tonic center."
        )

    if profile["density_std"] < 0.05:
        suggestions.append(
            "The arrangement density is fairly consistent across the set. To feel cooler and more memorable, exaggerate contrast: let one song stay bare much longer, and let another reach a genuinely maximal moment."
        )

    if profile["late_peak_ratio"] > 0.7:
        suggestions.append(
            "Many songs bloom late. That is a credible signature, but one immediate-impact track could sharpen the album arc by giving the listener an earlier statement of confidence."
        )

    if profile["overall_cohesion_score"] > 0.72:
        suggestions.append(
            "Cohesion is already strong, so avoid flattening the set further. Instead of making every song more similar, repeat one or two signature production gestures while widening rhythmic or structural contrast."
        )
    else:
        suggestions.append(
            "Cohesion is not yet doing enough work. Pick a few recurring anchors, such as one distinctive drum texture, a recurring harmonic color, or a specific type of intro space, and let them recur across multiple songs."
        )

    outliers = album_analysis.get("outlier_tracks", [])
    if outliers:
        suggestions.append(
            f"Treat {', '.join(outliers)} as a deliberate contrast case only if sequencing frames it that way; otherwise consider pulling one or two of its strongest sonic traits back toward the project's main language."
        )

    suggestions.append(
        "For future writing, try exaggeration instead of novelty for novelty's sake: take the strongest recurring trait in this set and write one song that pushes it further than the others in mood, pacing, and arrangement commitment."
    )

    return suggestions


def summarize_album_plain_english(album_analysis):
    profile = album_analysis["album_profile"]
    cohesion_label = album_analysis["cohesion_label"]
    song_count = album_analysis["valid_track_count"]
    tempo_mean = round(profile["tempo_mean"], 1)
    dominant_mode = profile["dominant_mode"]
    dominant_root = profile["dominant_root"]
    most_cohesive = ", ".join(album_analysis["most_cohesive_tracks"]) if album_analysis["most_cohesive_tracks"] else "n/a"
    outliers = ", ".join(album_analysis["outlier_tracks"]) if album_analysis["outlier_tracks"] else "none"
    mood_line = ", ".join(album_analysis["dominant_moods"][:4]) if album_analysis["dominant_moods"] else "mixed"
    identity = build_project_identity_insights(profile)

    return (
        f"Across {song_count} track(s), the collection feels {cohesion_label}. "
        f"The center of gravity sits around {tempo_mean} BPM, with a strong pull toward {dominant_root} and {dominant_mode} coloration. "
        f"The most cohesive songs are {most_cohesive}, while the main outlier(s) are {outliers}. "
        f"The emotional through-lines read as {mood_line}. {identity}"
    )


def analyze_album(song_results):
    valid_songs = [song for song in song_results if song["status"] == "ok"]
    invalid_songs = [song for song in song_results if song["status"] != "ok"]

    if not valid_songs:
        return {
            "status": "error",
            "message": "No valid songs were analyzed.",
            "valid_track_count": 0,
            "invalid_tracks": invalid_songs,
        }

    feature_df = pd.DataFrame(
        [{"track_name": song["track_name"], **song["feature_vector"]} for song in valid_songs]
    )
    numeric_features = feature_df.drop(columns=["track_name"]).astype(float)

    if len(valid_songs) == 1:
        similarity = np.ones((1, 1), dtype=float)
        pca_coords = np.array([[0.0, 0.0]])
        explained_variance_ratio = [1.0, 0.0]
        average_similarity_scores = np.array([1.0])
    else:
        scaled_features = StandardScaler().fit_transform(numeric_features.values)
        similarity = (cosine_similarity(scaled_features) + 1.0) / 2.0
        average_similarity_scores = (similarity.sum(axis=1) - 1.0) / (len(valid_songs) - 1)
        n_components = min(2, scaled_features.shape[0], scaled_features.shape[1])
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(scaled_features)
        if coords.shape[1] == 1:
            coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
        pca_coords = coords
        explained_variance_ratio = pca.explained_variance_ratio_.tolist()

    if len(valid_songs) == 1:
        most_cohesive_tracks = [valid_songs[0]["track_name"]]
        outlier_tracks = []
    else:
        ranking = sorted(
            zip([song["track_name"] for song in valid_songs], average_similarity_scores),
            key=lambda item: item[1],
            reverse=True,
        )
        most_cohesive_tracks = [ranking[0][0], ranking[1][0]] if len(ranking) > 1 else [ranking[0][0]]
        outlier_cutoff = float(np.mean(average_similarity_scores) - 0.5 * np.std(average_similarity_scores))
        outlier_tracks = [name for name, score in ranking[::-1] if score <= outlier_cutoff][:2]

    tempo_values = np.array([song["features"]["tempo_bpm"] for song in valid_songs], dtype=float)
    duration_values = np.array([song["features"]["duration_sec"] for song in valid_songs], dtype=float)
    dynamic_values = np.array([song["features"]["dynamic_range_db"] for song in valid_songs], dtype=float)
    centroid_values = np.array([song["features"]["spectral_centroid_mean"] for song in valid_songs], dtype=float)
    density_values = np.array([song["features"]["instrumentation_density"]["density_mean"] for song in valid_songs], dtype=float)
    peak_positions = np.array([song["features"]["energy_arc"]["peak_position_ratio"] for song in valid_songs], dtype=float)
    section_counts = np.array([song["features"]["sections"]["section_count"] for song in valid_songs], dtype=float)
    chroma_matrix = np.array([song["features"]["key"]["chroma_mean"] for song in valid_songs], dtype=float)

    key_counter = Counter(song["features"]["key"]["key"] for song in valid_songs)
    root_counter = Counter(song["features"]["key"]["root"] for song in valid_songs)
    mode_counter = Counter(song["features"]["key"]["mode"] for song in valid_songs)
    mood_counter = Counter(descriptor for song in valid_songs for descriptor in song["mood_descriptors"])

    cohesion_metrics = {
        "overall_similarity": pairwise_average(similarity),
        "tempo_alignment": clamp(1.0 - (np.std(tempo_values) / 35.0)),
        "tonal_alignment": pairwise_average(cosine_similarity(chroma_matrix)) if len(valid_songs) > 1 else 1.0,
        "dynamic_alignment": clamp(1.0 - (np.std(dynamic_values) / 10.0)),
        "timbral_alignment": subset_similarity(
            feature_df,
            ["spectral_centroid_mean", "spectral_bandwidth_mean", "zcr_mean", "spectral_flatness_mean"],
        ),
        "arrangement_alignment": clamp(
            1.0 - ((np.std(peak_positions) / 0.25) + (np.std(section_counts) / 2.5)) / 2.0
        ),
    }
    cohesion_metrics["overall_cohesion_score"] = float(np.mean(list(cohesion_metrics.values())))

    overall_cohesion = cohesion_metrics["overall_cohesion_score"]
    if overall_cohesion >= 0.78:
        cohesion_label = "strongly unified"
    elif overall_cohesion >= 0.60:
        cohesion_label = "balanced between unity and variety"
    else:
        cohesion_label = "somewhat scattered"

    note_keywords = get_note_keywords(valid_songs)
    sequence_roles = choose_sequence_roles(valid_songs, average_similarity_scores)

    album_profile = {
        "tempo_mean": float(np.mean(tempo_values)),
        "tempo_std": float(np.std(tempo_values)),
        "duration_mean_sec": float(np.mean(duration_values)),
        "dynamic_range_mean": float(np.mean(dynamic_values)),
        "dynamic_range_std": float(np.std(dynamic_values)),
        "spectral_centroid_mean": float(np.mean(centroid_values)),
        "density_mean": float(np.mean(density_values)),
        "density_std": float(np.std(density_values)),
        "late_peak_ratio": float(np.mean(peak_positions >= 0.6)),
        "dominant_key": key_counter.most_common(1)[0][0],
        "dominant_root": root_counter.most_common(1)[0][0],
        "dominant_mode": mode_counter.most_common(1)[0][0],
        "mode_concentration": float(mode_counter.most_common(1)[0][1] / len(valid_songs)),
        "section_count_mean": float(np.mean(section_counts)),
        "overall_cohesion_score": overall_cohesion,
    }

    if len(valid_songs) <= 6 and overall_cohesion >= 0.58:
        identity_scale = "The material currently reads more like a cohesive EP chapter than a sprawling full album."
    elif len(valid_songs) >= 7 and overall_cohesion >= 0.58:
        identity_scale = "The material suggests an album-scale identity rather than a loose collection of singles."
    else:
        identity_scale = "At this stage the material feels closer to a set of songs in orbit than to a fully locked project statement."

    recurring_signatures = [
        f"tempo center around {round(album_profile['tempo_mean'], 1)} BPM",
        f"{album_profile['dominant_mode']}-leaning tonal color",
        summarize_value(
            album_profile["spectral_centroid_mean"],
            [1500, 2500],
            ["dark-leaning timbre", "textural mid-bright timbre", "bright-edged timbre"],
        ),
        "late-arriving intensity arcs" if album_profile["late_peak_ratio"] >= 0.6 else "more evenly spread song arcs",
    ]

    if note_keywords:
        note_context = "Manual-note language repeatedly points toward " + ", ".join(word for word, _ in note_keywords[:5]) + "."
    else:
        note_context = "No manual note context was provided, so the conceptual reading is signal-driven only."

    album_analysis = {
        "status": "ok",
        "valid_track_count": len(valid_songs),
        "invalid_tracks": invalid_songs,
        "valid_songs": valid_songs,
        "feature_table": feature_df,
        "similarity_matrix": similarity,
        "average_similarity_scores": average_similarity_scores.tolist() if len(valid_songs) > 1 else [1.0],
        "pca_coordinates": pca_coords.tolist(),
        "pca_explained_variance_ratio": explained_variance_ratio,
        "cohesion_metrics": cohesion_metrics,
        "cohesion_label": cohesion_label,
        "album_profile": album_profile,
        "shared_bpm_tendencies": f"Most tracks orbit {round(album_profile['tempo_mean'], 1)} BPM with a spread of {round(album_profile['tempo_std'], 1)} BPM.",
        "shared_tonal_tendencies": f"The strongest tonal pull is toward {album_profile['dominant_root']} and a {album_profile['dominant_mode']}-leaning modal feel.",
        "shared_dynamic_behavior": f"Average dynamic range is {round(album_profile['dynamic_range_mean'], 1)} dB, with a spread of {round(album_profile['dynamic_range_std'], 1)} dB across tracks.",
        "shared_timbral_character": build_project_identity_insights(album_profile),
        "shared_arrangement_behavior": (
            "Many songs push their main peak later in the runtime."
            if album_profile["late_peak_ratio"] >= 0.6
            else "The set distributes its energy more evenly rather than depending on late blooms."
        ),
        "most_cohesive_tracks": most_cohesive_tracks,
        "outlier_tracks": outlier_tracks,
        "dominant_moods": [descriptor for descriptor, _ in mood_counter.most_common(6)],
        "recurring_stylistic_signatures": recurring_signatures,
        "identity_scale_statement": identity_scale,
        "note_context": note_context,
        "sequence_roles": sequence_roles,
    }

    album_analysis["project_identity_summary"] = summarize_album_plain_english(album_analysis)
    album_analysis["missing_element_summary"] = infer_missing_element(album_profile)
    album_analysis["coolness_suggestions"] = generate_coolness_suggestions(album_analysis)
    return album_analysis


def build_summary_table(song_results):
    valid_songs = [song for song in song_results if song["status"] == "ok"]
    if not valid_songs:
        return pd.DataFrame()
    rows = []
    for song in valid_songs:
        features = song["features"]
        rows.append(
            {
                "Track": song["track_name"],
                "Duration": seconds_to_mmss(features["duration_sec"]),
                "BPM": round(features["tempo_bpm"], 1),
                "Key": features["key"]["key"],
                "Onsets/sec": round(features["onset_density_per_sec"], 2),
                "DynRange dB": round(features["dynamic_range_db"], 1),
                "Centroid Hz": round(features["spectral_centroid_mean"], 0),
                "Density": features["instrumentation_density"]["density_label"],
                "Mood": ", ".join(song["mood_descriptors"][:3]),
            }
        )
    return pd.DataFrame(rows)


def plot_waveforms_and_energy(song_results, output_path=None):
    valid_songs = [song for song in song_results if song["status"] == "ok"]
    if not valid_songs:
        return None

    fig, axes = plt.subplots(len(valid_songs), 1, figsize=(14, max(3.5, 3.2 * len(valid_songs))), squeeze=False)
    axes = axes.flatten()

    for ax, song in zip(axes, valid_songs):
        features = song["features"]
        preview = features["waveform_preview"]
        rms_times = np.asarray(features["rms_time_sec"], dtype=float)
        energy_curve = np.asarray(features["energy_arc"]["energy_curve"], dtype=float)

        ax.plot(preview["time_sec"], preview["amplitude"], color="#4C78A8", linewidth=0.8, alpha=0.7, label="Waveform")
        ax.set_title(f"{song['track_name']} | waveform + normalized energy")
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0, max(preview["time_sec"]) if preview["time_sec"] else 1.0)

        twin = ax.twinx()
        twin.plot(rms_times, energy_curve, color="#F58518", linewidth=2.0, label="Energy")
        twin.set_ylabel("Normalized energy")
        twin.set_ylim(0, 1.05)

        ax.grid(alpha=0.2)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    return fig


def plot_tempo_key_table(summary_table, output_path=None):
    if summary_table.empty:
        return None

    fig_height = max(2.5, 0.55 * len(summary_table) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=summary_table.values,
        colLabels=summary_table.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.4)
    ax.set_title("Tempo / Key Summary Table", pad=14)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    return fig


def plot_similarity_heatmap(album_analysis, output_path=None):
    if album_analysis.get("status") != "ok" or album_analysis["valid_track_count"] < 2:
        return None

    names = [song["track_name"] for song in album_analysis["valid_songs"]]
    similarity_df = pd.DataFrame(album_analysis["similarity_matrix"], index=names, columns=names)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(similarity_df, annot=True, fmt=".2f", cmap="mako", vmin=0, vmax=1, ax=ax)
    ax.set_title("Pairwise Song Similarity")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    return fig


def plot_pca_map(album_analysis, output_path=None):
    if album_analysis.get("status") != "ok":
        return None

    coords = np.asarray(album_analysis["pca_coordinates"], dtype=float)
    if coords.size == 0:
        return None

    names = [song["track_name"] for song in album_analysis["valid_songs"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=120, color="#54A24B")
    for idx, name in enumerate(names):
        ax.text(coords[idx, 0] + 0.03, coords[idx, 1] + 0.03, name, fontsize=10)
    explained = album_analysis["pca_explained_variance_ratio"]
    x_pct = round(100 * explained[0], 1) if len(explained) >= 1 else 0.0
    y_pct = round(100 * explained[1], 1) if len(explained) >= 2 else 0.0
    ax.set_xlabel(f"PC1 ({x_pct}% var)")
    ax.set_ylabel(f"PC2 ({y_pct}% var)")
    ax.set_title("Song Map (PCA on Feature Vectors)")
    ax.axhline(0, color="lightgray", linewidth=0.8)
    ax.axvline(0, color="lightgray", linewidth=0.8)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    return fig


def plot_cohesion_chart(album_analysis, output_path=None):
    if album_analysis.get("status") != "ok":
        return None

    metrics = album_analysis["cohesion_metrics"]
    chart_df = pd.DataFrame(
        {
            "Metric": [
                "Overall similarity",
                "Tempo alignment",
                "Tonal alignment",
                "Dynamic alignment",
                "Timbral alignment",
                "Arrangement alignment",
            ],
            "Score": [
                metrics["overall_similarity"],
                metrics["tempo_alignment"],
                metrics["tonal_alignment"],
                metrics["dynamic_alignment"],
                metrics["timbral_alignment"],
                metrics["arrangement_alignment"],
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=chart_df, x="Metric", y="Score", palette="crest", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (0-1)")
    ax.set_xlabel("")
    ax.set_title(f"Album Cohesion Summary | {album_analysis['cohesion_label'].title()}")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    return fig


def generate_text_report(project_name, song_results, album_analysis):
    valid_songs = [song for song in song_results if song["status"] == "ok"]
    invalid_songs = [song for song in song_results if song["status"] != "ok"]

    lines = []
    lines.append("ALBUM INTELLIGENCE TOOL REPORT")
    lines.append("=" * 80)
    lines.append(f"Project: {project_name}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if album_analysis.get("status") != "ok":
        lines.append("No valid songs were analyzed.")
        if invalid_songs:
            lines.append("Errors:")
            for song in invalid_songs:
                lines.append(f"- {song['track_name']}: {song.get('error', 'Unknown error')}")
        return "\n".join(lines)

    lines.append("PROJECT SNAPSHOT")
    lines.append("-" * 80)
    lines.append(album_analysis["project_identity_summary"])
    lines.append(album_analysis["identity_scale_statement"])
    lines.append(album_analysis["shared_bpm_tendencies"])
    lines.append(album_analysis["shared_tonal_tendencies"])
    lines.append(album_analysis["shared_dynamic_behavior"])
    lines.append(album_analysis["shared_arrangement_behavior"])
    lines.append(album_analysis["note_context"])
    lines.append("")

    lines.append("TRACK-BY-TRACK INTELLIGENCE")
    lines.append("-" * 80)
    for song in valid_songs:
        lines.append(f"{song['track_name']}:")
        lines.append(song["plain_english_summary"])
        lines.append(song["identity_summary"])
        lines.append("")

    lines.append("COHESION, OUTLIERS, AND ALBUM ROLE")
    lines.append("-" * 80)
    lines.append(
        f"Most cohesive track(s): {', '.join(album_analysis['most_cohesive_tracks']) if album_analysis['most_cohesive_tracks'] else 'n/a'}."
    )
    lines.append(
        f"Outlier track(s): {', '.join(album_analysis['outlier_tracks']) if album_analysis['outlier_tracks'] else 'none'}."
    )
    lines.append(
        "Recurring stylistic signatures: "
        + ", ".join(album_analysis["recurring_stylistic_signatures"])
        + "."
    )
    lines.append(album_analysis["missing_element_summary"])
    lines.append(
        f"Sequencing suggestions: opener = {album_analysis['sequence_roles'].get('opener', 'n/a')}, "
        f"centerpiece = {album_analysis['sequence_roles'].get('centerpiece', 'n/a')}, "
        f"closer = {album_analysis['sequence_roles'].get('closer', 'n/a')}."
    )
    lines.append("")

    lines.append("CREATIVE DIRECTION")
    lines.append("-" * 80)
    lines.append(build_project_identity_insights(album_analysis["album_profile"]))
    lines.append(
        "The current sonic world suggests "
        + (
            "an introspective, cohesive chapter with strong internal logic."
            if album_analysis["cohesion_metrics"]["overall_cohesion_score"] >= 0.7
            else "a project still deciding how tightly its songs should relate to each other."
        )
    )
    lines.append("")

    lines.append("COOLNESS / DISTINCTIVENESS SUGGESTIONS")
    lines.append("-" * 80)
    for suggestion in album_analysis["coolness_suggestions"]:
        lines.append(f"- {suggestion}")
    lines.append("")

    if invalid_songs:
        lines.append("FILES WITH ERRORS")
        lines.append("-" * 80)
        for song in invalid_songs:
            lines.append(f"- {song['track_name']}: {song.get('error', 'Unknown error')}")
        lines.append("")

    return "\n".join(lines)


def export_outputs(results, summary_table, report_text, output_dir):
    ensure_output_dir(output_dir)

    results_json_path = os.path.join(output_dir, "album_intelligence_results.json")
    summary_csv_path = os.path.join(output_dir, "album_summary_table.csv")
    report_txt_path = os.path.join(output_dir, "album_report.txt")
    export_paths = {
        "results_json_path": results_json_path,
        "summary_csv_path": summary_csv_path if not summary_table.empty else None,
        "report_txt_path": report_txt_path,
    }
    results["exports"] = export_paths

    with open(results_json_path, "w", encoding="utf-8") as json_file:
        json.dump(to_serializable(results), json_file, indent=2)

    if not summary_table.empty:
        summary_table.to_csv(summary_csv_path, index=False)

    with open(report_txt_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_text)

    return export_paths


# %%
# ## 5. Per-Song Analysis Loop

if not wav_files:
    raise ValueError("Please add at least one WAV file path to the `wav_files` list.")

song_results = []

print(f"Analyzing project: {project_name}")
print(f"Requested files: {len(wav_files)}")
print("")

for file_path in wav_files:
    manual_note = song_notes.get(file_path, "")
    result = analyze_song(file_path, manual_note=manual_note)
    song_results.append(result)

    if result["status"] == "ok":
        features = result["features"]
        print(
            f"[OK] {result['track_name']}: "
            f"{seconds_to_mmss(features['duration_sec'])}, "
            f"{features['tempo_bpm']:.1f} BPM, "
            f"{features['key']['key']}, "
            f"{', '.join(result['mood_descriptors'][:3])}"
        )
    else:
        print(f"[ERROR] {result['track_name']}: {result.get('error', 'Unknown error')}")

print("")

# %%
# ## 6. Cross-Song Comparison

album_analysis = analyze_album(song_results)
summary_table = build_summary_table(song_results)

results = {
    "tool_name": "Album Intelligence Tool",
    "project_name": project_name,
    "generated_at": datetime.now().isoformat(),
    "input_files": wav_files,
    "song_notes": song_notes,
    "songs": song_results,
    "album_analysis": album_analysis,
}

if album_analysis.get("status") == "ok":
    print("Album/project summary")
    print(album_analysis["project_identity_summary"])
    print("")
    print("Shared tendencies")
    print("-", album_analysis["shared_bpm_tendencies"])
    print("-", album_analysis["shared_tonal_tendencies"])
    print("-", album_analysis["shared_dynamic_behavior"])
    print("-", album_analysis["shared_arrangement_behavior"])
    print("")
    print("Sequence roles")
    print("-", f"Opener candidate: {album_analysis['sequence_roles'].get('opener', 'n/a')}")
    print("-", f"Centerpiece candidate: {album_analysis['sequence_roles'].get('centerpiece', 'n/a')}")
    print("-", f"Closer candidate: {album_analysis['sequence_roles'].get('closer', 'n/a')}")
else:
    print("Album analysis could not be completed because no valid songs were available.")

# %%
# ## 7. Plotting Section

if create_plots and album_analysis.get("status") == "ok":
    if export_results:
        ensure_output_dir(output_dir)

    display(summary_table)

    plot_waveforms_and_energy(
        song_results,
        output_path=os.path.join(output_dir, "waveform_energy_overview.png") if export_results else None,
    )
    plot_tempo_key_table(
        summary_table,
        output_path=os.path.join(output_dir, "tempo_key_summary_table.png") if export_results else None,
    )
    plot_similarity_heatmap(
        album_analysis,
        output_path=os.path.join(output_dir, "song_similarity_heatmap.png") if export_results else None,
    )
    plot_pca_map(
        album_analysis,
        output_path=os.path.join(output_dir, "song_feature_map.png") if export_results else None,
    )
    plot_cohesion_chart(
        album_analysis,
        output_path=os.path.join(output_dir, "album_cohesion_chart.png") if export_results else None,
    )

# %%
# ## 8. Text Report Generation

report_text = generate_text_report(project_name, song_results, album_analysis)
print(report_text)

# %%
# ## 9. Optional Export to JSON / CSV / TXT

if export_results:
    export_paths = export_outputs(results, summary_table, report_text, output_dir)
    results["exports"] = export_paths
    print("")
    print("Exported files")
    for export_name, export_path in export_paths.items():
        if export_path:
            print(f"- {export_name}: {export_path}")
else:
    print("Export disabled; results remain in memory only.")
