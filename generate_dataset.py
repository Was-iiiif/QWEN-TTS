
import csv
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

ROOT = Path(__file__).resolve().parent

REFERENCE_FILE = ROOT / "dev_metadata_sampled_500.csv"
TARGET_FILE = ROOT / "missing_audio_metadata.csv"

REFERENCE_DIR = ROOT / "references"
OUTPUT_DIR = ROOT / "generated" / "clean"

METADATA_FILE = ROOT / "metadata.csv"
MISSING_FILE = ROOT / "missing_audio_report.csv"
MAPPING_FILE = ROOT / "speaker_mapping.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(8)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    """Convert a CSV value to a cleaned string."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_filename(value):
    """Make a speaker ID safe to use in a filename."""
    value = re.sub(r"[^A-Za-z0-9_-]", "_", str(value))
    return value[:80] or "unknown"


def save_csv(path, rows, fields):
    """Write CSV safely with a consistent column structure."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 3. LOAD REFERENCE RECORDINGS
# ============================================================

def load_references():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference CSV not found: {REFERENCE_FILE}"
        )

    df = pd.read_csv(REFERENCE_FILE, encoding="utf-8-sig")

    required = {"filename", "contributor_id", "transcript"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Reference CSV must contain {required}. "
            f"Found: {list(df.columns)}"
        )

    df["filename"] = df["filename"].apply(clean_text)
    df["contributor_id"] = df["contributor_id"].apply(clean_text)
    df["transcript"] = df["transcript"].apply(clean_text)

    df = df[
        (df["filename"] != "") &
        (df["contributor_id"] != "") &
        (df["transcript"] != "")
    ].copy()

    # Sort first so reference selection is reproducible.
    df = df.sort_values(
        ["contributor_id", "filename"]
    ).reset_index(drop=True)

    references = {}
    missing_reference_files = []

    for _, row in df.iterrows():
        contributor_id = row["contributor_id"]
        filename = row["filename"]

        # The CSV may contain a filename or a relative path.
        audio_path = Path(filename)

        if not audio_path.is_absolute():
            audio_path = REFERENCE_DIR / audio_path

        if not audio_path.is_file():
            missing_reference_files.append(filename)
            continue

        # Select one available recording per contributor.
        # The first valid recording after sorting is used.
        if contributor_id not in references:
            references[contributor_id] = {
                "contributor_id": contributor_id,
                "audio": audio_path,
                "text": row["transcript"]
            }

    if not references:
        raise ValueError(
            "No usable reference WAV files found. "
            "Check the filename column and references folder."
        )

    print(f"Reference contributors loaded: {len(references)}")
    print(f"Missing reference WAV entries: "
          f"{len(missing_reference_files)}")

    return references


# ============================================================
# 4. LOAD TARGET SENTENCES
# ============================================================

def load_targets():
    if not TARGET_FILE.exists():
        raise FileNotFoundError(
            f"Target CSV not found: {TARGET_FILE}"
        )

    df = pd.read_csv(TARGET_FILE, encoding="utf-8-sig")

    required = {"speaker_id", "transcript"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Target CSV must contain {required}. "
            f"Found: {list(df.columns)}"
        )

    df["speaker_id"] = df["speaker_id"].apply(clean_text)
    df["transcript"] = df["transcript"].apply(clean_text)

    df = df[
        (df["speaker_id"] != "") &
        (df["transcript"] != "")
    ].copy()

    # Preserve the original row number for unique output filenames.
    df["_source_row"] = df.index + 2
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid target sentences found.")

    print(f"Target records loaded: {len(df)}")
    print(f"Target speakers: {df['speaker_id'].nunique()}")

    return df


# ============================================================
# 5. CREATE / LOAD STABLE SPEAKER MAPPING
# ============================================================

def load_speaker_mapping(target_df, references):
    target_ids = sorted(target_df["speaker_id"].unique())
    reference_ids = sorted(references.keys())

    mapping = {}

    # Reuse previously saved assignments when possible.
    if MAPPING_FILE.exists():
        old = pd.read_csv(
            MAPPING_FILE,
            dtype=str,
            encoding="utf-8-sig"
        )

        required = {
            "target_speaker_id",
            "reference_contributor_id"
        }

        if required.issubset(old.columns):
            for _, row in old.iterrows():
                target_id = clean_text(row["target_speaker_id"])
                reference_id = clean_text(
                    row["reference_contributor_id"]
                )

                if (
                    target_id in target_ids
                    and reference_id in references
                ):
                    mapping[target_id] = reference_id

    # Assign new target speakers reproducibly.
    # Existing mappings remain unchanged.
    rng = random.Random(SEED)
    available = [
        rid for rid in reference_ids
        if rid not in mapping.values()
    ]
    rng.shuffle(available)

    for target_id in target_ids:
        if target_id in mapping:
            continue

        if available:
            reference_id = available.pop()
        else:
            # More target speakers than reference contributors:
            # some reference voices must be reused.
            reference_id = rng.choice(reference_ids)

        mapping[target_id] = reference_id

    rows = []

    for target_id in target_ids:
        reference_id = mapping[target_id]
        ref = references[reference_id]

        rows.append({
            "target_speaker_id": target_id,
            "reference_contributor_id": reference_id,
            "reference_audio": str(ref["audio"].relative_to(ROOT))
                .replace("\\", "/"),
            "reference_transcript": ref["text"]
        })

    save_csv(
        MAPPING_FILE,
        rows,
        [
            "target_speaker_id",
            "reference_contributor_id",
            "reference_audio",
            "reference_transcript"
        ]
    )

    print(f"Speaker mapping saved: {MAPPING_FILE}")
    print(f"Reference voices available: {len(reference_ids)}")

    return mapping


# ============================================================
# 6. LOAD EXISTING OUTPUT METADATA
# ============================================================

METADATA_FIELDS = [
    "audio_file",
    "target_speaker_id",
    "reference_contributor_id",
    "text",
    "sample_rate",
    "duration_seconds",
    "status",
    "error"
]

MISSING_FIELDS = [
    "audio_file",
    "target_speaker_id",
    "text",
    "status",
    "error"
]


def load_existing_metadata():
    if not METADATA_FILE.exists():
        return {}

    df = pd.read_csv(
        METADATA_FILE,
        dtype=str,
        encoding="utf-8-sig"
    )

    records = {}

    if "audio_file" in df.columns:
        for _, row in df.iterrows():
            audio_file = clean_text(row.get("audio_file", ""))

            if audio_file:
                records[audio_file] = {
                    field: clean_text(row.get(field, ""))
                    for field in METADATA_FIELDS
                }

    return records


def write_metadata(records):
    save_csv(
        METADATA_FILE,
        list(records.values()),
        METADATA_FIELDS
    )


# ============================================================
# 7. GENERATE CLEAN SPEECH
# ============================================================

def main():
    references = load_references()
    targets = load_targets()

    speaker_mapping = load_speaker_mapping(
        targets, references
    )

    metadata = load_existing_metadata()
    failures = {}

    print("\nLoading Qwen3-TTS on CPU...", flush=True)
    start = time.time()

    model = Qwen3TTSModel.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        dtype=torch.float32
    )

    print(
        f"Model loaded in {time.time() - start:.1f} seconds.",
        flush=True
    )

    total = len(targets)

    for position, (_, row) in enumerate(
        targets.iterrows(), start=1
    ):
        target_id = row["speaker_id"]
        sentence = row["transcript"]

        reference_id = speaker_mapping[target_id]
        reference = references[reference_id]

        # Source row ensures each target CSV record gets
        # its own output filename, even for duplicate sentences.
        source_row = int(row["_source_row"])
        filename = (
            f"target_{safe_filename(target_id)}"
            f"_row_{source_row:06d}.wav"
        )

        output_path = OUTPUT_DIR / filename
        relative_path = output_path.relative_to(ROOT).as_posix()

        print(
            f"\n[{position}/{total}] "
            f"Target speaker: {target_id} | "
            f"Reference: {reference_id}",
            flush=True
        )
        print(f"Text: {sentence}", flush=True)

        # Resume when an existing WAV is valid.
        if output_path.is_file():
            try:
                info = sf.info(output_path)

                metadata[relative_path] = {
                    "audio_file": relative_path,
                    "target_speaker_id": target_id,
                    "reference_contributor_id": reference_id,
                    "text": sentence,
                    "sample_rate": info.samplerate,
                    "duration_seconds": round(info.duration, 3),
                    "status": "success",
                    "error": ""
                }

                failures.pop(relative_path, None)
                write_metadata(metadata)

                print(f"Already exists; skipped: {filename}")
                continue

            except Exception:
                output_path.unlink(missing_ok=True)

        start = time.time()

        try:
            wavs, sr = model.generate_voice_clone(
                text=sentence,
                language="English",
                ref_audio=str(reference["audio"]),
                ref_text=reference["text"],
                max_new_tokens=256
            )

            audio = np.asarray(wavs[0]).squeeze()

            if audio.size == 0:
                raise ValueError("Generated audio is empty.")

            if not np.isfinite(audio).all():
                raise ValueError(
                    "Generated audio contains invalid numbers."
                )

            sf.write(output_path, audio, sr)

            # Verify that the WAV can be read.
            info = sf.info(output_path)

            metadata[relative_path] = {
                "audio_file": relative_path,
                "target_speaker_id": target_id,
                "reference_contributor_id": reference_id,
                "text": sentence,
                "sample_rate": info.samplerate,
                "duration_seconds": round(info.duration, 3),
                "status": "success",
                "error": ""
            }

            failures.pop(relative_path, None)

            print(
                f"Saved: {filename} | "
                f"Duration: {info.duration:.1f}s | "
                f"Time: {time.time() - start:.1f}s",
                flush=True
            )

        except Exception as e:
            output_path.unlink(missing_ok=True)

            error_message = str(e).replace("\n", " ")[:500]

            metadata[relative_path] = {
                "audio_file": relative_path,
                "target_speaker_id": target_id,
                "reference_contributor_id": reference_id,
                "text": sentence,
                "sample_rate": "",
                "duration_seconds": "",
                "status": "failed",
                "error": error_message
            }

            failures[relative_path] = {
                "audio_file": relative_path,
                "target_speaker_id": target_id,
                "text": sentence,
                "status": "failed",
                "error": error_message
            }

            print(
                f"FAILED: {filename} | {error_message}",
                flush=True
            )

        # Save after every item so progress survives interruption.
        write_metadata(metadata)

        save_csv(
            MISSING_FILE,
            list(failures.values()),
            MISSING_FIELDS
        )

    # ========================================================
    # 8. FINAL AUDIT
    # ========================================================

    # Recheck all expected output files.
    final_missing = []

    for _, row in targets.iterrows():
        target_id = row["speaker_id"]
        sentence = row["transcript"]
        source_row = int(row["_source_row"])

        filename = (
            f"target_{safe_filename(target_id)}"
            f"_row_{source_row:06d}.wav"
        )

        output_path = OUTPUT_DIR / filename
        relative_path = output_path.relative_to(ROOT).as_posix()

        status = "missing"
        error = "Output file does not exist."

        if output_path.is_file():
            try:
                sf.info(output_path)
                status = "failed" if (
                    relative_path in failures
                ) else "success"
                error = (
                    failures[relative_path]["error"]
                    if relative_path in failures
                    else ""
                )
            except Exception as e:
                status = "invalid_wav"
                error = str(e)[:500]

        if status != "success":
            final_missing.append({
                "audio_file": relative_path,
                "target_speaker_id": target_id,
                "text": sentence,
                "status": status,
                "error": error
            })

    save_csv(
        MISSING_FILE,
        final_missing,
        MISSING_FIELDS
    )

    print("\n" + "=" * 55)
    print("DATASET GENERATION FINISHED")
    print("=" * 55)
    print(f"Expected records: {total}")
    print(f"Successful records: {total - len(final_missing)}")
    print(f"Missing/failed/invalid: {len(final_missing)}")
    print(f"Audio directory: {OUTPUT_DIR}")
    print(f"Metadata: {METADATA_FILE}")
    print(f"Speaker mapping: {MAPPING_FILE}")
    print(f"Missing report: {MISSING_FILE}")


if __name__ == "__main__":
    main()
