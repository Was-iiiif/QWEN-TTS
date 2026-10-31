# Synthetic Speech Generation for ASR Dataset Completion

A reference-conditioned synthetic speech generation pipeline for supplementing missing audio records in an Automatic Speech Recognition (ASR) dataset using **Qwen3-TTS**, GPU inference, persistent speaker mapping, prompt caching, and CSV-based metadata auditing.

> **Important:** This project generates new synthetic audio from existing transcripts. It does not recover the original missing recordings or verify the real identity of target speakers. Synthetic records should be clearly labeled and evaluated before being used in ASR experiments.

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [How It Works](#how-it-works)
- [Pipeline Architecture](#pipeline-architecture)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Input Data](#input-data)
- [Installation](#installation)
- [Configuration](#configuration)
- [Run the Pipeline](#run-the-pipeline)
- [Output Files](#output-files)
- [Quality Checks](#quality-checks)
- [Limitations](#limitations)
- [Research and Data Integrity](#research-and-data-integrity)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

## Overview

Speech datasets often contain transcripts and speaker metadata for records whose original audio files are missing or inaccessible. This project provides a pipeline for generating synthetic speech for eligible records rather than simply discarding them.

The pipeline uses the target transcript as the text to synthesize and a separate, available reference recording plus its transcript to condition the generated voice. It assigns each target speaker ID to an available reference contributor ID and records that assignment for traceability.

### Key Features

- **Qwen3-TTS synthesis:** Generates speech from target transcripts using reference-conditioned voice cloning.
- **GPU inference:** Uses CUDA with FP32 precision and eager attention, a configuration tested in the project environment.
- **Persistent speaker mapping:** Reuses saved target-to-reference assignments across runs.
- **Prompt caching:** Prepares a voice-cloning prompt once per reference voice and reuses it for multiple target records.
- **Resumable generation:** Skips existing readable WAV files to avoid unnecessary regeneration.
- **Metadata logging:** Records transcript, target ID, reference ID, sample rate, duration, and generation status.
- **Failure reporting:** Tracks missing or failed outputs in a separate CSV.
- **Final audit:** Checks output files for basic readability and nonzero audio frames.

## Objectives

1. Generate synthetic audio for eligible records with missing audio.
2. Retain the target transcript and record its relationship to the generated WAV.
3. Keep target speaker IDs distinct from reference contributor IDs.
4. Make reference assignment reproducible and inspectable.
5. Reduce repeated prompt preparation through caching.
6. Support interrupted runs and basic output auditing.
7. Provide a foundation for evaluating synthetic-data augmentation for ASR.

## How It Works

1. **Load reference metadata.** Read available reference filenames, contributor IDs, and transcripts.
2. **Validate reference audio.** Locate the corresponding WAV files and exclude unusable references.
3. **Load target records.** Read target speaker IDs and transcripts for records requiring audio.
4. **Build speaker mapping.** Reuse valid saved assignments and assign references to unmapped target speakers.
5. **Load Qwen3-TTS.** Initialize the model on the CUDA device.
6. **Prepare voice prompts.** Create and cache a prompt for each reference voice as needed.
7. **Generate speech.** Synthesize each target transcript with its assigned reference prompt.
8. **Validate and save audio.** Check the waveform and write a WAV file through a temporary output path.
9. **Record metadata.** Save success or failure information after each record.
10. **Audit outputs.** Check the resulting WAV files and update the missing-output report.

## Pipeline Architecture

```text
Reference Metadata CSV + Reference WAVs
                    |
                    v
           Reference Validation
                    |
                    v
            Usable Reference Pool
                    |
                    +----------------------+
                                           |
Target Metadata CSV                        |
        |                                  |
        v                                  |
    Target Validation                      |
        |                                  |
        v                                  |
 Persistent Speaker Mapping <---------------+
        |
        v
   Qwen3-TTS on CUDA
        |
        v
 Voice Prompt Creation / Cache
        |
        v
   Target Text Synthesis
        |
        v
  WAV Validation and Saving
        |
        +----------------------+
        |                      |
        v                      v
  metadata.csv       missing_audio_report.csv
        |
        v
    Final Audit
```

## Requirements

### Software

- Python environment compatible with the installed `qwen-tts` and PyTorch versions
- PyTorch with CUDA support
- `qwen-tts`
- `pandas`
- `numpy`
- `soundfile`

The tested project environment used these package versions:

| Package / Component | Version or setting |
|---|---|
| `qwen-tts` | `0.1.1` |
| `transformers` | `4.57.3` |
| PyTorch | `2.14.1+cu130` in the tested environment |
| Model | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| Device | `cuda:0` |
| Data type | FP32 |
| Attention | `eager` |

Package compatibility can change. Use the versions compatible with your CUDA installation and the Qwen3-TTS package; the table records the project-tested configuration, not a universal compatibility guarantee.

### Hardware

- NVIDIA GPU with sufficient VRAM for the selected model and inference settings
- CUDA-compatible PyTorch installation
- Storage for reference recordings and generated WAV files

The model was tested on an NVIDIA RTX 4070 in the project environment. Other GPUs may require different settings.

## Project Structure

Expected layout:

```text
asr-synthetic-data/
├── references/
│   ├── speaker_01.wav
│   └── ...
├── dev_metadata_sampled_500.csv
├── missing_audio_metadata.csv
├── generate_dataset.py
├── generated/
│   └── clean/
├── metadata.csv                 # Created/updated by the pipeline
├── speaker_mapping.csv          # Created/updated by the pipeline
└── missing_audio_report.csv     # Created/updated by the pipeline
```

The exact filenames can be changed in the configuration section of `generate_dataset.py`.

## Input Data

### 1. Reference metadata

File: `dev_metadata_sampled_500.csv`

Required columns:

| Column | Description |
|---|---|
| `filename` | Reference audio filename or path |
| `contributor_id` | Identifier for the reference contributor |
| `transcript` | Text spoken in the reference recording |

The transcript must accurately match the speech in the reference WAV. Reference paths should resolve to existing audio files.

### 2. Target metadata

File: `missing_audio_metadata.csv`

Required columns:

| Column | Description |
|---|---|
| `speaker_id` | Target record's speaker identifier |
| `transcript` | Text to synthesize |

Each eligible target record should have a nonempty speaker ID and transcript.

### Speaker mapping is not identity recovery

A target `speaker_id` does not need to match a reference `contributor_id`. The pipeline assigns an available reference voice when there is no matching ID. This enables synthesis, but it does **not** prove that the target and reference IDs belong to the same person.

## Installation

Create and activate a virtual environment. For example, on Windows PowerShell:

```powershell
cd F:\SpeechTech\asr-synthetic-data
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies using versions compatible with your environment:

```powershell
python -m pip install --upgrade pip
pip install pandas numpy soundfile
pip install qwen-tts==0.1.1
```

Install the appropriate CUDA-enabled PyTorch build by following the official PyTorch installation instructions for your system. Then verify CUDA availability:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
```

The script also imports `transformers` through the Qwen3-TTS environment. If you need to reproduce the tested configuration, use the project-compatible version rather than upgrading dependencies indiscriminately.

## Configuration

Review the configuration near the top of `generate_dataset.py`:

```python
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

REFERENCE_FILE = ROOT / "dev_metadata_sampled_500.csv"
TARGET_FILE = ROOT / "missing_audio_metadata.csv"

REFERENCE_DIR = ROOT / "references"
OUTPUT_DIR = ROOT / "generated" / "clean"

METADATA_FILE = ROOT / "metadata.csv"
MISSING_FILE = ROOT / "missing_audio_report.csv"
MAPPING_FILE = ROOT / "speaker_mapping.csv"

DEVICE = "cuda:0"
DTYPE = torch.float32
ATTENTION = "eager"

LANGUAGE = "English"
MAX_NEW_TOKENS = 256

# Test a small batch first; use None to process all eligible rows.
MAX_RECORDS = 10
```

### Important settings

- **`LANGUAGE`:** Must match the target text and a language supported by the model. The current configuration is for English. Do not assume it will work for Bengali or another unsupported language.
- **`MAX_RECORDS`:** Start with `10` to test the pipeline. Set it to `None` after validating the output to process all eligible records.
- **`DTYPE` and `ATTENTION`:** FP32 and eager attention reflect the tested configuration. Do not change them without testing stability and memory use.
- **`SEED`:** Controls reproducible assignment of new speaker mappings. Existing valid mappings are retained from `speaker_mapping.csv`.

## Run the Pipeline

From the project directory, activate the virtual environment and run:

```powershell
python generate_dataset.py
```

For the first run:

1. Keep `MAX_RECORDS = 10`.
2. Confirm that the model loads on the GPU.
3. Listen to several generated WAV files.
4. Inspect `metadata.csv` and `speaker_mapping.csv`.
5. Review `missing_audio_report.csv` for failures.
6. If the test output is acceptable, set `MAX_RECORDS = None` and run again.

The script checks for existing readable outputs and skips them, allowing a run to resume after interruption. If you change transcripts, reference files, model settings, or mapping logic, consider using a new output directory or carefully reviewing existing metadata to avoid mixing outputs generated under different configurations.

## Output Files

### `generated/clean/`

Contains generated WAV files. Filenames include the target speaker ID and original target CSV row number to reduce naming collisions.

Example:

```text
target_0033c51f-bd0f-4fcf-c71b-08dd93dc955e_row_000002.wav
```

### `metadata.csv`

Records information about generated audio.

| Field | Description |
|---|---|
| `audio_file` | Relative path to the generated WAV |
| `target_speaker_id` | Target speaker ID |
| `reference_contributor_id` | Assigned reference contributor ID |
| `text` | Target transcript used for synthesis |
| `sample_rate` | Sample rate of the saved audio |
| `duration_seconds` | Duration of the saved audio |
| `status` | Generation status |
| `error` | Error message, if any |

### `speaker_mapping.csv`

Records the assignment of each target speaker ID to a reference contributor ID, including the reference audio path and transcript. This makes the voice assignment inspectable and reusable.

### `missing_audio_report.csv`

Records output files that remain missing, invalid, or failed, together with the target ID, transcript, status, and error message.

## Quality Checks

The current pipeline performs basic technical checks:

- Required CSV columns exist.
- Reference audio paths can be resolved.
- Generated waveforms are nonempty and contain finite values.
- Saved WAV files can be read and contain audio frames.
- Generation errors are logged.
- The final output directory is audited.

These checks **do not** establish that the generated speech is natural, correctly pronounced, or semantically faithful to the target transcript.

Recommended additional checks:

1. **Manual listening:** Listen to a representative sample of generated audio.
2. **Independent ASR check:** Transcribe generated audio using a separate ASR system and compare the result with the target transcript.
3. **WER:** Measure word-level transcription disagreement where appropriate.
4. **Acoustic checks:** Inspect duration, clipping, near-silence, sample-rate consistency, and artifacts.
5. **Speaker consistency:** If important, evaluate whether records assigned to the same reference voice sound consistent.
6. **Downstream ASR evaluation:** Compare a baseline trained on original training data with a model trained using synthetic augmentation, using the same held-out real-speech test set.

Do not claim that augmentation improves ASR performance until controlled experiments support that conclusion.

## Limitations

- Generated audio is synthetic and is not a reconstruction of the original missing recording.
- A target-to-reference mapping does not recover or verify the target speaker's identity.
- Limited reference diversity can cause multiple target speakers to share the same synthetic voice.
- Incorrect reference transcripts can affect voice conditioning.
- The configured language must be supported by the model.
- Synthetic speech may differ from real speech and introduce a domain mismatch for ASR training.
- File-level validation does not guarantee transcript accuracy or perceptual quality.
- Exact outputs may vary with software, hardware, and inference settings.
- The current script uses a persistent mapping file; mapping history can influence assignments.
- Full-scale generation can require substantial GPU time and storage.

## Research and Data Integrity

- Clearly label generated records as synthetic. Consider adding an `audio_origin` field with values such as `original` and `synthetic_tts`.
- Use reference recordings only when their dataset terms and consent arrangements permit the intended use.
- Keep original and synthetic recordings distinguishable.
- Avoid using development or test recordings as references for training augmentation when this would compromise the independence of evaluation.
- Do not treat synthetic replacements for missing test audio as equivalent to authentic test recordings.
- Document model version, configuration, reference selection, quality checks, and data splits in research outputs.

## Troubleshooting

### CUDA is unavailable

Check that the virtual environment uses a CUDA-enabled PyTorch build and that the NVIDIA driver is working:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### Reference WAV files are not found

- Confirm that `references/` exists.
- Check the `filename` values in `dev_metadata_sampled_500.csv`.
- Ensure paths are relative to the project or point to accessible files.
- Verify that each referenced file is a valid audio file.

### Generation fails for every record

- Verify that the model downloads and loads successfully.
- Confirm the target language is supported and matches `LANGUAGE`.
- Check that reference transcripts match their audio.
- Inspect the console traceback and `missing_audio_report.csv`.
- Test with a short target sentence and one known-good reference WAV.

### Existing outputs are skipped

The script is designed to skip existing readable WAV files. If you need to regenerate them after changing model settings or source text, back up the results and metadata, then use a clean output directory or remove only the outputs you intend to regenerate.

### Audio is generated but sounds incorrect

File validation does not guarantee correct pronunciation or transcript fidelity. Review the reference transcript, language configuration, target text, and model output. Use manual listening and independent ASR checks before accepting the audio.

## Future Improvements

- Add explicit provenance and model-version fields to the metadata.
- Add automated transcript fidelity checks with an independent ASR model.
- Add acoustic-quality checks and configurable rejection thresholds.
- Improve mapping and audit logic to detect stale outputs when source text or generation settings change.
- Track GPU memory, total runtime, per-record latency, and generation statistics.
- Evaluate different synthetic-data proportions in controlled ASR experiments.
- Add optional noise augmentation as a separate, documented processing stage.

## Acknowledgements

This project uses the Qwen3-TTS model and the PyTorch ecosystem. Please consult the relevant official model and software documentation for licensing, usage terms, and hardware setup.

## Disclaimer

This repository describes a synthetic data-generation workflow for research and experimentation. Users are responsible for verifying data permissions, accurately labeling synthetic recordings, maintaining evaluation integrity, and assessing the quality and suitability of generated audio before use.
