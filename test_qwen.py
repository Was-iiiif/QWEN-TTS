
import time
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

torch.set_num_threads(8)

print("Loading Qwen3-TTS on CPU...", flush=True)
start = time.time()

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cpu",
    dtype=torch.float32
)

print(f"Model loaded in {time.time() - start:.1f} seconds.", flush=True)
print("Generating short test audio...", flush=True)

reference_text = (
    "This is a short reference recording "
    "for a voice cloning experiment."
)

generation_start = time.time()

wavs, sr = model.generate_voice_clone(
    text="Hello! This is a test.",
    language="English",
    ref_audio="references/speaker_01.wav",
    ref_text=reference_text,
    max_new_tokens=256
)

print(
    f"Generation finished in {time.time() - generation_start:.1f} seconds.",
    flush=True
)

sf.write("generated/test_output.wav", wavs[0], sr)

print("Saved generated/test_output.wav", flush=True)
print("Sample rate:", sr, flush=True)
