import librosa
import soundfile as sf

audio, sr = librosa.load(
    "references/Recording.m4a",
    sr=24000,
    mono=True
)

sf.write("references/speaker_01.wav", audio, sr)

print("Reference audio converted successfully!")