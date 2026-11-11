import asyncio
import base64
import os
import sys
import math
import struct
import wave

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from google import genai
from app.config import settings

ASSETS_DIR = Path("app/assets/acknowledgements")

CLIPS = {
    "en": [
        ("one_moment.wav", "One moment.", "Kore"),
        ("let_me_check.wav", "Let me check that.", "Kore"),
        ("sure_give_me_a_second.wav", "Sure, give me a second.", "Kore"),
    ],
    "hi": [
        ("ek_second.wav", "Ek second, main check karta hoon.", "Puck"),
        ("zara_dekhta_hoon.wav", "Zara dekhta hoon.", "Puck"),
        ("rukiye_check_karta_hoon.wav", "Rukiye, main abhi dekhta hoon.", "Puck"),
    ],
    "ar": [
        ("lahza_wahida.wav", "لحظة واحدة، خلني أتأكد لك.", "Leda"),
        ("sawani_ashoof.wav", "ثواني وأشوف لك.", "Leda"),
        ("abshir_daqiqa.wav", "أبشر، ثواني وأرد عليك.", "Leda"),
    ]
}

def generate_earcon_wav(filename: Path, freq1=523.25, freq2=659.25, sample_rate=24000):
    """Generates a pleasant 2-tone harmonic soft chime (C5 -> E5) with exponential decay."""
    duration = 0.45  # seconds
    num_samples = int(sample_rate * duration)
    half_samples = num_samples // 2

    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        decay = math.exp(-6.0 * (t % (duration / 2)))
        freq = freq1 if i < half_samples else freq2
        # Fundamental + gentle 2nd harmonic
        val = (math.sin(2 * math.pi * freq * t) * 0.7 + math.sin(4 * math.pi * freq * t) * 0.3) * decay
        val = max(-1.0, min(1.0, val))
        samples.append(int(val * 16000))  # 16-bit PCM, moderate volume

    raw_bytes = struct.pack(f"<{len(samples)}h", *samples)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)
    print(f"Generated neutral audio cue: {filename}")

def generate_soft_chime_wav(filename: Path, freq1=587.33, freq2=880.00, sample_rate=24000):
    """Generates a pleasant gentle high bell chime (D5 -> A5)."""
    duration = 0.40
    num_samples = int(sample_rate * duration)
    half_samples = num_samples // 2

    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        decay = math.exp(-7.0 * (t % (duration / 2)))
        freq = freq1 if i < half_samples else freq2
        val = (math.sin(2 * math.pi * freq * t) * 0.8 + math.sin(6 * math.pi * freq * t) * 0.2) * decay
        val = max(-1.0, min(1.0, val))
        samples.append(int(val * 14000))

    raw_bytes = struct.pack(f"<{len(samples)}h", *samples)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)
    print(f"Generated neutral soft chime: {filename}")

async def generate_speech_clips():
    client = genai.Client(api_key=settings.GEMINI_API_KEY, http_options={"api_version": "v1beta"})
    for lang, clips in CLIPS.items():
        lang_dir = ASSETS_DIR / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        for fname, text, voice in clips:
            out_path = lang_dir / fname
            if out_path.exists():
                print(f"File already exists, skipping: {out_path}")
                continue
            print(f"Generating [{lang}] {fname} ('{text}') with voice {voice}...")
            try:
                res = await client.aio.interactions.create(
                    model=settings.GEMINI_TTS_MODEL,
                    input=[{
                        "type": "user_input",
                        "content": [{
                            "type": "text",
                            "text": text,
                            "annotations": [{"type": "speech_metadata", "style": "natural and conversational"}]
                        }]
                    }],
                    response_format={"type": "audio"},
                    generation_config={"speech_config": [{"voice": voice}]}
                )
                audio_b64 = getattr(res.output_audio, "data", None)
                if audio_b64:
                    wav_data = base64.b64decode(audio_b64)
                    with open(out_path, "wb") as f:
                        f.write(wav_data)
                    print(f"Saved: {out_path} ({len(wav_data)} bytes)")
                else:
                    print(f"Failed to get audio data for {fname}")
            except Exception as e:
                print(f"Error generating {fname}: {e}")

async def main():
    print("Starting generation of pre-recorded acknowledgement assets...")
    neutral_dir = ASSETS_DIR / "neutral"
    generate_earcon_wav(neutral_dir / "earcon.wav")
    generate_soft_chime_wav(neutral_dir / "soft_chime.wav")
    await generate_speech_clips()
    print("Finished generating acknowledgement assets!")

if __name__ == "__main__":
    asyncio.run(main())
