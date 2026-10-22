"""Audio analysis for the class assessor.

Transcribes audio with faster-whisper, optionally assigns speakers via
pyannote diarization, and derives aggregate audio metrics (speech ratio,
words per minute, pauses, questions, energy variation).
"""

from pathlib import Path

import librosa
import numpy as np
from faster_whisper import WhisperModel

from class_assessor import config
from class_assessor.schemas import AudioMetrics, Segment


def transcribe(
    wav_path: Path, duration: float = 0.0, progress_callback=None
) -> list[Segment]:
    """Transcribe ``wav_path`` into a list of :class:`Segment` objects with live streaming feedback."""
    print("[Whisper] Loading audio track into memory...")
    audio, _sr = librosa.load(str(wav_path), sr=16000)
    total_sec = duration if duration > 0 else (len(audio) / 16000.0)

    print(f"[Whisper] Initializing WhisperModel('{config.WHISPER_MODEL}', device='cpu', compute_type='int8')...")
    model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")

    print(f"[Whisper] Starting transcription ({total_sec:.1f}s total)...")
    segments_iter, _info = model.transcribe(audio, vad_filter=True)

    segments: list[Segment] = []
    for s in segments_iter:
        seg = Segment(start=s.start, end=s.end, text=s.text.strip())
        segments.append(seg)

        # Real-time console log
        print(f"  [{seg.start:6.1f}s -> {seg.end:6.1f}s] {seg.text[:80]}")

        # Real-time UI progress update
        if progress_callback and total_sec > 0:
            fraction = min(1.0, seg.end / total_sec)
            pct = 0.35 + 0.40 * fraction
            mins_done = int(seg.end // 60)
            mins_total = int(total_sec // 60)
            progress_callback(
                pct,
                desc=f"Transcribing: {mins_done}m / {mins_total}m ({int(fraction * 100)}%)...",
            )

    print(f"[Whisper] Completed transcribing {len(segments)} spoken segments.")
    return segments


def assign_speakers(wav_path: Path, segments: list[Segment]) -> list[Segment]:
    """Assign speaker labels to ``segments`` using pyannote diarization.

    Returns ``segments`` unchanged when no Hugging Face token is configured
    or if the model agreement has not been accepted yet.
    """
    if not config.HF_TOKEN:
        return segments

    try:
        import soundfile as sf
        import torch
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=config.HF_TOKEN,
        )

        # Pre-decode waveform in-memory via soundfile to bypass Windows torchcodec/FFmpeg C-DLL issues
        wav, sr = sf.read(str(wav_path))
        wav_t = torch.tensor(wav, dtype=torch.float32)
        if wav_t.ndim == 1:
            wav_t = wav_t.unsqueeze(0)
        elif wav_t.ndim == 2:
            wav_t = wav_t.transpose(0, 1)

        output = pipeline({"waveform": wav_t, "sample_rate": sr})
        annotation = getattr(output, "speaker_diarization", output)

        turns = [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]

        for seg in segments:
            midpoint = (seg.start + seg.end) / 2.0
            for start, end, speaker in turns:
                if start <= midpoint <= end:
                    seg.speaker = speaker
                    break
        print(f"[Diarization] Successfully assigned speakers across {len(turns)} conversational turns.")
    except Exception as exc:
        print(f"Notice: Diarization skipped ({exc}); proceeding with transcription metrics.")

    return segments


def compute_metrics(
    segments: list[Segment], wav_path: Path, duration: float
) -> AudioMetrics:
    """Compute aggregate :class:`AudioMetrics` from transcript segments."""
    total_words = sum(len(seg.text.split()) for seg in segments)
    speech_time = sum(max(0.0, seg.end - seg.start) for seg in segments)

    long_pause_times: list[float] = []
    for prev, curr in zip(segments, segments[1:]):
        gap = curr.start - prev.end
        if gap > 5.0:
            long_pause_times.append(round(prev.end, 2))

    question_times = [
        round(seg.start, 2) for seg in segments if seg.text.strip().endswith("?")
    ]

    speakers = {seg.speaker for seg in segments if seg.speaker}
    teacher_talk_share = None
    speaker_count = None
    if speakers:
        speaker_count = len(speakers)
        speaker_times: dict[str, float] = {}
        for seg in segments:
            if seg.speaker:
                speaker_times[seg.speaker] = speaker_times.get(seg.speaker, 0.0) + max(
                    0.0, seg.end - seg.start
                )
        primary_speaker_time = max(speaker_times.values()) if speaker_times else 0.0
        teacher_talk_share = primary_speaker_time / speech_time if speech_time > 0 else 0.0

    energy_variation = 0.5
    if wav_path and Path(wav_path).exists():
        try:
            y, _sr = librosa.load(str(wav_path), sr=16000)
            rms = librosa.feature.rms(y=y)[0]
            energy_variation = float(np.std(rms) / (np.mean(rms) + 1e-9))
        except Exception:
            pass

    speech_ratio = speech_time / duration if duration > 0 else 0.0
    words_per_minute = (total_words / duration) * 60.0 if duration > 0 else 0.0

    return AudioMetrics(
        duration_sec=duration,
        speech_ratio=speech_ratio,
        words_per_minute=words_per_minute,
        long_pause_times=long_pause_times,
        question_times=question_times,
        teacher_talk_share=teacher_talk_share,
        speaker_count=speaker_count,
        energy_variation=energy_variation,
    )
