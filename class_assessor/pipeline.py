"""End-to-end pipeline for the class assessor.

Wires together media extraction, audio/visual analysis, and the LLM
assessor into a single :func:`run` and streaming :func:`run_stream` entry point.
"""

import tempfile
from pathlib import Path
from typing import Generator, Optional, Tuple

import librosa
from faster_whisper import WhisperModel

from class_assessor import assessor, audio_analysis, config, media, visual_analysis
from class_assessor.schemas import Assessment, FeatureBundle, Segment


def run_stream(
    video_path: Optional[Path] = None,
    vtt_path: Optional[Path] = None,
    fast_mode: bool = False,
) -> Generator[Tuple[str, Optional[Assessment]], None, None]:
    """Run the pipeline as a streaming generator yielding live progress updates."""
    # Fast-path: Pre-labeled WebVTT transcript provided
    if vtt_path and Path(vtt_path).exists():
        from class_assessor.vtt_parser import parse_vtt

        transcript, duration = parse_vtt(Path(vtt_path))
        mins = int(duration // 60)
        secs = int(duration % 60)
        yield (
            f"### Ingested .VTT Transcript ({len(transcript)} dialogue turns, {mins}m {secs}s)\n\n"
            f"> Official Zoom transcript detected with pre-labeled speakers. Bypassing speech-to-text models.",
            None,
        )
        print(f"[Pipeline VTT] Parsed {len(transcript)} dialogue turns across {duration:.1f}s.")

        with tempfile.TemporaryDirectory() as work:
            work_dir = Path(work)
            frames: list[tuple[float, Path]] = []
            if video_path and Path(video_path).exists():
                yield ("### Sampling video keyframes from recording...", None)
                interval = max(config.FRAME_INTERVAL_SEC, duration / config.MAX_FRAMES)
                frames = media.sample_frames(
                    video_path, work_dir / "frames", interval, config.MAX_FRAMES
                )

            yield (
                "### Evaluating pedagogical signals with DeepSeek LLM\n\n"
                "*Analyzing engagement, clarity, pacing, rapport, homework assignment, and technical quality against 8x8 rubric.*",
                None,
            )
            print("[Pipeline] Scoring feature bundle against 8x8 rubric with DeepSeek...")
            bundle = FeatureBundle(
                transcript=transcript,
                audio=audio_analysis.compute_metrics(transcript, Path("dummy.wav"), duration),
                visual=visual_analysis.compute_metrics(frames),
                frame_notes=visual_analysis.describe_frames(frames),
            )

            result = assessor.assess(bundle)
            print(f"Assessment complete: {result.label} (Score: {result.overall_score:.2f})")
            yield ("Complete", result)
            return

    if video_path is None or not Path(video_path).exists():
        raise RuntimeError("No video file or .VTT transcript provided for assessment.")

    with tempfile.TemporaryDirectory() as work:
        work_dir = Path(work)

        yield ("### [1/4] Inspecting media duration and extracting audio track...", None)
        print("[Pipeline 1/4] Probing duration and extracting audio track...")
        duration = media.probe_duration(video_path)
        wav_path = media.extract_audio(video_path, work_dir)

        mins = int(duration // 60)
        secs = int(duration % 60)
        yield (
            f"### [2/4] Sampling 36 video keyframes across {mins}m {secs}s session...",
            None,
        )
        print(f"[Pipeline 2/4] Sampling frames for {duration:.1f}s recording...")
        interval = max(config.FRAME_INTERVAL_SEC, duration / config.MAX_FRAMES)
        frames = media.sample_frames(
            video_path, work_dir / "frames", interval, config.MAX_FRAMES
        )

        yield (
            f"### [3/4] Starting local faster-whisper speech transcription on CPU (0%)\n\n"
            f"*Loading audio samples for {mins}m {secs}s recording...*",
            None,
        )
        print("[Pipeline 3/4] Transcribing audio locally on CPU with live progress...")
        audio, _sr = librosa.load(str(wav_path), sr=16000)
        model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
        segments_iter, _info = model.transcribe(audio, vad_filter=True)

        transcript: list[Segment] = []
        last_yield_sec = 0.0
        for s in segments_iter:
            seg = Segment(start=s.start, end=s.end, text=s.text.strip())
            transcript.append(seg)
            print(f"  [{seg.start:6.1f}s -> {seg.end:6.1f}s] {seg.text[:80]}")

            # Yield live UI update every ~2 seconds of transcript progress
            if seg.end - last_yield_sec >= 2.0:
                last_yield_sec = seg.end
                pct = int(min(1.0, seg.end / duration) * 100) if duration > 0 else 0
                cur_m = int(seg.end // 60)
                cur_s = int(seg.end % 60)
                yield (
                    f"### [3/4] Transcribing speech locally on CPU ({pct}% complete)\n\n"
                    f"**Timeline:** `{cur_m:02d}m {cur_s:02d}s` / `{mins:02d}m {secs:02d}s`\n\n"
                    f"> *\"{seg.text}\"*",
                    None,
                )

        if config.HF_TOKEN and not fast_mode:
            yield (
                "### [4/5] Neural Speaker Diarization (Pyannote Audio)\n\n"
                "*Separating teacher vs student speech turns on CPU (~30–60s)...*",
                None,
            )
            print("[Pipeline 4/5] Running neural speaker diarization on CPU...")
            transcript = audio_analysis.assign_speakers(wav_path, transcript)
            step_label = "5/5"
        else:
            step_label = "4/4"
            if fast_mode and config.HF_TOKEN:
                print("[Pipeline 4/4] Fast mode active: utilizing rapid speech turn metrics...")

        yield (
            f"### [{step_label}] Evaluating pedagogical signals with DeepSeek LLM\n\n"
            "*Analyzing engagement, clarity, pacing, rapport, homework assignment, and technical quality against 8x8 rubric.*",
            None,
        )
        print(f"[Pipeline {step_label}] Extracting metrics and scoring against rubric with DeepSeek...")
        bundle = FeatureBundle(
            transcript=transcript,
            audio=audio_analysis.compute_metrics(transcript, wav_path, duration),
            visual=visual_analysis.compute_metrics(frames),
            frame_notes=visual_analysis.describe_frames(frames),
        )

        result = assessor.assess(bundle)
        print(f"Assessment complete: {result.label} (Score: {result.overall_score:.2f})")
        yield ("Complete", result)


def run(
    video_path: Optional[Path] = None,
    vtt_path: Optional[Path] = None,
    progress_callback=None,
    fast_mode: bool = False,
) -> Assessment:
    """Run the full assessment pipeline synchronously."""
    final_result: Optional[Assessment] = None
    for msg, res in run_stream(video_path=video_path, vtt_path=vtt_path, fast_mode=fast_mode):
        if res is not None:
            final_result = res
    if final_result is None:
        raise RuntimeError("Pipeline failed to produce an assessment result.")
    return final_result
