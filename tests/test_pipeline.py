"""Tests for the class assessor pipeline.

Covers schema validation, calibration mapping, audio pause metrics, and
end-to-end media processing on a synthetic ffmpeg-generated video.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from class_assessor import assessor, media, schemas
from class_assessor.schemas import DimensionScore, Evidence, LLMOutput, Segment


def test_evidence_validation_requires_two_items() -> None:
    """DimensionScore must reject fewer than 2 evidence items."""
    with pytest.raises(ValidationError):
        DimensionScore(
            name="engagement",
            score=4,
            evidence=[Evidence(timestamp=1.0, observation="Good eye contact here")],
        )


def test_evidence_validation_requires_min_length() -> None:
    """Evidence observations shorter than 10 chars must be rejected."""
    with pytest.raises(ValidationError):
        Evidence(timestamp=1.0, observation="short")


def _make_llm_output(score: int) -> LLMOutput:
    """Build a minimal LLMOutput with all dimensions at ``score``."""
    evidence = [
        Evidence(timestamp=1.0, observation="Clear explanation of the topic"),
        Evidence(timestamp=2.0, observation="Students responded positively"),
    ]
    dimensions = [
        DimensionScore(name=name, score=score, evidence=evidence)
        for name in (
            "engagement",
            "clarity",
            "pacing",
            "rapport",
            "content",
            "tech_management",
        )
    ]
    return LLMOutput(
        dimensions=dimensions,
        strengths=["Clear delivery"],
        weaknesses=["Could improve pacing"],
        summary="Overall solid session.",
    )


def test_calibration_mapping() -> None:
    """calibrate() maps weighted scores to the expected labels."""
    assert assessor.calibrate(_make_llm_output(1)).label == "Poor"
    assert assessor.calibrate(_make_llm_output(3)).label == "Average"
    assert assessor.calibrate(_make_llm_output(4)).label in ("Good", "Excellent")
    assert assessor.calibrate(_make_llm_output(5)).label == "Excellent"


def test_audio_pause_metrics() -> None:
    """Gaps longer than 5 seconds must appear in long_pause_times."""
    segments = [
        Segment(start=0.0, end=2.0, text="First sentence."),
        Segment(start=10.0, end=12.0, text="Second sentence."),
    ]
    ffmpeg_bin = media._get_ffmpeg()
    with tempfile.TemporaryDirectory() as work:
        wav = Path(work) / "audio.wav"
        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-t",
                "12",
                str(wav),
            ],
            capture_output=True,
            check=True,
        )
        metrics = schemas.AudioMetrics(
            duration_sec=12.0,
            speech_ratio=0.33,
            words_per_minute=20.0,
            long_pause_times=[2.0],
            question_times=[],
            energy_variation=0.1,
        )
        assert metrics.long_pause_times == [2.0]

    from class_assessor import audio_analysis

    with tempfile.TemporaryDirectory() as work:
        wav = Path(work) / "audio.wav"
        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-t",
                "12",
                str(wav),
            ],
            capture_output=True,
            check=True,
        )
        computed = audio_analysis.compute_metrics(segments, wav, 12.0)
        assert 2.0 in computed.long_pause_times


def test_synthetic_media_processing() -> None:
    """probe_duration, extract_audio, and sample_frames work on a test video."""
    ffmpeg_bin = media._get_ffmpeg()
    with tempfile.TemporaryDirectory() as work:
        work_dir = Path(work)
        video = work_dir / "test.mp4"
        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=2:size=320x240:rate=10",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-shortest",
                str(video),
            ],
            capture_output=True,
            check=True,
        )

        duration = media.probe_duration(video)
        assert 1.5 <= duration <= 2.5

        wav = media.extract_audio(video, work_dir / "audio")
        assert wav.exists()
        assert wav.name == "audio.wav"

        frames = media.sample_frames(video, work_dir / "frames", 0.5, 10)
        assert len(frames) > 0
        assert all(path.exists() for _ts, path in frames)


def test_vtt_parser() -> None:
    """Test WebVTT parser extracts cue timestamps, speaker names, and text."""
    from class_assessor.vtt_parser import parse_vtt

    sample_vtt = """WEBVTT

1
00:00:01.500 --> 00:00:04.200
Teacher John: Welcome to our chess class today!

2
00:00:05.100 --> 00:00:08.500
Student Alice: Hello Teacher, ready to learn opposition!
"""
    segments, duration = parse_vtt(sample_vtt)
    assert len(segments) == 2
    assert duration == 8.5
    assert segments[0].speaker == "Teacher John"
    assert segments[0].text == "Welcome to our chess class today!"
    assert segments[0].start == 1.5
    assert segments[0].end == 4.2
    assert segments[1].speaker == "Student Alice"


def test_visual_analysis_resilience() -> None:
    """visual_analysis handles missing files and synthetic frames without error."""
    from class_assessor import visual_analysis

    # 1. Missing paths
    scores = visual_analysis.change_scores([(1.0, Path("non_existent.jpg"))])
    assert scores == []
    metrics = visual_analysis.compute_metrics([(1.0, Path("non_existent.jpg"))])
    assert metrics.frames_sampled == 1
    assert metrics.mean_change == 0.0

    # 2. Empty list
    metrics_empty = visual_analysis.compute_metrics([])
    assert metrics_empty.frames_sampled == 0
    assert metrics_empty.mean_change == 0.0
