"""Media pipeline helpers for the class assessor.

Wraps ffmpeg / ffprobe for duration probing, mono 16 kHz audio extraction,
and uniform frame sampling. Automatically falls back to imageio-ffmpeg and OpenCV
when system binaries are not on PATH.
"""

import json
import shutil
import subprocess
from pathlib import Path


def _get_ffmpeg() -> str:
    """Return the path to the ffmpeg executable."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg not found on PATH or via imageio-ffmpeg") from exc


def _run(cmd: list[str]) -> str:
    """Run a subprocess command and return its stdout."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as exc:
        err_msg = exc.stderr.strip() if exc.stderr else f"Exit code {exc.returncode}"
        raise RuntimeError(f"FFmpeg execution failed: {err_msg}") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command not found: {cmd[0]}") from exc


def probe_duration(video: Path) -> float:
    """Return the duration of ``video`` in seconds."""
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        cmd = [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ]
        try:
            output = _run(cmd)
            return float(json.loads(output)["format"]["duration"])
        except Exception:
            pass

    # OpenCV fallback for duration probing
    try:
        import cv2
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if frames > 0 and fps > 0:
            return float(frames / fps)
    except Exception:
        pass

    raise RuntimeError(f"Could not determine video duration for: {video}")


def extract_audio(video: Path, out_dir: Path) -> Path:
    """Extract mono 16 kHz WAV audio from ``video`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "audio.wav"
    ffmpeg_bin = _get_ffmpeg()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav),
    ]
    _run(cmd)
    return wav


def sample_frames(
    video: Path, out_dir: Path, interval: float, max_frames: int
) -> list[tuple[float, Path]]:
    """Sample JPEG frames from ``video`` every ``interval`` seconds.

    Returns a sorted list of ``(timestamp, frame_path)`` tuples.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = _get_ffmpeg()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps=1/{interval},scale=640:-1",
        "-frames:v",
        str(max_frames),
        str(out_dir / "frame_%04d.jpg"),
    ]
    _run(cmd)

    frames = sorted(out_dir.glob("frame_*.jpg"))
    return [(round(i * interval, 1), p) for i, p in enumerate(frames)]
