"""Visual analysis for the class assessor.

Computes frame-to-frame change scores with OpenCV, optionally describes
sampled frames via an LLM vision call (only when cloud sending is enabled),
and derives aggregate visual metrics.
"""

import base64
from pathlib import Path

import cv2
import numpy as np

from class_assessor import config
from class_assessor.schemas import FrameNote, VisualMetrics


def change_scores(frames: list[tuple[float, Path]]) -> list[tuple[float, float]]:
    """Return ``(timestamp, mean_change)`` for consecutive frame pairs.

    Each frame is converted to grayscale and compared to the previous frame
    using the mean absolute pixel difference.
    """
    scores: list[tuple[float, float]] = []
    prev_gray = None
    for timestamp, path in frames:
        if not Path(path).exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            score = float(np.mean(cv2.absdiff(gray, prev_gray)))
            scores.append((timestamp, score))
        prev_gray = gray
    return scores


def describe_frames(frames: list[tuple[float, Path]]) -> list[FrameNote]:
    """Describe sampled frames using an LLM vision call.

    Returns an empty list when ``SEND_FRAMES_TO_CLOUD`` is disabled. Frames
    are batched (up to 6 per call) and encoded as base64 JPEGs. The prompt
    explicitly forbids identifying individuals.
    """
    if not config.SEND_FRAMES_TO_CLOUD:
        return []

    from openai import OpenAI

    client = OpenAI(api_key=config.LLM_API_KEY, base_url="https://openrouter.ai/api/v1")

    notes: list[FrameNote] = []
    batch_size = 6
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "Describe the classroom activity in these frames. "
                    "Do NOT identify any individuals. Focus on engagement, "
                    "board use, and general activity. Reply with one short "
                    "sentence per frame, prefixed by its timestamp."
                ),
            }
        ]
        for timestamp, path in batch:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(
                {"type": "text", "text": f"Timestamp: {timestamp}s"}
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )

        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": content}],
        )
        text = response.choices[0].message.content or ""
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            for timestamp, _path in batch:
                if line.startswith(str(timestamp)):
                    notes.append(FrameNote(timestamp=timestamp, note=line))
                    break
    return notes


def compute_metrics(frames: list[tuple[float, Path]]) -> VisualMetrics:
    """Compute aggregate :class:`VisualMetrics` from sampled frames."""
    scores = change_scores(frames)
    values = [v for _ts, v in scores]
    mean_change = float(np.mean(values)) if values else 0.0
    static_stretches = [ts for ts, v in scores if v < 1.0]
    return VisualMetrics(
        frames_sampled=len(frames),
        mean_change=mean_change,
        static_stretches=static_stretches,
    )
