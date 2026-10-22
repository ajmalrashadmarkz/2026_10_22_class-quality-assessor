"""Parser for WebVTT (.vtt) caption and transcript files.

Extracts cue timestamps, dialogue content, and Zoom speaker identities into
the domain :class:`Segment` model without running speech-to-text models.
"""

from pathlib import Path
import re
from typing import Tuple, Union

from class_assessor.schemas import Segment


def _parse_time(time_str: str) -> float:
    """Convert WebVTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) into seconds."""
    parts = time_str.strip().replace(",", ".").split(":")
    if len(parts) == 3:
        hours = float(parts[0])
        mins = float(parts[1])
        secs = float(parts[2])
        return hours * 3600.0 + mins * 60.0 + secs
    elif len(parts) == 2:
        mins = float(parts[0])
        secs = float(parts[1])
        return mins * 60.0 + secs
    return float(parts[0])


def parse_vtt(vtt_source: Union[Path, str]) -> Tuple[list[Segment], float]:
    """Parse WebVTT content and return ``(segments, total_duration_seconds)``."""
    if isinstance(vtt_source, Path) or (isinstance(vtt_source, str) and Path(vtt_source).exists()):
        text = Path(vtt_source).read_text(encoding="utf-8", errors="replace")
    else:
        text = str(vtt_source)

    segments: list[Segment] = []
    ts_regex = re.compile(
        r"(\d{1,2}:(?:\d{2}:)?\d{2}[.,]\d{3})\s*-->\s*(\d{1,2}:(?:\d{2}:)?\d{2}[.,]\d{3})"
    )

    blocks = re.split(r"\n\s*\n", text.strip())
    max_end = 0.0

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            match = ts_regex.search(line)
            if match:
                start_sec = _parse_time(match.group(1))
                end_sec = _parse_time(match.group(2))
                max_end = max(max_end, end_sec)

                spoken_text = " ".join(lines[idx + 1:]).strip()
                if not spoken_text:
                    continue

                speaker = None
                if ":" in spoken_text:
                    candidate, content = spoken_text.split(":", 1)
                    if len(candidate) <= 40 and not any(p in candidate for p in ".!?"):
                        speaker = candidate.strip()
                        spoken_text = content.strip()

                segments.append(
                    Segment(
                        start=round(start_sec, 2),
                        end=round(end_sec, 2),
                        text=spoken_text,
                        speaker=speaker,
                    )
                )
                break

    return segments, max_end
