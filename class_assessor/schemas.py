"""Pydantic schemas for the class assessor.

Defines the data contracts shared across the media pipeline, audio/visual
analysis, the LLM assessor, and the final assessment output.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

# --- Type aliases -----------------------------------------------------------
Label = Literal["Poor", "Average", "Good", "Excellent"]
DimensionName = Literal[
    "engagement",
    "clarity",
    "pacing",
    "rapport",
    "content",
    "tech_management",
]


# --- Transcript / audio -----------------------------------------------------
class Segment(BaseModel):
    """A single transcript segment."""

    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class AudioMetrics(BaseModel):
    """Derived audio features for a class recording."""

    duration_sec: float
    speech_ratio: float
    words_per_minute: float
    long_pause_times: list[float]
    question_times: list[float]
    teacher_talk_share: Optional[float] = None
    speaker_count: Optional[int] = None
    energy_variation: float


# --- Visual -----------------------------------------------------------------
class FrameNote(BaseModel):
    """A short observation tied to a specific frame timestamp."""

    timestamp: float
    note: str


class VisualMetrics(BaseModel):
    """Derived visual features for a class recording."""

    frames_sampled: int
    mean_change: float
    static_stretches: list[float]


# --- Aggregated features ----------------------------------------------------
class FeatureBundle(BaseModel):
    """All features gathered before LLM assessment."""

    transcript: list[Segment]
    audio: AudioMetrics
    visual: VisualMetrics
    frame_notes: list[FrameNote] = []


# --- LLM output -------------------------------------------------------------
class Evidence(BaseModel):
    """A timestamped observation supporting a dimension score."""

    timestamp: float
    observation: str = Field(min_length=10)


class DimensionScore(BaseModel):
    """Score for a single teaching dimension, with supporting evidence."""

    name: DimensionName
    score: int = Field(ge=1, le=5)
    evidence: list[Evidence] = Field(min_length=2)


class LLMOutput(BaseModel):
    """Structured output returned by the LLM assessor."""

    dimensions: list[DimensionScore]
    strengths: list[str]
    weaknesses: list[str]
    summary: str


class Assessment(LLMOutput):
    """Final assessment, extending the LLM output with aggregate fields."""

    overall_score: float
    label: Label
    confidence: Literal["low", "medium", "high"]
