"""LLM-based assessor for the class assessor.

Builds a prompt from a :class:`FeatureBundle`, calls an OpenAI-compatible
chat completion endpoint, validates the structured JSON response, and
calibrates the result into a final :class:`Assessment`.
"""

import json
import os
from pathlib import Path

from openai import OpenAI

from class_assessor import config
from class_assessor.schemas import Assessment, FeatureBundle, LLMOutput

# --- Rubric weights ---------------------------------------------------------
WEIGHTS = {
    "engagement": 0.25,
    "rapport": 0.25,
    "clarity": 0.20,
    "pacing": 0.15,
    "content": 0.10,
    "tech_management": 0.05,
}

SYSTEM = (
    "You are the senior pedagogical auditor for 8x8 children's live online tutoring classes. "
    "Score each teaching dimension on a 1-5 integer scale strictly according to official 8x8 ground-truth benchmarks:\n\n"
    "1. rapport (Weight: 0.25) - Warmth, Energy, and Introductions:\n"
    "   - 5 (Excellent): Enthusiastic, smiling, and pleasant tone. Welcomes every student by name, builds strong connection.\n"
    "   - 4 (Good): Good introduction, polite greeting, pleasant interaction.\n"
    "   - 3 (Average): Functional greeting but monotone or flat communication ('needs more energetic and pleasant interaction').\n"
    "   - 1-2 (Weak/Poor): Critical failure - directly enters class/revision with NO rapport building, or fails to ask new joiners for self-introduction, or cold/unpleasant tone.\n\n"
    "2. engagement (Weight: 0.25) - Individual Student Consideration:\n"
    "   - 5 (Excellent): Calls every student individually by name, gives multiple turns to everyone during puzzles/board work, ensures 100% active participation.\n"
    "   - 4 (Good): Considers everyone in class, asks individual questions to each student.\n"
    "   - 3 (Average): Uneven consideration (interacts primarily with 1 student while others wait).\n"
    "   - 1-2 (Weak/Poor): Teacher monologue (>70% talk), ignores students, or does not check understanding.\n\n"
    "3. clarity (Weight: 0.20) - Concept Explanation:\n"
    "   - 5 (Excellent): Crystal-clear structured breakdown of chess tactics/concepts with continuous checks for understanding.\n"
    "   - 4 (Good): Clear explanations with understandable examples.\n"
    "   - 3 (Average): Functional but mechanical explanation.\n"
    "   - 1-2 (Weak/Poor): Confusing instructions, student struggles without guidance.\n\n"
    "4. pacing (Weight: 0.15) - Time & Flow Management:\n"
    "   - 5 (Excellent): Smooth flow, zero awkward silences, proper recap summary and warm farewell.\n"
    "   - 4 (Good): Good time management throughout.\n"
    "   - 3 (Average): Noticeable dead air (>5-10s pauses) or ends prematurely (>5 mins early).\n"
    "   - 1-2 (Weak/Poor): Abrupt cutoff, no summary conclusion, or frequent lengthy silences (>15s).\n\n"
    "5. content (Weight: 0.10) - Lesson Structure & Homework:\n"
    "   - 5 (Excellent): Clear topic progression AND explicitly assigns Homework (HW) with instructions.\n"
    "   - 4 (Good): Strong lesson delivery, mentions practice/next class.\n"
    "   - 3 (Average): Solid content but fails to give homework ('should give HWs').\n"
    "   - 1-2 (Weak/Poor): Disorganized content, no homework, no lesson structure.\n\n"
    "6. tech_management (Weight: 0.05) - Tech & Environment:\n"
    "   - 5 (Excellent): Virtual background, camera on, clear audio. (When evaluating transcript-only without video, default to 4-5 if audio is clear).\n"
    "   - 3 (Average): Dim lighting, background glitching, or background noise.\n"
    "   - 1-2 (Weak/Poor): Camera off during lesson, face not visible, or severe audio disruption.\n\n"
    "CRITICAL 8x8 QUALITY HARD GATES (Strictly enforce based on official human QA audits):\n"
    "- Introduction & New Joiners: Tutors must dedicate the first 2-3 minutes to warm introductions. If new students join the class and the tutor fails to give them dedicated time for self-introduction, or jumps directly into chess moves/revision without rapport building, RAPPORT MUST BE SCORED 1 or 2.\n"
    "- Tone & Pleasantness: If communication is flat, monotone, or mechanical rather than energetic, pleasant, and happy, RAPPORT MUST BE SCORED 1 or 2.\n"
    "- Uneven Consideration: If the tutor interacts primarily with 1 student while another student sits idle without equal turns, ENGAGEMENT MUST BE CAPPED AT 2 or 3.\n"
    "- Abrupt Conclusion: If the class ends abruptly without a recap summary and warm signoff, PACING MUST BE SCORED 1 or 2.\n"
    "- Homework: Explicitly check for homework. If given with clear instructions award 4-5 in content; if no homework is assigned, cap content at 3.\n\n"
    "LANGUAGE & CODE-MIXING (MANGLISH / INDIC ENGLISH):\n"
    "- Classes frequently use English or code-mixed English and Malayalam (Manglish) or Indian English.\n"
    "- Do NOT penalize tutors or students for natural bilingual code-switching (e.g. 'manasilayo', 'clear aano', 'doubt undo', 'nannayi cheythu').\n"
    "- Evaluate pedagogical intent, rapport, and instructional clarity equally across English and Manglish expressions.\n\n"
    "Provide specific timestamped evidence quotes for every dimension. Do NOT give polite inflation: score strictly according to the evidence."
)


def build_prompt(bundle: FeatureBundle) -> str:
    """Build the user prompt string from a :class:`FeatureBundle`."""
    audio_json = bundle.audio.model_dump_json(indent=2)
    visual_json = bundle.visual.model_dump_json(indent=2)
    frame_notes = "\n".join(
        f"- {note.timestamp}s: {note.note}" for note in bundle.frame_notes
    ) or "(none)"
    transcript_text = "\n".join(
        f"[{seg.start:.1f}-{seg.end:.1f}]"
        + (f" {seg.speaker}:" if seg.speaker else "")
        + f" {seg.text}"
        for seg in bundle.transcript
    )[:60000]
    schema = json.dumps(LLMOutput.model_json_schema(), indent=2)

    return (
        "Evaluate the following classroom recording against 8x8 quality standards.\n\n"
        f"Rubric weights (dimension: weight):\n{json.dumps(WEIGHTS, indent=2)}\n\n"
        f"Audio metrics:\n{audio_json}\n\n"
        f"Visual metrics:\n{visual_json}\n\n"
        f"Frame notes:\n{frame_notes}\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        f"Output JSON schema:\n{schema}\n"
    )


def _call(prompt: str) -> LLMOutput:
    """Call the LLM and parse its JSON response into :class:`LLMOutput`."""
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
    if base_url:
        base_url = base_url.removesuffix("/chat/completions")
    elif "deepseek" in config.LLM_MODEL:
        base_url = "https://api.deepseek.com"

    client = OpenAI(api_key=config.LLM_API_KEY, base_url=base_url)
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    raw_text = response.choices[0].message.content or ""
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text
        if raw_text.endswith("```"):
            raw_text = raw_text[: -len("```")]
        raw_text = raw_text.strip()
    return LLMOutput.model_validate_json(raw_text)


def calibrate(out: LLMOutput) -> Assessment:
    """Calibrate an :class:`LLMOutput` into a final :class:`Assessment`."""
    calibration_path = Path(__file__).parent / "calibration.json"
    with calibration_path.open("r", encoding="utf-8") as fh:
        calibration = json.load(fh)

    thresholds = calibration.get(
        "thresholds",
        {"Poor": 0.0, "Average": 2.5, "Good": 4.0, "Excellent": 4.5},
    )
    confidence_margin = float(calibration.get("confidence_margin", 0.20))

    weighted = 0.0
    total_weight = 0.0
    for dim in out.dimensions:
        weight = WEIGHTS.get(dim.name, 0.0)
        weighted += dim.score * weight
        total_weight += weight
    overall_score = weighted / total_weight if total_weight > 0 else 0.0

    label = "Poor"
    for name in ("Average", "Good", "Excellent"):
        if overall_score >= thresholds.get(name, float("inf")):
            label = name

    margin = min(abs(overall_score - t) for t in thresholds.values())
    if margin >= confidence_margin:
        confidence = "high"
    elif margin >= confidence_margin / 2:
        confidence = "medium"
    else:
        confidence = "low"

    return Assessment(
        dimensions=out.dimensions,
        strengths=out.strengths,
        weaknesses=out.weaknesses,
        summary=out.summary,
        overall_score=round(overall_score, 2),
        label=label,
        confidence=confidence,
    )


def assess(bundle: FeatureBundle) -> Assessment:
    """Run the full assessment with one retry on validation error."""
    prompt = build_prompt(bundle)
    try:
        return calibrate(_call(prompt))
    except Exception:
        return calibrate(_call(prompt))
