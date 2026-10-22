"""Gradio UI for the class quality assessor.

Accepts an official .VTT transcript or video file/URL, runs the pedagogical
assessment pipeline, and renders a diagnostic Markdown audit report.
Designed according to the New Era Blueprint: zero emojis, clinical typography,
evidence-first layout, and calm visual hierarchy.
"""

import tempfile
from pathlib import Path

import gradio as gr
import requests

from class_assessor import config, pipeline


CUSTOM_CSS = """
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    max-width: 1040px !important;
    margin: 0 auto !important;
    padding-top: 1.5rem !important;
}
.eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 4px;
}
.title-text {
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #f8fafc;
    margin: 0 0 6px 0;
}
.sub-text {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-bottom: 12px;
    line-height: 1.5;
}
.privacy-bar {
    font-size: 0.8rem;
    color: #64748b;
    border-left: 2px solid #334155;
    padding-left: 10px;
    margin-bottom: 22px;
    line-height: 1.4;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-poor {
    background: rgba(239, 68, 68, 0.12);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
}
.badge-average {
    background: rgba(245, 158, 11, 0.12);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.badge-good {
    background: rgba(59, 130, 246, 0.12);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.35);
}
.badge-excellent {
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}
.audit-card {
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid #1f2937;
    border-radius: 6px;
    padding: 14px 18px;
    margin-top: 12px;
    margin-bottom: 16px;
}
.audit-card-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 6px;
}

/* --- New Era Styling for API Documentation & Settings Views --- */
div[role="dialog"],
div[data-testid="modal"],
.gradio-container div[class*="modal"] {
    background: #0b0f19 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 8px !important;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8) !important;
}

pre, code, .code-block, div[class*="code"] {
    background: #0d1117 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
}

button[role="tab"] {
    border-radius: 4px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}

select, input[type="text"], input[type="checkbox"] {
    background-color: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 4px !important;
    color: #f3f4f6 !important;
}

footer {
    display: none !important;
}
"""


def _download(url: str) -> Path:
    """Download video from URL (supports YouTube, Vimeo, direct MP4/MOV links)."""
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024

    # 1. First attempt extraction using yt-dlp (handles YouTube, Loom, Vimeo, etc.)
    try:
        import yt_dlp
        from class_assessor.media import _get_ffmpeg

        tmp_dir = tempfile.mkdtemp()
        output_template = str(Path(tmp_dir) / "video.%(ext)s")
        ydl_opts = {
            "ffmpeg_location": _get_ffmpeg(),
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "outtmpl": output_template,
            "max_filesize": max_bytes,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        matches = list(Path(tmp_dir).glob("video.*"))
        if matches:
            return matches[0]
    except Exception:
        # Fall back to standard HTTP stream
        pass

    # 2. Standard HTTP streaming fallback for direct video URLs
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise gr.Error(f"Failed to download video: {exc}") from exc

    content_type = response.headers.get("content-type", "").lower()
    if "text/html" in content_type:
        raise gr.Error(
            "The URL points to a webpage rather than a playable video stream. "
            "Please upload the video file or provide a direct video URL."
        )

    suffix = Path(url.split("?")[0]).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    downloaded = 0
    try:
        with tmp as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    raise gr.Error(
                        f"Video exceeds the {config.MAX_UPLOAD_MB} MB limit."
                    )
                fh.write(chunk)
    except requests.RequestException as exc:
        raise gr.Error(f"Download interrupted: {exc}") from exc
    return Path(tmp.name)


def format_report(result) -> str:
    """Format an :class:`Assessment` object into clean New Era diagnostic Markdown."""
    badge_class = {
        "Excellent": "badge-excellent",
        "Good": "badge-good",
        "Average": "badge-average",
        "Poor": "badge-poor",
    }.get(result.label, "badge-average")

    badge_html = f'<span class="badge {badge_class}">{result.label.upper()}</span>'

    lines = [
        f"### {badge_html} &nbsp;&nbsp; <b>Score: {result.overall_score:.2f} / 5.00</b> &nbsp;&nbsp; <span style='color:#64748b; font-size:0.85rem;'>CONFIDENCE: {result.confidence.upper()}</span>",
        "",
        '<div class="audit-card">',
        '  <div class="audit-card-title">EXECUTIVE SUMMARY</div>',
        f'  <p style="margin:0; font-size:0.95rem; line-height:1.6; color:#e2e8f0;">{result.summary}</p>',
        '</div>',
        "",
        "### Observed Strengths",
    ]
    lines.extend(f"- {item}" for item in result.strengths)
    lines.extend(["", "### Deficiencies & Recommended Adjustments"])
    lines.extend(f"- {item}" for item in result.weaknesses)
    lines.extend(["", "### Rubric Dimension Breakdown & Citations"])
    for dim in result.dimensions:
        dim_title = dim.name.replace("_", " ").title()
        lines.append(f"#### {dim_title} · Score: {dim.score} / 5")
        for ev in dim.evidence:
            lines.append(f"- `[{ev.timestamp:.1f}s]` {ev.observation}")
        lines.append("")
    return "\n".join(lines)


def assess(vtt_file, file, url, fast_mode: bool = True):
    """Run the assessment pipeline yielding live streaming progress updates."""
    clean_url = (url or "").strip()
    if not file and not clean_url and not vtt_file:
        raise gr.Error("Please upload a .VTT transcript, upload a video recording, or provide a video URL.")

    video = None
    if file:
        video = Path(file)
    elif clean_url:
        yield "### Ingesting media stream from URL..."
        video = _download(clean_url)

    vtt = Path(vtt_file) if vtt_file else None

    try:
        for status_msg, result in pipeline.run_stream(
            video_path=video, vtt_path=vtt, fast_mode=fast_mode
        ):
            if result is None:
                yield status_msg
            else:
                yield format_report(result)
    except Exception as exc:
        raise gr.Error(f"Assessment failed: {exc}") from exc


with gr.Blocks(title="Class Quality Assessor — Humaniti AI") as demo:
    gr.HTML(
        """
        <div class="eyebrow">HUMANITI AI · AUTOMATED PEDAGOGICAL AUDITOR</div>
        <h1 class="title-text">Class Quality Assessor</h1>
        <div class="sub-text">Autonomous multimodal evaluation of children's online tutoring against empirical pedagogical benchmarks.</div>
        <div class="privacy-bar">Ephemeral Processing: Recordings and transcripts are processed in temporary local memory and wiped immediately after analysis. Individuals are never identified.</div>
        """
    )

    with gr.Tabs():
        with gr.TabItem("Official .VTT Transcript (Recommended — 3s)"):
            gr.Markdown(
                "*Directly ingests Zoom Cloud transcripts with pre-labeled speaker turns. Bypasses speech-to-text models for instant evaluation.*"
            )
            vtt_input = gr.File(
                label="Upload WebVTT Transcript (.vtt)",
                file_types=[".vtt", "text"],
            )

        with gr.TabItem("Video Recording (MP4 / Web URL)"):
            gr.Markdown(
                "*Extracts audio for local faster-whisper transcription and samples visual keyframes across the session timeline.*"
            )
            with gr.Row():
                file_input = gr.File(
                    label="Upload Video Recording (MP4, MOV, MKV)",
                    file_types=["video"],
                )
                url_input = gr.Textbox(
                    label="Or Paste a Direct Video / YouTube URL",
                    placeholder="https://example.com/class_recording.mp4",
                )

    with gr.Row():
        fast_mode_cb = gr.Checkbox(
            label="Fast Analysis (derives conversational metrics without heavy CPU neural clustering)",
            value=True,
        )

    assess_btn = gr.Button("Run Pedagogical Assessment", variant="primary", size="lg")

    report = gr.Markdown(
        value="*Upload a .VTT transcript or video recording and click 'Run Pedagogical Assessment' to begin.*",
        label="Diagnostic Assessment Report",
    )

    # Disable button immediately upon click -> run streaming pipeline -> re-enable upon completion
    assess_btn.click(
        lambda: gr.Button(interactive=False, value="Assessing Recording..."),
        outputs=assess_btn,
        api_name=False,
    ).then(
        assess,
        inputs=[vtt_input, file_input, url_input, fast_mode_cb],
        outputs=report,
        api_name="assess_class",
    ).then(
        lambda: gr.Button(interactive=True, value="Run Pedagogical Assessment"),
        outputs=assess_btn,
        api_name=False,
    )


if __name__ == "__main__":
    demo.queue(max_size=2).launch(
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        footer_links=[],
    )
