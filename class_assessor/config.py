"""Configuration for the class assessor.

Loads environment variables from a local ``.env`` file (if present) and
exposes them as module-level constants. All values have sensible defaults
so the application can start without a ``.env`` file.
"""

import os

from dotenv import load_dotenv

# Load variables from a .env file in the current working directory (if any).
load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# --- LLM settings -----------------------------------------------------------
LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("DEEPSEEK_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
    or ""
)
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"

# --- Whisper / speech-to-text settings --------------------------------------
WHISPER_MODEL = os.getenv("WHISPER_MODEL") or os.getenv("WHISPER_MODEL_SIZE") or "small"
HF_TOKEN = os.getenv("HF_TOKEN", "")

# --- Media / frame extraction settings --------------------------------------
FRAME_INTERVAL_SEC = float(os.getenv("FRAME_INTERVAL_SEC", "20"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "36"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))

# --- Privacy / cloud settings -----------------------------------------------
SEND_FRAMES_TO_CLOUD = os.getenv("SEND_FRAMES_TO_CLOUD", "false").lower() == "true"
