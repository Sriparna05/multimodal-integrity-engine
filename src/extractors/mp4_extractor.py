"""MP4 audio extractor and transcriber using faster-whisper with timestamp tracking."""

import logging
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Optional
import wave

import faster_whisper
import imageio_ffmpeg
import numpy as np
import torch

from src.models import DocumentResult
from src.text_processing import clean_text, chunk_audio_segments

# Suppress symlink warning on Windows if not running as administrator
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

logger = logging.getLogger(__name__)

# Cached Whisper model instance
_WHISPER_MODEL: Optional[faster_whisper.WhisperModel] = None


def format_timestamp(seconds: float) -> str:
    """Formats a duration or offset in seconds to HH:MM:SS format (e.g. 00:00:24)."""
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_whisper_model(model_size: str = "tiny.en") -> faster_whisper.WhisperModel:
    """Returns a lazily-instantiated faster-whisper model."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        logger.info(f"Loading Whisper model '{model_size}' on {device} ({compute_type})...")
        _WHISPER_MODEL = faster_whisper.WhisperModel(model_size, device=device, compute_type=compute_type)
    return _WHISPER_MODEL


def extract_audio_waveform(video_path: Path) -> Optional[np.ndarray]:
    """Extracts a mono 16kHz float32 audio waveform from a video using FFmpeg.

    Returns:
        1D float32 numpy array sampled at 16000Hz, or None if no audio track exists
        or the audio is silent.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    temp_wav = Path(tempfile.mktemp(suffix=".wav"))

    try:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(temp_wav),
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if proc.returncode != 0 or not temp_wav.exists() or temp_wav.stat().st_size <= 44:
            return None

        with wave.open(str(temp_wav), "rb") as wf:
            n_frames = wf.getnframes()
            if n_frames == 0:
                return None
            raw_frames = wf.readframes(n_frames)

        # Convert 16-bit signed PCM to float32 normalized between -1.0 and 1.0
        audio_array = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0

        # Check for absolute silence (RMS amplitude below noise floor threshold)
        rms = np.sqrt(np.mean(audio_array ** 2)) if audio_array.size > 0 else 0.0
        if rms < 1e-4:
            return None

        return audio_array

    except Exception as exc:
        logger.warning(f"Audio extraction failed on {video_path.name}: {exc}")
        return None
    finally:
        if temp_wav.exists():
            try:
                temp_wav.unlink()
            except OSError:
                pass


def extract_mp4(file_path: Path | str, document_id: Optional[str] = None) -> DocumentResult:
    """Extracts and transcribes audio from an MP4 file, capturing segment timestamps.

    Catches silent or non-audio videos gracefully and returns a warning without crashing.
    """
    path = Path(file_path)
    doc_id = document_id or path.name

    if not path.exists():
        msg = f"MP4 file does not exist: {path}"
        logger.error(msg)
        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality="mp4",
            status="error",
            error_message=msg,
        )

    if path.stat().st_size == 0:
        msg = f"MP4 file is empty (0 bytes): {path.name}"
        logger.warning(msg)
        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality="mp4",
            status="error",
            error_message=msg,
        )

    audio_array = extract_audio_waveform(path)
    if audio_array is None or audio_array.size == 0:
        msg = f"Video has no audio track or is completely silent: {path.name}"
        logger.warning(msg)
        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality="mp4",
            status="warning",
            error_message=msg,
            chunks=[],
            full_text="",
        )

    try:
        model = get_whisper_model("tiny.en")
        segments_gen, info = model.transcribe(audio_array, beam_size=5, language="en")

        collected_segments: list[dict] = []
        for s in segments_gen:
            clean_s = clean_text(s.text)
            if clean_s:
                collected_segments.append({
                    "start": s.start,
                    "end": s.end,
                    "text": clean_s,
                })

        if not collected_segments:
            msg = f"No speech recognized in audio track: {path.name}"
            logger.warning(msg)
            return DocumentResult(
                document_id=doc_id,
                file_path=str(path),
                modality="mp4",
                status="warning",
                error_message=msg,
                chunks=[],
                full_text="",
            )

        full_text = " ".join(s["text"] for s in collected_segments).strip()
        chunks = chunk_audio_segments(collected_segments, document_id=doc_id, modality="mp4")

        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality="mp4",
            status="ok",
            chunks=chunks,
            full_text=full_text,
        )

    except Exception as exc:
        msg = f"Failed to transcribe MP4 {path.name}: {exc}"
        logger.warning(msg)
        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality="mp4",
            status="error",
            error_message=msg,
        )
