"""
extract_transcript.py — Transcribe one or more video files using OpenAI Whisper.

Usage (CLI):
    python extract_transcript.py <video_path> [<video_path> ...] [--output-dir <dir>]

    When processing multiple videos, each video's transcript is saved in a
    subdirectory named after the video stem inside <output_dir> (or next to
    each video if --output-dir is not given).

Usage (module):
    from extract_transcript import extract_transcript
    output_dir = extract_transcript("video.mp4", "output/transcript")
"""

import argparse
import json
import sys
from pathlib import Path


def _seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_transcript(video_path: str | Path, output_dir: str | Path) -> Path:
    """
    Transcribe *video_path* using Whisper (medium model, English only).

    Saves:
      - transcript_raw.txt          — plain text
      - transcript_timestamped.json — list of {start, end, text} segments
      - transcript_timed.txt        — [HH:MM:SS] text, one line per segment

    Returns the output directory path.
    Raises FileNotFoundError if the video does not exist.
    Raises RuntimeError if openai-whisper is not installed.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        import whisper  # type: ignore[import]
    except ImportError:
        raise RuntimeError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Whisper model (medium)…")
    model = whisper.load_model("medium")

    print(f"Transcribing {video_path.name}…")
    result = model.transcribe(
        str(video_path),
        language="en",
        verbose=False,
        word_timestamps=True,
    )

    # Plain text transcript
    raw_text: str = result["text"].strip()
    raw_path = output_dir / "transcript_raw.txt"
    raw_path.write_text(raw_text, encoding="utf-8")

    # Timestamped segments — use first word's start time for accuracy
    segments = []
    for seg in result["segments"]:
        words = seg.get("words", [])
        start = words[0]["start"] if words else seg["start"]
        end = words[-1]["end"] if words else seg["end"]
        segments.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": seg["text"].strip(),
        })
    timestamped_path = output_dir / "transcript_timestamped.json"
    timestamped_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")

    # Timestamped plain text
    timed_lines = [f"[{_seconds_to_hms(seg['start'])}] {seg['text']}" for seg in segments]
    timed_path = output_dir / "transcript_timed.txt"
    timed_path.write_text("\n".join(timed_lines), encoding="utf-8")

    return output_dir


_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}


def _resolve_video_paths(inputs: list[str]) -> list[Path]:
    """Expand files and folders into a flat, sorted list of video paths."""
    resolved: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            videos = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in _VIDEO_EXTENSIONS)
            if not videos:
                print(f"Warning: no video files found in directory: {p}", file=sys.stderr)
            resolved.extend(videos)
        elif p.is_file():
            resolved.append(p)
        else:
            print(f"Warning: path not found, skipping: {p}", file=sys.stderr)
    return resolved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe one or more video files using OpenAI Whisper (medium model, English)."
    )
    parser.add_argument(
        "video_paths",
        nargs="+",
        help="Path(s) to one or more video files or folders containing videos.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Base directory for output (default: output/). Each video gets its own subdirectory named after the video stem.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    video_paths = _resolve_video_paths(args.video_paths)
    if not video_paths:
        print("Error: no video files to process.", file=sys.stderr)
        sys.exit(1)

    failed: list[str] = []
    for video_path in video_paths:
        p = Path(video_path)
        out = Path(args.output_dir) / p.stem / "transcript"
        try:
            output_dir = extract_transcript(p, out)
            print(f"Transcript saved to: {output_dir}")
            print(f"  - {output_dir / 'transcript_raw.txt'}")
            print(f"  - {output_dir / 'transcript_timestamped.json'}")
            print(f"  - {output_dir / 'transcript_timed.txt'}")
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Error ({video_path}): {exc}", file=sys.stderr)
            failed.append(video_path)

    if failed:
        print(f"\n{len(failed)} video(s) failed:", file=sys.stderr)
        for path in failed:
            print(f"  - {path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
