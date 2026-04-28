"""
main.py — Orchestrate frame extraction and transcription for one or more video files.

Usage:
    python main.py <video_path> [<video_path> ...]

Output layout (per video):
    <video_stem>/
    ├── frames/
    │   ├── frame_0001.webp
    │   ├── …
    │   └── frames_manifest.json
    └── transcript/
        ├── transcript_raw.txt
        └── transcript_timestamped.json
"""

import argparse
import json
import sys
import time
from pathlib import Path


def _print_summary(
    output_dir: Path,
    frame_count: int,
    segment_count: int,
    frames_elapsed: float,
    transcript_elapsed: float,
) -> None:
    total = frames_elapsed + transcript_elapsed
    print()
    print("=" * 50)
    print("  Extraction complete")
    print("=" * 50)
    print(f"  Output directory : {output_dir}")
    print(f"  Frames extracted : {frame_count}")
    print(f"  Transcript segs  : {segment_count}")
    print(f"  Frames time      : {frames_elapsed:.1f}s")
    print(f"  Transcript time  : {transcript_elapsed:.1f}s")
    print(f"  Total time       : {total:.1f}s")
    print("=" * 50)


def run(video_path: str | Path, output_base: Path = Path("output"), *, fps: float = 1.0, quality: int = 95, do_frames: bool = True, do_transcript: bool = True, model_name: str = "medium") -> None:
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"video file not found: {video_path}")

    # Build output directory inside output_base
    output_dir = output_base / video_path.stem
    frames_dir = output_dir / "frames"
    transcript_dir = output_dir / "transcript"

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    frames_elapsed = 0.0
    segment_count = 0
    transcript_elapsed = 0.0

    if do_frames:
        try:
            from clip2context.extract_frames import extract_frames
        except ImportError:
            raise ImportError("extract_frames module not found. Make sure clip2context is installed.")

        step = "1/2" if do_transcript else "1/1"
        print(f"[{step}] Extracting frames from: {video_path.name}")
        t0 = time.perf_counter()
        try:
            _, frame_count = extract_frames(video_path, frames_dir, fps, quality)
        except (FileNotFoundError, RuntimeError):
            raise
        except Exception as exc:
            raise RuntimeError(f"Unexpected error during frame extraction: {exc}") from exc
        frames_elapsed = time.perf_counter() - t0
        print(f"    Done — {frame_count} frames in {frames_elapsed:.1f}s")

    # ── Step 2: Extract transcript ────────────────────────────────────────────
    if do_transcript:
        try:
            from clip2context.extract_transcript import extract_transcript
        except ImportError:
            raise ImportError("extract_transcript module not found. Make sure clip2context is installed.")

        step = "2/2" if do_frames else "1/1"
        print(f"[{step}] Transcribing audio from: {video_path.name}")
        t1 = time.perf_counter()
        try:
            extract_transcript(video_path, transcript_dir, model_name)
        except (FileNotFoundError, RuntimeError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unexpected error during transcription: {exc}") from exc
        transcript_elapsed = time.perf_counter() - t1

        # Count segments from the saved JSON
        timestamped_json = transcript_dir / "transcript_timestamped.json"
        if timestamped_json.exists():
            try:
                segments = json.loads(timestamped_json.read_text(encoding="utf-8"))
                segment_count = len(segments)
            except (json.JSONDecodeError, OSError):
                pass

        print(f"    Done — {segment_count} segments in {transcript_elapsed:.1f}s")

    _print_summary(output_dir, frame_count, segment_count, frames_elapsed, transcript_elapsed)


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
        description=(
            "Extract frames and transcript from one or more video files."
            "Output is saved in a directory named after each video."
        )
    )
    parser.add_argument(
        "video_paths",
        nargs="+",
        help="Path(s) to one or more video files or folders containing videos.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Base directory for all output (default: output/).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only-frames",
        action="store_true",
        help="Extract frames only; skip transcription.",
    )
    group.add_argument(
        "--only-transcripts",
        action="store_true",
        help="Extract transcripts only; skip frame extraction.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1.0). Use 0.5 for 1 frame every 2 seconds, 2 for 2 frames per second, etc.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        metavar="1-100",
        help="WebP compression quality (default: 95). Lower values produce smaller files with some quality loss.",
    )
    parser.add_argument(
        "--model",
        default="medium",
        help="Whisper model to use (default: medium). Options: tiny, base, small, medium, large, turbo.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    do_frames = not args.only_transcripts
    do_transcript = not args.only_frames
    output_base = Path(args.output_dir)

    video_paths = _resolve_video_paths(args.video_paths)
    if not video_paths:
        print("Error: no video files to process.", file=sys.stderr)
        sys.exit(1)

    failed: list[str] = []
    for i, video_path in enumerate(video_paths, 1):
        if len(video_paths) > 1:
            print(f"\n[Video {i}/{len(video_paths)}] {video_path}")
        try:
            run(video_path, output_base, fps=args.fps, quality=args.quality, do_frames=do_frames, do_transcript=do_transcript, model_name=args.model)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}", file=sys.stderr)
            failed.append(str(video_path))

    if failed:
        print(f"\n{len(failed)} video(s) failed:", file=sys.stderr)
        for path in failed:
            print(f"  - {path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
