"""
extract_frames.py — Extract frames from one or more video files using FFmpeg.

Usage (CLI):
    python extract_frames.py <video_path> [<video_path> ...] [--output-dir <dir>]

    When processing multiple videos, each video's frames are saved in a
    subdirectory named after the video stem inside <output_dir> (or next to
    each video if --output-dir is not given).

Usage (module):
    from extract_frames import extract_frames
    output_dir, frame_count = extract_frames("video.mp4", "output/frames")
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from clip2context.utils import parse_time


def _seconds_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def extract_frames(video_path: str | Path, output_dir: str | Path, fps: float = 1.0, quality: int = 95, start_time: float | None = None, end_time: float | None = None) -> tuple[Path, int]:
    """
    Extract frames from *video_path* into *output_dir* at the given *fps* rate.
    Frames keep original resolution and are compressed as high-quality WebP.

    *quality* controls WebP compression (1–100; lower = smaller files).
    *start_time* and *end_time* are in seconds; if provided, only extract frames in that range.

    Returns (output_dir_path, frame_count).
    Raises FileNotFoundError if the video does not exist.
    Raises RuntimeError if ffmpeg/ffprobe is not installed or extraction fails.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Verify ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. "
            "See README.md for installation instructions."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command with optional time range
    ffmpeg_cmd = ["ffmpeg", "-i", str(video_path)]
    if start_time is not None:
        ffmpeg_cmd.extend(["-ss", str(start_time)])
    if end_time is not None:
        duration = end_time - (start_time or 0)
        ffmpeg_cmd.extend(["-t", str(duration)])

    # Keep source resolution for text/UI clarity and compress with high-quality
    # WebP so files are much smaller than PNG while staying visually sharp.
    frame_pattern = str(output_dir / "frame_%04d.webp")
    ffmpeg_cmd.extend([
        "-vf", f"fps={fps}",
        "-c:v", "libwebp",
        "-quality", str(quality),
        "-compression_level", "6",
        "-vsync", "vfr",
        "-hide_banner",
        "-loglevel", "error",
        frame_pattern,
    ])
    
    subprocess.run(ffmpeg_cmd, check=True)

    # Collect generated frames and build manifest
    frames = sorted(output_dir.glob("frame_*.webp"))
    frame_count = len(frames)

    interval = 1.0 / fps
    manifest = []
    for idx, frame_file in enumerate(frames):
        timestamp_seconds = idx * interval + (start_time or 0)
        manifest.append(
            {
                "frame_filename": frame_file.name,
                "timestamp_seconds": round(timestamp_seconds, 3),
                "timestamp_formatted": _seconds_to_hms(timestamp_seconds),
            }
        )

    manifest_path = output_dir / "frames_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return output_dir, frame_count


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
        description="Extract 1 frame every 2 seconds from one or more videos using FFmpeg."
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
        "--start-time",
        type=parse_time,
        default=None,
        help="Start time (format: SS, MM:SS, or HH:MM:SS). Extract frames only from this point onward.",
    )
    parser.add_argument(
        "--end-time",
        type=parse_time,
        default=None,
        help="End time (format: SS, MM:SS, or HH:MM:SS). Extract frames only up to this point.",
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
        out = Path(args.output_dir) / p.stem / "frames"
        try:
            output_dir, frame_count = extract_frames(p, out, args.fps, args.quality, args.start_time, args.end_time)
            print(f"Extracted {frame_count} frames to: {output_dir}")
            print(f"Manifest saved to: {output_dir / 'frames_manifest.json'}")
        except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"Error ({video_path}): {exc}", file=sys.stderr)
            failed.append(video_path)

    if failed:
        print(f"\n{len(failed)} video(s) failed:", file=sys.stderr)
        for path in failed:
            print(f"  - {path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
