"""Shared utility functions for clip2context."""


def parse_time(time_str: str) -> float:
    """
    Parse time string in formats: SS, MM:SS, or HH:MM:SS.
    Examples: "40" -> 40, "05:40" -> 340, "1:05:40" -> 3940
    """
    parts = time_str.split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        minutes, seconds = float(parts[0]), float(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = float(parts[0]), float(parts[1]), float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid time format: {time_str}. Use SS, MM:SS, or HH:MM:SS")
