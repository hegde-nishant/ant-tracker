#!/usr/bin/env python3
"""
Split long videos into smaller chunks for easier annotation.

Usage:
    python utils/split_video.py --video long_video.mp4 --chunks 20 --output chunks/
"""

import os
import subprocess
from pathlib import Path


def get_video_duration(video_path):
    """Get video duration in seconds."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def split_video(input_file, num_chunks=20, output_dir="chunks"):
    """Split video into equal chunks."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Get total duration
    print(f"Getting duration of {input_file}...")
    total_duration = get_video_duration(input_file)
    print(
        f"Total duration: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)"
    )

    # Calculate chunk duration
    chunk_duration = total_duration / num_chunks
    print(
        f"Each chunk will be {chunk_duration:.2f} seconds ({chunk_duration / 60:.2f} minutes)"
    )

    # Split video into chunks
    for i in range(num_chunks):
        start_time = i * chunk_duration
        output_file = os.path.join(output_dir, f"chunk_{i + 1:02d}.mp4")

        print(f"\nProcessing chunk {i + 1}/{num_chunks}...")
        print(f"  Start time: {start_time:.2f}s")
        print(f"  Output: {output_file}")

        cmd = [
            "ffmpeg",
            "-ss",
            str(start_time),
            "-i",
            input_file,
            "-t",
            str(chunk_duration),
            "-c",
            "copy",  # Copy codec without re-encoding for speed
            "-avoid_negative_ts",
            "1",
            output_file,
            "-y",  # Overwrite output file if exists
        ]

        subprocess.run(cmd, check=True)
        print(f"  ✓ Chunk {i + 1} completed")

    print(f"\n✓ All {num_chunks} chunks created in '{output_dir}' directory!")


if __name__ == "__main__":
    input_video = "av7.mp4"

    if not os.path.exists(input_video):
        print(f"Error: {input_video} not found!")
        exit(1)

    split_video(input_video, num_chunks=20, output_dir="chunks")
