#!/usr/bin/env python3
"""
Extract frames from videos for creating training datasets.

Extracts evenly-spaced frames for annotation.

Usage:
    python utils/extract_frames.py --video video.mp4 --output frames/ --interval 30

Interval: 10-15 for fast motion, 30-60 for slow motion.
"""

import glob
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


def extract_frames_from_video(video_path, output_dir, num_frames=7):
    """Extract evenly-spaced frames from a video."""
    # Get video duration
    duration = get_video_duration(video_path)

    # Calculate time interval between frames
    # We extract frames at evenly spaced intervals
    interval = duration / (num_frames + 1)

    # Extract frames
    video_name = Path(video_path).stem
    frames_extracted = 0

    for i in range(1, num_frames + 1):
        timestamp = i * interval
        output_file = os.path.join(output_dir, f"{video_name}_frame_{i:02d}.jpg")

        cmd = [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-q:v",
            "2",  # High quality JPEG
            output_file,
            "-y",
        ]

        # Run quietly
        subprocess.run(cmd, capture_output=True, check=True)
        frames_extracted += 1

    return frames_extracted


def process_all_chunks(
    chunks_dir="chunks", frames_dir="frames", num_frames_per_video=7
):
    """Process all video chunks and extract frames."""
    # Create frames directory
    Path(frames_dir).mkdir(exist_ok=True)

    # Get all chunk files
    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.mp4")))

    if not chunk_files:
        print(f"Error: No chunk files found in '{chunks_dir}' directory!")
        return

    print(f"Found {len(chunk_files)} chunk files")
    print(f"Extracting {num_frames_per_video} frames from each chunk...\n")

    total_frames = 0

    for idx, chunk_file in enumerate(chunk_files, 1):
        chunk_name = os.path.basename(chunk_file)
        print(f"Processing {chunk_name} ({idx}/{len(chunk_files)})...")

        frames_count = extract_frames_from_video(
            chunk_file, frames_dir, num_frames=num_frames_per_video
        )

        total_frames += frames_count
        print(f"  ✓ Extracted {frames_count} frames")

    print(
        f"\n✓ Complete! Extracted {total_frames} total frames to '{frames_dir}' directory"
    )


if __name__ == "__main__":
    process_all_chunks(chunks_dir="chunks", frames_dir="frames", num_frames_per_video=7)
