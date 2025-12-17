#!/usr/bin/env python3
"""
Track ants in videos and assign each ant a unique ID.

Detects ants in each frame, tracks them over time, and creates:
- Video with boxes around ants
- CSV file with ant positions and IDs
- Statistics (optional)

Uses BoT-SORT tracking to maintain IDs even when ants temporarily disappear.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from src.preprocessing.image_enhancement import ImageEnhancer
import subprocess
import shutil
import glob
import torch


def get_device():
    """Return GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        return 0
    else:
        return 'cpu'


def frames_to_video_ffmpeg(frames_dir, output_video, fps=30, pattern="frame_%06d.jpg"):
    """Create video from frames using ffmpeg or OpenCV."""
    frames_dir = Path(frames_dir)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    # Check if ffmpeg is available
    ffmpeg_available = shutil.which("ffmpeg") is not None

    if ffmpeg_available:
        # Try ffmpeg first
        print(f"Creating video from frames using ffmpeg...")

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-framerate", str(fps),
            "-i", str(frames_dir / pattern),
            "-c:v", "libx264",  # H.264 codec
            "-pix_fmt", "yuv420p",  # Compatibility
            "-crf", "23",  # Quality (lower = better, 18-28 is good range)
            str(output_video)
        ]

        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"\nVideo created successfully with ffmpeg: {output_video}")
            return str(output_video)
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed: {e.stderr}")
            print("Falling back to OpenCV VideoWriter...")
        except FileNotFoundError:
            print("ffmpeg not found, falling back to OpenCV VideoWriter...")
    else:
        print("ffmpeg not available, using OpenCV VideoWriter...")

    # Fallback to OpenCV VideoWriter
    print(f"Creating video from frames using OpenCV...")

    # Get list of frame files
    frame_files = sorted(glob.glob(str(frames_dir / "frame_*.jpg")))

    if not frame_files:
        raise ValueError(f"No frame files found in {frames_dir}")

    print(f"Found {len(frame_files)} frames")

    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {frame_files[0]}")

    height, width = first_frame.shape[:2]
    print(f"Frame size: {width}x{height}")

    # Try multiple codecs
    codecs_to_try = [
        ('avc1', 'H.264'),
        ('X264', 'x264'),
        ('XVID', 'XVID'),
        ('MJPG', 'Motion JPEG'),
    ]

    out = None
    for codec_code, codec_name in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_code)
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
            if out.isOpened():
                print(f"Using codec: {codec_name} ({codec_code})")
                break
            else:
                print(f"Codec {codec_name} ({codec_code}) not available, trying next...")
        except Exception as e:
            print(f"Failed to initialize {codec_name} ({codec_code}): {e}")
            continue

    if out is None or not out.isOpened():
        raise RuntimeError("Failed to initialize video writer with any available codec")

    # Write frames
    print("Writing frames to video...")
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        if frame is None:
            print(f"Warning: Could not read frame {frame_file}, skipping...")
            continue
        out.write(frame)

        if (i + 1) % 30 == 0:
            print(f"Written {i + 1}/{len(frame_files)} frames ({(i + 1) / len(frame_files) * 100:.1f}%)")

    out.release()
    print(f"\nVideo created successfully with OpenCV: {output_video}")
    return str(output_video)


def track_video_with_roi_and_stats(
    model_path,
    source,
    output_path=None,
    conf_threshold=0.25,
    iou_threshold=0.7,
    imgsz=640,
    device=None,
    tracker="botsort.yaml",
    roi=None,
    fps=None,
    line_width=2,
    trail_length=30,
    enable_statistics=False,
    stats_start_time=0.0,
    stats_end_time=0.0,
    enhancement_type='none',
    super_res_scale=1.0,
    show_trails=False,
    show_ids=False,
):
    """Track ants in video with optional ROI, enhancement, and statistics."""
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
        print(f"Auto-detected device: {device}")

    # Initialize image enhancer
    enhancer = ImageEnhancer(enhancement_type=enhancement_type, scale_factor=super_res_scale)

    # Load model
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {source}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_fps = fps if fps else original_fps

    # Apply ROI if specified
    if roi:
        x1, y1, x2, y2 = roi
        output_width = x2 - x1
        output_height = y2 - y1
    else:
        x1, y1, x2, y2 = 0, 0, width, height
        output_width = width
        output_height = height

    print(f"Video: {width}x{height} @ {original_fps} FPS, {total_frames} frames")
    if fps and fps != original_fps:
        print(f"Output FPS: {output_fps} (custom - video will play at {original_fps/output_fps:.2f}x {'faster' if fps < original_fps else 'slower'} speed)")
    else:
        print(f"Output FPS: {output_fps} (original)")
    if roi:
        print(f"ROI: ({x1}, {y1}) to ({x2}, {y2}), Output: {output_width}x{output_height}")
    if enhancement_type != 'none':
        from src.preprocessing.image_enhancement import get_enhancement_description
        print(f"Enhancement: {get_enhancement_description(enhancement_type)}")
        if super_res_scale > 1.0:
            print(f"Super-Resolution: {super_res_scale}x upscaling")
            output_width = int(output_width * super_res_scale)
            output_height = int(output_height * super_res_scale)
            print(f"Enhanced output size: {output_width}x{output_height}")

    # Setup for frame saving (more reliable than VideoWriter)
    frames_dir = None
    if output_path:
        # Create frames directory named after the video
        output_path_obj = Path(output_path)
        video_name = Path(source).stem  # Get video filename without extension
        frames_dir = output_path_obj.parent / f"{video_name}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"Frames will be saved to: {frames_dir}")

    # Tracking data
    track_history = {}  # {track_id: [(frame, x, y, timestamp), ...]}
    colors = {}  # {track_id: (B, G, R)}
    frame_count = 0

    # Statistics
    frame_detections = []  # List of detection counts per frame
    stats_start_frame = int(stats_start_time * original_fps)
    stats_end_frame = int(stats_end_time * original_fps) if stats_end_time > 0 else total_frames

    print("Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / original_fps

        # Apply ROI crop
        if roi:
            frame_cropped = frame[y1:y2, x1:x2].copy()
        else:
            frame_cropped = frame.copy()

        # Apply image enhancement
        if enhancement_type != 'none':
            frame_cropped = enhancer.enhance(frame_cropped)

        # Run tracking
        results = model.track(
            frame_cropped,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            tracker=tracker,
            persist=True,
            verbose=False,
        )

        # Annotate frame
        annotated_frame = frame_cropped.copy()
        current_detections = 0

        if results[0].obb is not None and results[0].obb.id is not None:
            boxes = results[0].obb.xyxyxyxy.cpu().numpy()
            track_ids = results[0].obb.id.cpu().numpy().astype(int)
            confidences = results[0].obb.conf.cpu().numpy()

            current_detections = len(track_ids)

            for box, track_id, conf in zip(boxes, track_ids, confidences):
                # Assign color to track
                if track_id not in colors:
                    colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())

                color = colors[track_id]

                # Draw oriented bounding box
                pts = box.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], True, color, line_width)

                # Get center point
                center = np.mean(box, axis=0).astype(int)

                # Update track history
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append((frame_count, center[0], center[1], timestamp))

                # Keep only last N points for trail
                if len(track_history[track_id]) > trail_length:
                    track_history[track_id] = track_history[track_id][-trail_length:]

                # Draw track trail (optional - controlled by show_trails)
                if show_trails:
                    points = np.array([(p[1], p[2]) for p in track_history[track_id]], dtype=np.int32)
                    if len(points) > 1:
                        cv2.polylines(annotated_frame, [points], False, color, line_width)

                # Draw ID label (optional - controlled by show_ids)
                if show_ids:
                    label = f"ID:{track_id}"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (center[0], center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        line_width,
                    )

        # Record detection count
        frame_detections.append((frame_count, timestamp, current_detections))

        # Draw current ant count only (no frame number)
        count_text = f"Ants: {current_detections}"
        text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.rectangle(
            annotated_frame,
            (10, 10),
            (20 + text_size[0], 40 + text_size[1]),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            annotated_frame,
            count_text,
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        # Save frame as image
        if frames_dir:
            frame_filename = frames_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_filename), annotated_frame)

        # Progress
        if frame_count % 30 == 0:
            print(
                f"Saved {frame_count}/{total_frames} frames "
                f"({frame_count / total_frames * 100:.1f}%)"
            )

    # Cleanup
    cap.release()

    print(f"\nTracking complete!")
    print(f"Total frames: {frame_count}")
    print(f"Unique tracks: {len(track_history)}")

    # Calculate statistics
    statistics = {}
    if enable_statistics:
        # Filter detections by time range
        stats_detections = [
            det for det in frame_detections
            if stats_start_frame <= det[0] <= stats_end_frame
        ]

        if stats_detections:
            counts = [det[2] for det in stats_detections]
            statistics = {
                'total_frames_analyzed': len(stats_detections),
                'time_range': (stats_start_time, stats_end_time if stats_end_time > 0 else frame_count / original_fps),
                'average_ants': np.mean(counts),
                'min_ants': np.min(counts),
                'max_ants': np.max(counts),
                'median_ants': np.median(counts),
                'std_ants': np.std(counts),
                'total_unique_tracks': len(track_history),
            }

            print(f"\n=== Statistics ({stats_start_time}s - {statistics['time_range'][1]:.1f}s) ===")
            print(f"Average ants detected: {statistics['average_ants']:.2f}")
            print(f"Min ants: {statistics['min_ants']}")
            print(f"Max ants: {statistics['max_ants']}")
            print(f"Median ants: {statistics['median_ants']:.2f}")
            print(f"Std deviation: {statistics['std_ants']:.2f}")
            print(f"Total unique tracks: {statistics['total_unique_tracks']}")

    # Automatically create video from frames
    if frames_dir and output_path:
        print("\n" + "=" * 70)
        print("Creating video from frames...")
        print("=" * 70)

        frames_to_video_ffmpeg(frames_dir, output_path, fps=output_fps)

        print(f"\nVideo created: {output_path}")
        print("=" * 70)

    return {
        'track_history': track_history,
        'statistics': statistics,
        'frame_detections': frame_detections,
        'output_path': output_path
    }
