#!/usr/bin/env python3
"""
Track ants in video using trained YOLOv8-OBB model with BoT-SORT tracker.

BoT-SORT: Robust Associations Multi-Pedestrian Tracking
- Camera motion compensation
- Better handling of occlusions
- Improved re-identification

Usage:
    python track_ants_botsort.py --model path/to/best.pt --source video.mp4
"""

import argparse
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np


def track_video(
    model_path,
    source,
    output_dir="runs/track",
    conf_threshold=0.25,
    iou_threshold=0.7,
    imgsz=640,
    device=0,
    tracker="botsort.yaml",
    show=False,
    save_vid=True,
    save_txt=True,
    save_json=False,
    line_width=2,
    show_labels=True,
    show_conf=True,
):
    """
    Track ants in video using YOLOv8-OBB with BoT-SORT.

    Args:
        model_path: Path to trained YOLO model
        source: Path to video file or camera ID (0, 1, etc.)
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        imgsz: Image size for inference
        device: GPU device ID or 'cpu'
        tracker: Tracker config file (botsort.yaml or bytetrack.yaml)
        show: Display tracking results in real-time
        save_vid: Save tracked video
        save_txt: Save tracking results to txt files
        save_json: Save results in JSON format
        line_width: Bounding box line width
        show_labels: Show labels on bounding boxes
        show_conf: Show confidence scores
    """
    print("=" * 60)
    print("YOLOv8-OBB Ant Tracking with BoT-SORT")
    print("=" * 60)

    # Check environment
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print()

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {source}")
    print(f"Output directory: {output_dir}")
    print(f"Tracker: {tracker}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print()

    # Run tracking
    print("Starting tracking...")
    results = model.track(
        source=source,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        device=device,
        tracker=tracker,
        stream=True,
        show=show,
        save=save_vid,
        save_txt=save_txt,
        save_conf=True,
        line_width=line_width,
        show_labels=show_labels,
        show_conf=show_conf,
        project=str(output_dir),
        name="ant_tracking",
        exist_ok=False,
        verbose=True,
    )

    # Process and display results
    frame_count = 0
    track_ids = set()

    for result in results:
        frame_count += 1

        # Get tracking boxes
        if result.obb is not None and result.obb.id is not None:
            boxes = result.obb.xyxyxyxy  # OBB coordinates (4 corners)
            track_id_list = result.obb.id.cpu().numpy().astype(int)
            confidences = result.obb.conf.cpu().numpy()

            # Collect unique track IDs
            track_ids.update(track_id_list)

            # Print frame info
            if frame_count % 30 == 0:  # Print every 30 frames
                print(
                    f"Frame {frame_count}: {len(track_id_list)} ants tracked, "
                    f"Total unique IDs: {len(track_ids)}"
                )

    print("\n" + "=" * 60)
    print("Tracking Complete!")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}")
    print(f"Unique ant tracks: {len(track_ids)}")
    print(f"Results saved to: {output_dir}/ant_tracking")
    print("=" * 60)

    return results


def track_video_custom(
    model_path,
    source,
    output_path=None,
    conf_threshold=0.25,
    iou_threshold=0.7,
    imgsz=640,
    device=0,
    tracker="botsort.yaml",
):
    """
    Custom tracking with frame-by-frame processing and visualization.
    Useful for custom post-processing and analysis.

    Args:
        model_path: Path to trained YOLO model
        source: Path to video file
        output_path: Path to save output video
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        imgsz: Image size
        device: Device ID
        tracker: Tracker config
    """
    # Load model
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {source}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")

    # Setup video writer - try multiple codecs in order of preference
    if output_path:
        codecs_to_try = [
            ('avc1', 'H.264'),
            ('X264', 'x264'),
            ('XVID', 'XVID'),
            ('MJPG', 'Motion JPEG'),
            # mp4v removed - creates corrupted videos
        ]

        out = None
        for codec_code, codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_code)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
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

    # Tracking data
    track_history = {}  # {track_id: [(x, y), ...]}
    colors = {}  # {track_id: (B, G, R)}
    frame_count = 0

    print("Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run tracking
        results = model.track(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            tracker=tracker,
            persist=True,  # Persist tracks between frames
            verbose=False,
        )

        # Annotate frame
        annotated_frame = frame.copy()

        if results[0].obb is not None and results[0].obb.id is not None:
            boxes = results[0].obb.xyxyxyxy.cpu().numpy()  # OBB corners
            track_ids = results[0].obb.id.cpu().numpy().astype(int)
            confidences = results[0].obb.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, track_ids, confidences):
                # Assign color to track
                if track_id not in colors:
                    colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())

                color = colors[track_id]

                # Draw oriented bounding box
                pts = box.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], True, color, 2)

                # Get center point
                center = np.mean(box, axis=0).astype(int)

                # Update track history
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append(tuple(center))

                # Keep only last 30 points
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

                # Draw track trail (optional - commented out for cleaner display)
                # points = np.array(track_history[track_id], dtype=np.int32)
                # cv2.polylines(annotated_frame, [points], False, color, 2)

                # No label - just bounding box

            # Draw current ant count only (no frame number)
            current_count = len(track_ids)
            count_text = f"Ants: {current_count}"

            # Add background rectangle for better readability
            text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(
                annotated_frame,
                (10, 10),
                (20 + text_size[0], 40 + text_size[1]),
                (0, 0, 0),
                -1
            )

            # Draw count text
            cv2.putText(
                annotated_frame,
                count_text,
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

        # Write frame
        if output_path:
            out.write(annotated_frame)

        # Progress
        if frame_count % 30 == 0:
            print(
                f"Processed {frame_count}/{total_frames} frames "
                f"({frame_count / total_frames * 100:.1f}%)"
            )

    # Cleanup
    cap.release()
    if output_path:
        out.release()

    print(f"\nTracking complete!")
    print(f"Total frames: {frame_count}")
    print(f"Unique tracks: {len(track_history)}")
    if output_path:
        print(f"Output saved to: {output_path}")

    return track_history


def track_video_save_frames(
    model_path,
    source,
    output_dir,
    conf_threshold=0.25,
    iou_threshold=0.7,
    imgsz=640,
    device=0,
    tracker="botsort.yaml",
    max_frames=None,
):
    """
    Track video and save annotated frames to folder (more reliable than VideoWriter).
    Then use frames_to_video() to create the final video.

    Args:
        model_path: Path to trained YOLO model
        source: Path to video file
        output_dir: Directory to save frames
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        imgsz: Image size
        device: Device ID
        tracker: Tracker config
        max_frames: Maximum frames to process (None = all)

    Returns:
        dict: Contains track_history, frame_count, fps, dimensions
    """
    import os

    # Load model
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {source}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    print(f"Saving frames to: {output_dir}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tracking data
    track_history = {}  # {track_id: [(x, y), ...]}
    colors = {}  # {track_id: (B, G, R)}
    frame_count = 0

    print("Processing video and saving frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_count >= max_frames:
            break

        frame_count += 1

        # Run tracking
        results = model.track(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            tracker=tracker,
            persist=True,
            verbose=False,
        )

        # Annotate frame
        annotated_frame = frame.copy()

        if results[0].obb is not None and results[0].obb.id is not None:
            boxes = results[0].obb.xyxyxyxy.cpu().numpy()
            track_ids = results[0].obb.id.cpu().numpy().astype(int)
            confidences = results[0].obb.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, track_ids, confidences):
                # Assign color to track
                if track_id not in colors:
                    colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())

                color = colors[track_id]

                # Draw oriented bounding box
                pts = box.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], True, color, 2)

                # Get center point
                center = np.mean(box, axis=0).astype(int)

                # Update track history
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append(tuple(center))

                # Keep only last 30 points
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

            # Draw current ant count
            current_count = len(track_ids)
            count_text = f"Ants: {current_count}"
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
        frame_filename = output_dir / f"frame_{frame_count:06d}.jpg"
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
    print(f"Total frames saved: {frame_count}")
    print(f"Unique tracks: {len(track_history)}")
    print(f"Frames location: {output_dir}")

    return {
        'track_history': track_history,
        'frame_count': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
        'output_dir': str(output_dir)
    }


def frames_to_video_ffmpeg(frames_dir, output_video, fps=30, pattern="frame_%06d.jpg"):
    """
    Create video from frames using ffmpeg (more reliable than OpenCV VideoWriter).

    Args:
        frames_dir: Directory containing frames
        output_video: Output video path
        fps: Frames per second
        pattern: Frame filename pattern (default: frame_%06d.jpg)

    Example:
        frames_to_video_ffmpeg("runs/track/frames", "output.mp4", fps=30)
    """
    import subprocess

    frames_dir = Path(frames_dir)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

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

    print(f"Creating video from frames using ffmpeg...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"\nVideo created successfully: {output_video}")
        return str(output_video)
    except subprocess.CalledProcessError as e:
        print(f"Error creating video with ffmpeg:")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


def export_track_data(track_history, output_file):
    """
    Export tracking data to CSV for analysis.

    Args:
        track_history: Dictionary of {track_id: [(x, y), ...]}
        output_file: Path to output CSV file
    """
    import csv

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "frame", "x", "y"])

        for track_id, positions in track_history.items():
            for frame_idx, (x, y) in enumerate(positions):
                writer.writerow([track_id, frame_idx, x, y])

    print(f"Track data exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Track ants in video using YOLOv8-OBB with BoT-SORT"
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained YOLO model (.pt file)"
    )
    parser.add_argument(
        "--source", type=str, required=True, help="Path to video file or camera ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/track",
        help="Output directory for results",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size for inference"
    )
    parser.add_argument(
        "--device", type=str, default="0", help="Device (0, 1, ... or cpu)"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="botsort.yaml",
        choices=["botsort.yaml", "bytetrack.yaml"],
        help="Tracker configuration",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display results in real-time"
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save video")
    parser.add_argument(
        "--save-txt", action="store_true", help="Save tracking results to txt"
    )
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Use custom tracking with trail visualization",
    )
    parser.add_argument(
        "--export-csv", action="store_true", help="Export track data to CSV"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save frames to folder instead of video (more reliable, allows verification)",
    )
    parser.add_argument(
        "--create-video",
        action="store_true",
        help="Create video from saved frames using ffmpeg (use after --save-frames)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (for testing)",
    )

    args = parser.parse_args()

    # Convert device to int if numeric
    try:
        device = int(args.device)
    except ValueError:
        device = args.device

    # Handle --create-video mode (create video from existing frames)
    if args.create_video:
        frames_dir = Path(args.output_dir) / "frames"
        if not frames_dir.exists():
            print(f"Error: Frames directory not found: {frames_dir}")
            print("Run with --save-frames first to generate frames")
            return

        output_video = Path(args.output_dir) / "ant_tracking_from_frames.mp4"

        # Try to get fps from args or use default
        fps = 30  # Default FPS
        frames_to_video_ffmpeg(frames_dir, output_video, fps=fps)
        return

    # Handle --save-frames mode (save frames instead of video)
    if args.save_frames:
        frames_dir = Path(args.output_dir) / "frames"

        result = track_video_save_frames(
            model_path=args.model,
            source=args.source,
            output_dir=str(frames_dir),
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
            device=device,
            tracker=args.tracker,
            max_frames=args.max_frames,
        )

        # Export to CSV if requested
        if args.export_csv:
            csv_path = Path(args.output_dir) / "track_data.csv"
            export_track_data(result['track_history'], csv_path)

        print("\n" + "=" * 60)
        print("Next step: Create video from frames using:")
        print(f"python track_ants_botsort.py --create-video --output-dir {args.output_dir}")
        print("=" * 60)
        return

    if args.custom:
        # Custom tracking with visualization
        output_path = Path(args.output_dir) / "ant_tracking_custom.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        track_history = track_video_custom(
            model_path=args.model,
            source=args.source,
            output_path=str(output_path) if not args.no_save else None,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
            device=device,
            tracker=args.tracker,
        )

        # Export to CSV if requested
        if args.export_csv:
            csv_path = Path(args.output_dir) / "track_data.csv"
            export_track_data(track_history, csv_path)

    else:
        # Standard YOLO tracking
        track_video(
            model_path=args.model,
            source=args.source,
            output_dir=args.output_dir,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
            device=device,
            tracker=args.tracker,
            show=args.show,
            save_vid=not args.no_save,
            save_txt=args.save_txt,
        )

if __name__ == "__main__":
    main()
