#!/usr/bin/env python3
"""
Train a YOLOv8 model to detect ants in images.

Uses oriented bounding boxes (rotated rectangles) for better ant detection.
Requires labeled training images.

Usage:
    python -m src.training.train_yolov8_obb

Training time: 20 minutes to 4 hours depending on dataset size and GPU.
"""

import gc
import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# Set PyTorch memory management for better CUDA allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def check_environment(device=0):
    """Print system and CUDA information."""
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        # IMPORTANT: Set device FIRST before any operations
        torch.cuda.set_device(device)
        print(f"✓ Set default CUDA device to: {device}")

        print(f"Target GPU {device}: {torch.cuda.get_device_name(device)}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB"
        )

        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()

        # Show memory status
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        # Verify current device
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    print(f"CPU cores: {os.cpu_count()}")
    print("=" * 60 + "\n")


def train_yolov8_obb(
    model_size="n",
    data_yaml="dataset.yaml",
    epochs=150,
    batch_size=8,
    imgsz=640,
    device=0,
    project="runs/obb",
    name="ant_yolov8n_obb",
    resume=False,
    pretrained=True,
    cache=False,
    workers=2,
    patience=50,
    save_period=10,
):
    """
    Train YOLOv8 OBB model.

    Args:
        model_size: Model size - 'n', 's', 'm', 'l', 'x' (nano to extra-large)
        data_yaml: Path to dataset YAML configuration
        epochs: Number of training epochs
        batch_size: Batch size (-1 for auto-batch)
        imgsz: Input image size
        device: GPU device ID (0) or 'cpu'
        project: Project directory for results
        name: Experiment name
        resume: Resume from last checkpoint
        pretrained: Use pretrained weights
        cache: Cache images for faster training (True/False/'ram'/'disk')
        workers: Number of dataloader workers
        patience: Early stopping patience (epochs)
        save_period: Save checkpoint every N epochs
    """
    check_environment(device)

    # Load model
    model_name = (
        f"yolov8{model_size}-obb.pt" if pretrained else f"yolov8{model_size}-obb.yaml"
    )
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    print(f"\nStarting training with configuration:")
    print(f"  Model: YOLOv8{model_size}-OBB")
    print(f"  Dataset: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {imgsz}")
    print(f"  Device: {device}")
    print(f"  Workers: {workers}")
    print(f"  Cache: {cache}")
    print(f"  Patience: {patience}")
    print()

    # Training arguments optimized for A100
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        # Optimizer settings
        optimizer="AdamW",  # AdamW for better convergence
        lr0=0.01,  # Initial learning rate
        lrf=0.01,  # Final learning rate (lr0 * lrf)
        momentum=0.937,  # SGD momentum/Adam beta1
        weight_decay=0.0005,  # Optimizer weight decay
        # Augmentation (standard)
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,  # HSV-Saturation augmentation
        hsv_v=0.4,  # HSV-Value augmentation
        degrees=0.0,  # Rotation (+/- deg)
        translate=0.1,  # Translation (+/- fraction)
        scale=0.5,  # Scaling (+/- gain)
        shear=0.0,  # Shear (+/- deg)
        perspective=0.0,  # Perspective (+/- fraction)
        flipud=0.0,  # Vertical flip probability
        fliplr=0.5,  # Horizontal flip probability
        mosaic=1.0,  # Mosaic augmentation probability
        mixup=0.0,  # Mixup augmentation probability
        copy_paste=0.0,  # Copy-paste augmentation probability
        # Training settings
        resume=resume,
        pretrained=pretrained,
        cache=cache,
        workers=workers,
        patience=patience,
        save=True,
        save_period=save_period,
        # Validation
        val=True,
        plots=True,
        # Performance
        amp=True,  # Automatic Mixed Precision
        fraction=1.0,  # Use full dataset
        # Other
        verbose=True,
        seed=0,
        deterministic=False,  # Faster but non-deterministic
        single_cls=False,
        rect=False,  # Rectangular training
        cos_lr=True,  # Cosine learning rate scheduler
        close_mosaic=10,  # Disable mosaic in last N epochs
        # Callbacks
        save_json=True,
        save_hybrid=False,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {project}/{name}")
    print(f"Best model: {project}/{name}/weights/best.pt")
    print(f"Last model: {project}/{name}/weights/last.pt")

    return results


def validate_model(model_path, data_yaml, imgsz=640, device=0):
    """
    Validate trained model on test set.

    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset YAML
        imgsz: Image size for validation
        device: GPU device ID
    """
    print("\n" + "=" * 60)
    print("Validating Model on Test Set")
    print("=" * 60)

    model = YOLO(model_path)

    # Update YAML to use test set
    with open(data_yaml, "r") as f:
        data_config = yaml.safe_load(f)

    # Validate on test set
    results = model.val(
        data=data_yaml,
        split="test",
        imgsz=imgsz,
        device=device,
        plots=True,
        save_json=True,
    )

    print(f"\nTest Set Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")

    return results


def main():
    """Main training pipeline."""

    # Training configuration
    CONFIG = {
        "model_size": "n",  # Model size: n, s, m, l, x (larger = more accurate but slower)
        "data_yaml": "dataset.yaml",  # Path to dataset configuration
        "epochs": 150,  # Number of training iterations
        "batch_size": 64,  # Images per batch (reduce if out of memory)
        "imgsz": 1280,  # Training image size in pixels
        "device": 0,  # 0 = GPU, 'cpu' = CPU
        "project": "runs/obb",  # Output directory
        "name": "ant_yolov8n_obb",  # Training run name
        "resume": False,  # Resume from checkpoint
        "cache": False,  # Cache images in RAM (faster but uses more memory)
        "workers": 2,  # Parallel data loading threads
        "patience": 50,  # Stop if no improvement for N epochs
        "save_period": 10,  # Save checkpoint every N epochs
    }

    # Train model
    results = train_yolov8_obb(**CONFIG)

    # Validate on test set
    best_model = f"{CONFIG['project']}/{CONFIG['name']}/weights/best.pt"
    if os.path.exists(best_model):
        validate_model(
            model_path=best_model,
            data_yaml=CONFIG["data_yaml"],
            imgsz=CONFIG["imgsz"],
            device=CONFIG["device"],
        )

if __name__ == "__main__":
    main()
