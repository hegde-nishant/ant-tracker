#!/usr/bin/env python3
"""
Split labeled images into train/validation/test sets.

Splits dataset into:
- Training (80%): Model learns from these
- Validation (10%): Checks learning progress
- Test (10%): Final evaluation

Usage:
    python -m src.training.prepare_dataset --source /path/to/images --output ./dataset
"""

import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split


def prepare_yolo_dataset(data_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split images and labels into train/val/test sets."""
    random.seed(seed)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    print(f"Source directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Get all image files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"\nFound {len(image_files)} total images")

    # Filter to only images with non-empty labels
    valid_samples = []
    empty_samples = []

    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists() and os.path.getsize(label_path) > 0:
            valid_samples.append(img_path.stem)
        else:
            empty_samples.append(img_path.stem)

    print(f"Images with annotations: {len(valid_samples)}")
    print(f"Images without annotations: {len(empty_samples)}")

    # Split dataset
    # First split: separate test set
    train_val_samples, test_samples = train_test_split(
        valid_samples,
        test_size=test_ratio,
        random_state=seed
    )

    # Second split: separate train and val
    val_relative_size = val_ratio / (train_ratio + val_ratio)
    train_samples, val_samples = train_test_split(
        train_val_samples,
        test_size=val_relative_size,
        random_state=seed
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_samples)} ({len(train_samples)/len(valid_samples)*100:.1f}%)")
    print(f"  Val:   {len(val_samples)} ({len(val_samples)/len(valid_samples)*100:.1f}%)")
    print(f"  Test:  {len(test_samples)} ({len(test_samples)/len(valid_samples)*100:.1f}%)")

    # Create output directory structure
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }

    for split_name, samples in splits.items():
        # Create directories
        split_images_dir = output_dir / split_name / "images"
        split_labels_dir = output_dir / split_name / "labels"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nCopying {split_name} set...")

        # Copy files
        for sample_name in samples:
            # Copy image
            img_src = images_dir / f"{sample_name}.jpg"
            if not img_src.exists():
                img_src = images_dir / f"{sample_name}.png"
            img_dst = split_images_dir / img_src.name
            shutil.copy2(img_src, img_dst)

            # Copy label
            label_src = labels_dir / f"{sample_name}.txt"
            label_dst = split_labels_dir / f"{sample_name}.txt"
            shutil.copy2(label_src, label_dst)

        print(f"  ✓ Copied {len(samples)} image-label pairs to {split_name}/")

    # Copy classes.txt
    classes_src = data_dir / "classes.txt"
    classes_dst = output_dir / "classes.txt"
    shutil.copy2(classes_src, classes_dst)
    print(f"\n✓ Copied classes.txt")

    # Generate statistics
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    print(f"\nDataset statistics:")
    print(f"  Total annotated samples: {len(valid_samples)}")
    print(f"  Training samples:   {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    print(f"  Test samples:       {len(test_samples)}")
    print(f"\nOutput directory: {output_dir}")

    # Count total annotations
    for split_name in ['train', 'val', 'test']:
        label_dir = output_dir / split_name / "labels"
        total_annotations = 0
        for label_file in label_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                total_annotations += len(f.readlines())
        print(f"  {split_name.capitalize()} annotations: {total_annotations}")


def main():
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    output_dir = script_dir.parent / "dataset_split"

    # Prepare dataset with 80/10/10 split
    prepare_yolo_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )


if __name__ == "__main__":
    main()
