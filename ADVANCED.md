# Advanced Usage Guide

This guide covers advanced features for users who need custom configurations, HPC deployments, or specialized training setups.

## Table of Contents

- [HPC/Cluster Setup](#hpccluster-setup)
- [Advanced Training Configuration](#advanced-training-configuration)
- [GPU Configuration and Optimization](#gpu-configuration-and-optimization)
- [Model Export and Deployment](#model-export-and-deployment)
- [Advanced Tracking Features](#advanced-tracking-features)
- [Batch Processing](#batch-processing)
- [Performance Optimization](#performance-optimization)
- [Understanding Oriented Bounding Boxes](#understanding-oriented-bounding-boxes)

---

## HPC/Cluster Setup

For high-performance computing environments with module loading systems.

### Environment Setup

```bash
# Load modules (adjust for your HPC environment)
module load gcc/11.2.0
module load cuda/11.8
module load python/3.10

# Run HPC setup script
bash setup_environment.sh

# Activate environment
source yolo_env/bin/activate
```

### Batch Job Script Example

Example SLURM script for training on cluster:

```bash
#!/bin/bash
#SBATCH --job-name=ant_yolo_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=train_%j.log

# Load modules
module load cuda/11.8
module load python/3.10

# Activate environment
source yolo_env/bin/activate

# Navigate to project
cd $SLURM_SUBMIT_DIR

# Run training
python -m src.training.train_yolov8_obb

# Deactivate
deactivate
```

### Multi-GPU Training

```python
# In src/training/train_yolov8_obb.py
CONFIG = {
    "device": "0,1,2,3",  # Use 4 GPUs
    "batch_size": 256,    # Scale batch size with GPU count
    ...
}
```

---

## Advanced Training Configuration

### Complete CONFIG Options

Edit `src/training/train_yolov8_obb.py`:

```python
CONFIG = {
    # Model configuration
    "model_size": "n",          # Model size: n/s/m/l/x
    "epochs": 150,              # Number of training epochs
    "batch_size": 64,           # Batch size (-1 for auto-batch)
    "imgsz": 1280,              # Input image size

    # Hardware
    "device": 0,                # GPU device ID, "cpu", or "0,1,2,3" for multi-GPU
    "workers": 8,               # Dataloader workers (match CPU cores)
    "cache": False,             # Cache images: False, True, 'ram', 'disk'

    # Optimization
    "optimizer": "AdamW",       # Optimizer: SGD, Adam, AdamW, etc.
    "lr0": 0.01,                # Initial learning rate
    "lrf": 0.01,                # Final learning rate (fraction of lr0)
    "momentum": 0.937,          # SGD momentum/Adam beta1
    "weight_decay": 0.0005,     # Optimizer weight decay

    # Learning rate schedule
    "warmup_epochs": 3.0,       # Warmup epochs
    "warmup_momentum": 0.8,     # Warmup initial momentum
    "warmup_bias_lr": 0.1,      # Warmup initial bias lr

    # Augmentation
    "hsv_h": 0.015,             # Hue augmentation
    "hsv_s": 0.7,               # Saturation augmentation
    "hsv_v": 0.4,               # Value augmentation
    "degrees": 0.0,             # Rotation (+/- degrees)
    "translate": 0.1,           # Translation (+/- fraction)
    "scale": 0.5,               # Scale (+/- gain)
    "shear": 0.0,               # Shear (+/- degrees)
    "perspective": 0.0,         # Perspective (+/- fraction)
    "flipud": 0.0,              # Vertical flip probability
    "fliplr": 0.5,              # Horizontal flip probability
    "mosaic": 1.0,              # Mosaic augmentation probability
    "mixup": 0.0,               # Mixup augmentation probability
    "copy_paste": 0.0,          # Copy-paste augmentation probability

    # Regularization
    "dropout": 0.0,             # Dropout rate
    "label_smoothing": 0.0,     # Label smoothing epsilon

    # Saving and validation
    "save_period": 10,          # Save checkpoint every N epochs
    "patience": 50,             # Early stopping patience
    "val": True,                # Validate during training
    "plots": True,              # Generate plots

    # Mixed precision
    "amp": True,                # Automatic Mixed Precision

    # Resume and pretrained
    "resume": False,            # Resume from last checkpoint
    "pretrained": True,         # Use pretrained weights
}
```

### Recommended Settings by GPU

**Consumer GPUs (6-8GB VRAM)**:
```python
CONFIG = {
    "model_size": "n",
    "batch_size": 8,
    "imgsz": 640,
    "epochs": 100,
    "workers": 4,
}
```

**Mid-Range GPUs (12GB VRAM)**:
```python
CONFIG = {
    "model_size": "s",
    "batch_size": 16,
    "imgsz": 1024,
    "epochs": 150,
    "workers": 6,
}
```

**High-End GPUs (24GB VRAM)**:
```python
CONFIG = {
    "model_size": "m",
    "batch_size": 32,
    "imgsz": 1280,
    "epochs": 150,
    "workers": 8,
}
```

**HPC GPUs (40-80GB VRAM)**:
```python
CONFIG = {
    "model_size": "l",
    "batch_size": 64,
    "imgsz": 1280,
    "epochs": 200,
    "workers": 12,
}
```

**CPU-Only** (for testing only):
```python
CONFIG = {
    "model_size": "n",
    "batch_size": 2,
    "imgsz": 480,
    "epochs": 20,
    "device": "cpu",
    "workers": 2,
}
```

### Custom Augmentation Strategy

For specialized datasets:

```python
CONFIG = {
    # Conservative augmentation (when images are already varied)
    "mosaic": 0.5,
    "degrees": 5.0,
    "translate": 0.05,
    "scale": 0.2,
    "fliplr": 0.5,

    # Aggressive augmentation (for limited data)
    "mosaic": 1.0,
    "degrees": 15.0,
    "translate": 0.2,
    "scale": 0.9,
    "fliplr": 0.5,
    "mixup": 0.1,
}
```

### Monitoring Training Progress

**TensorBoard**:
```bash
tensorboard --logdir outputs/training
```

**Test Intermediate Checkpoints**:
```bash
# Test latest checkpoint
python -m src.evaluation.test_intermediate_models \
    --video test_videos/av1.mp4 \
    --latest

# Test all checkpoints
python -m src.evaluation.test_intermediate_models \
    --video test_videos/av1.mp4 \
    --all

# Test specific epoch
python -m src.evaluation.test_intermediate_models \
    --video test_videos/av1.mp4 \
    --epoch 50

# Quick test (first 300 frames)
python -m src.evaluation.test_intermediate_models \
    --video test_videos/av1.mp4 \
    --latest \
    --max-frames 300
```

### Resume Training

If training is interrupted:

```python
CONFIG = {
    'resume': True,  # Resume from last checkpoint
    ...
}
```

---

## GPU Configuration and Optimization

### CUDA Memory Management

For large models or high-resolution training:

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

### Memory Troubleshooting

**Out of Memory Solutions**:
1. Reduce batch size:
   ```python
   CONFIG["batch_size"] = 32  # or 16, 8, 4
   ```
2. Reduce image size:
   ```python
   CONFIG["imgsz"] = 640  # or 512, 480
   ```
3. Use smaller model:
   ```python
   CONFIG["model_size"] = "n"
   ```
4. Enable gradient checkpointing (advanced)
5. Close other GPU applications

### Mixed Precision Training

Enabled by default for performance:

```python
CONFIG["amp"] = True  # Automatic Mixed Precision
```

Disabling AMP (if compatibility issues):
```python
CONFIG["amp"] = False
```

### Dataset Caching

Speed up training by caching images in RAM:

```python
CONFIG["cache"] = "ram"  # Requires sufficient RAM (dataset size * 2)
```

Cache to disk (slower than RAM, faster than no cache):
```python
CONFIG["cache"] = "disk"
```

---

## Model Export and Deployment

### Export to Different Formats

```python
from ultralytics import YOLO

model = YOLO('models/trained/best.pt')

# ONNX format (cross-platform)
model.export(format='onnx')

# TensorRT (NVIDIA GPUs, fastest)
model.export(format='engine')

# TorchScript (PyTorch deployment)
model.export(format='torchscript')

# OpenVINO (Intel hardware)
model.export(format='openvino')

# CoreML (Apple devices)
model.export(format='coreml')
```

Supported formats:
- `onnx` - Open Neural Network Exchange (cross-platform)
- `engine` - TensorRT (NVIDIA GPUs, ~2-5x faster)
- `torchscript` - PyTorch TorchScript
- `openvino` - Intel OpenVINO toolkit
- `coreml` - Apple CoreML (iOS/macOS)
- `tflite` - TensorFlow Lite (mobile/edge)

### Validate Model Performance

```python
from ultralytics import YOLO

model = YOLO('models/trained/best.pt')
results = model.val(data='config/dataset.yaml', split='test')

print(f'mAP50: {results.box.map50:.4f}')
print(f'mAP50-95: {results.box.map:.4f}')
print(f'Precision: {results.box.p:.4f}')
print(f'Recall: {results.box.r:.4f}')
```

### Using Exported Models

**ONNX Runtime**:
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('best.onnx')
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: image_array})
```

**TensorRT**:
```python
from ultralytics import YOLO

model = YOLO('best.engine')
results = model('image.jpg')
```

---

## Advanced Tracking Features

### Custom Tracker Configuration

Create custom tracker YAML:

```yaml
# custom_botsort.yaml
tracker_type: botsort
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 30
match_thresh: 0.8
proximity_thresh: 0.5
appearance_thresh: 0.25
cmc_method: sparseOptFlow
frame_rate: 30
```

Use custom config:
```bash
python -m src.tracking.track_enhanced \
    --model models/trained/best.pt \
    --source video.mp4 \
    --tracker-config custom_botsort.yaml
```

### Advanced ROI Usage

**Multiple ROIs**:
Process different regions separately:

```bash
# Process region 1
python -m src.tracking.track_enhanced \
    --model models/trained/best.pt \
    --source video.mp4 \
    --roi "100,100,400,400" \
    --output-dir outputs/region1

# Process region 2
python -m src.tracking.track_enhanced \
    --model models/trained/best.pt \
    --source video.mp4 \
    --roi "500,100,800,400" \
    --output-dir outputs/region2
```

### Statistical Analysis

Advanced statistics with custom parameters:

```bash
python -m src.tracking.track_enhanced \
    --model models/trained/best.pt \
    --source video.mp4 \
    --enable-statistics \
    --stats-start 10 \
    --stats-end 60 \
    --export-csv \
    --export-json
```

Analyze CSV output in Python:

```python
import pandas as pd

# Load tracking data
df = pd.read_csv('outputs/tracking/video_track_data.csv')

# Calculate statistics
print(f"Total ants tracked: {df['track_id'].nunique()}")
print(f"Average ants per frame: {df.groupby('frame')['track_id'].count().mean():.2f}")
print(f"Max ants in single frame: {df.groupby('frame')['track_id'].count().max()}")

# Plot ant counts over time
import matplotlib.pyplot as plt

ant_counts = df.groupby('frame')['track_id'].count()
plt.plot(ant_counts)
plt.xlabel('Frame')
plt.ylabel('Ant Count')
plt.title('Ant Counts Over Time')
plt.savefig('ant_counts.png')
```

---

## Batch Processing

### Process Multiple Videos

Create a batch processing script:

```python
# batch_track.py
import os
from pathlib import Path

videos = Path('test_videos').glob('*.mp4')
model = 'models/trained/best.pt'

for video in videos:
    output_dir = f'outputs/tracking/{video.stem}'
    cmd = f"""
    python -m src.tracking.track_enhanced \
        --model {model} \
        --source {video} \
        --output-dir {output_dir} \
        --export-csv
    """
    os.system(cmd)
```

### Parallel Processing

Use GNU Parallel for faster batch processing:

```bash
# Install GNU Parallel
sudo apt install parallel  # Linux
brew install parallel      # macOS

# Process videos in parallel
find test_videos -name "*.mp4" | parallel -j 4 \
    python -m src.tracking.track_enhanced \
        --model models/trained/best.pt \
        --source {} \
        --output-dir outputs/tracking/{/.}
```

---

## Performance Optimization

### Training Performance

1. **Increase dataloader workers**:
   ```python
   CONFIG["workers"] = 8  # Match CPU cores
   ```

2. **Enable dataset caching**:
   ```python
   CONFIG["cache"] = "ram"  # If enough RAM
   ```

3. **Use mixed precision** (enabled by default):
   ```python
   CONFIG["amp"] = True
   ```

4. **Optimize batch size**:
   ```python
   CONFIG["batch_size"] = -1  # Auto-batch
   ```

### Tracking Performance

1. **Lower processing resolution**:
   ```bash
   --imgsz 640  # Instead of 1280
   ```

2. **Use ByteTrack** (faster than BoT-SORT):
   ```bash
   --tracker bytetrack
   ```

3. **Reduce trail length**:
   ```bash
   --trail-length 10  # Instead of 30
   ```

4. **Disable visualization** (process only):
   ```bash
   --no-display --save-frames
   ```

### Memory Optimization

**Reduce VRAM usage**:
```python
CONFIG = {
    "batch_size": 8,     # Smaller batches
    "imgsz": 640,        # Lower resolution
    "workers": 4,        # Fewer workers
    "cache": False,      # No caching
}
```

**Reduce RAM usage**:
- Disable dataset caching
- Reduce dataloader workers
- Process videos in chunks

---

## Understanding Oriented Bounding Boxes

### Why OBB for Ants?

Traditional axis-aligned boxes (AABB) waste space for elongated objects like ants. Oriented Bounding Boxes (OBB) rotate to fit the object shape, providing:

1. **Better fit**: Reduces background inclusion
2. **More accurate**: Clearer separation between close ants
3. **Orientation info**: Know which direction ant is facing

### OBB Format

YOLO OBB format (in label files):
```
class_id x_center y_center width height rotation
```

Example:
```
0 0.5 0.5 0.1 0.05 0.785
```
- Class 0 (ant)
- Center at (0.5, 0.5) in normalized coordinates
- Width 0.1, height 0.05
- Rotation 0.785 radians (45 degrees)

### Visualizing OBB

OBB annotations appear as rotated rectangles that follow ant body orientation, unlike regular boxes which are always horizontal/vertical.

---

## Custom Augmentation Strategies

### Domain-Specific Augmentation

For ants in specific environments:

**Laboratory arenas** (controlled lighting):
```python
CONFIG = {
    "hsv_h": 0.01,      # Minimal color variation
    "hsv_s": 0.3,       # Moderate saturation
    "hsv_v": 0.2,       # Moderate brightness
    "mosaic": 0.5,      # Some mosaic
    "fliplr": 0.5,      # Horizontal flip OK
}
```

**Outdoor** (variable lighting):
```python
CONFIG = {
    "hsv_h": 0.02,      # More hue variation
    "hsv_s": 0.7,       # High saturation variation
    "hsv_v": 0.5,       # High brightness variation
    "mosaic": 1.0,      # Full mosaic
    "fliplr": 0.5,      # Horizontal flip
}
```

**Multi-scale** (varying distances):
```python
CONFIG = {
    "scale": 0.9,       # Large scale variation
    "degrees": 10.0,    # Some rotation
    "translate": 0.2,   # Translation variation
    "mosaic": 1.0,      # Full mosaic
}
```

---

## Troubleshooting Advanced Issues

### Training Divergence

If loss increases or becomes NaN:

1. Lower learning rate:
   ```python
   CONFIG["lr0"] = 0.001  # Instead of 0.01
   ```

2. Increase warmup:
   ```python
   CONFIG["warmup_epochs"] = 5.0
   ```

3. Reduce augmentation:
   ```python
   CONFIG["mosaic"] = 0.5
   CONFIG["mixup"] = 0.0
   ```

### Overfitting

If validation loss increases while training loss decreases:

1. Add regularization:
   ```python
   CONFIG["weight_decay"] = 0.001
   CONFIG["dropout"] = 0.1
   CONFIG["label_smoothing"] = 0.1
   ```

2. More augmentation:
   ```python
   CONFIG["mosaic"] = 1.0
   CONFIG["mixup"] = 0.1
   ```

3. Early stopping:
   ```python
   CONFIG["patience"] = 30  # Stop after 30 epochs no improvement
   ```

### Poor Generalization

Model works on training data but not new videos:

1. Ensure diverse training data
2. Check for dataset leakage
3. Validate on truly held-out test set
4. Try larger model
5. Review augmentation strategy

---

## Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [BoT-SORT Paper](https://arxiv.org/abs/2206.14651)
- [YOLO OBB Guide](https://docs.ultralytics.com/tasks/obb/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
