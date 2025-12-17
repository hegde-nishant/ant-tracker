# Ant Tracker

This tool helps you automatically find and track individual ants in videos. The software learns to recognize ants and follows each one throughout your video, giving each ant a unique ID number.

## What Can This Do?

- Automatically detect ants in videos
- Track individual ants over time (each ant gets a unique ID)
- Count how many ants are present at any time
- Analyze ant movement patterns and behavior
- Export data to spreadsheets (Excel, CSV) for further analysis

## What Do I Need?

**Essential:**
- Python 3.10 or 3.11
- 8GB+ RAM
- 5GB free disk space

**Optional (But Recommended):**
- NVIDIA graphics card (GPU) for faster processing
  - With GPU: 10-minute video → 3-10 minutes processing
  - Without GPU: 10-minute video → 30-60 minutes processing

---

## Quick Start (5 Minutes)

### Step 1: Install Python

Download and install **Python 3.11** from [python.org/downloads](https://www.python.org/downloads/)

**Important for macOS users**: If using Miniconda/Anaconda, install tkinter:
```bash
conda install -c conda-forge tk
```

### Step 2: Install Ant Tracker

Open terminal/command prompt and navigate to the ant-tracker folder:

```bash
cd /path/to/ant-tracker
```

Create virtual environment and install. Choose ONE option based on your hardware:

**Windows (CPU only - no NVIDIA GPU):**
```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
```

**Windows (with NVIDIA GPU):**
```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-gpu.txt
```

**macOS/Linux (CPU only - no NVIDIA GPU):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
```

**macOS/Linux (with NVIDIA GPU):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-gpu.txt
```

### Step 3: Run the Tracker

```bash
python -m src.gui.ant_tracker_gui
```

In the GUI:
1. Click "Browse" next to Model Path → select `models/trained/best.pt`
2. Click "Browse" next to Video Path → select your ant video
3. Click "Browse" next to Output Directory → choose where to save results
4. Click "Start Tracking"

Done! Your tracked video and data will be in the output directory.

---

## Detailed Installation

### Windows

**1. Install Python 3.11**
- Go to [python.org/downloads](https://www.python.org/downloads/)
- Download Python 3.11
- Run installer and **check "Add Python to PATH"**
- Verify: Open Command Prompt, type `python --version`

**2. Install Ant Tracker**

Choose ONE option based on your hardware:

**With NVIDIA GPU:**
```bash
cd C:\path\to\ant-tracker
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-gpu.txt
```

**Without GPU (or non-NVIDIA):**
```bash
cd C:\path\to\ant-tracker
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
```

**3. Install FFmpeg** (for video processing)
- Download from [ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- Extract to `C:\ffmpeg`
- Add `C:\ffmpeg\bin` to system PATH

**4. Test Installation**
```bash
python -m src.gui.ant_tracker_gui
```

### macOS

**1. Install Python 3.11**

**Using official installer (recommended):**
- Download from [python.org/downloads](https://www.python.org/downloads/)
- Install the .pkg file
- Verify: `python3 --version`

**Using Homebrew:**
```bash
brew install python@3.11
```

**2. Install Ant Tracker**

Choose ONE option based on your hardware:

**M1/M2 Macs or without GPU:**
```bash
cd ~/Desktop/ant-tracker  # or your path
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
```

**Intel Macs with NVIDIA GPU (rare):**
```bash
cd ~/Desktop/ant-tracker  # or your path
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-gpu.txt
```

**3. Install FFmpeg**
```bash
brew install ffmpeg
```

**4. Test Installation**
```bash
python -m src.gui.ant_tracker_gui
```

### Linux (Ubuntu/Debian)

**1. Install Python 3.11**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip python3-tk
```

**2. Install NVIDIA Drivers** (if you have NVIDIA GPU)
```bash
# Check for NVIDIA GPU
lspci | grep -i nvidia

# Install drivers
sudo apt install nvidia-driver-535
sudo reboot

# Verify
nvidia-smi
```

**3. Install Ant Tracker**

Choose ONE option based on your hardware:

**With NVIDIA GPU:**
```bash
cd ~/Desktop/ant-tracker  # or your path
python3.11 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-gpu.txt
```

**CPU only:**
```bash
cd ~/Desktop/ant-tracker  # or your path
python3.11 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
```

**4. Install FFmpeg**
```bash
sudo apt install ffmpeg
```

**5. Test Installation**
```bash
python -m src.gui.ant_tracker_gui
```

---

## Using the Tracker

### GUI Method (Recommended)

1. **Activate environment and start GUI:**
   ```bash
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   python -m src.gui.ant_tracker_gui
   ```

2. **Configure tracking:**
   - **Model**: Select `models/trained/best.pt` (included)
   - **Video**: Select your ant video file
   - **Output**: Choose where to save results
   - **Settings**: Hover over any option to see what it does

3. **Common adjustments:**
   - Lower "Confidence Threshold" (try 0.15) if ants are being missed
   - Enable "Image Enhancement" if ants are small or video quality is poor
   - Enable "ROI Cropping" to focus on specific area (speeds up processing)

4. **Click "Start Tracking"** and wait for completion

---

## Understanding Your Results

After tracking completes, you'll find:

### 1. Tracked Video (`video_tracked.mp4`)
- Your original video with colored boxes around each ant
- Each ant has a unique ID number
- Optional: Movement trails showing where ants traveled

### 2. Tracking Data (`video_track_data.csv`)
Open in Excel or Google Sheets:

| Column | Description | Example |
|--------|-------------|---------|
| frame | Frame number | 150 |
| track_id | Ant's unique ID | 5 |
| x | Horizontal position (pixels) | 423.5 |
| y | Vertical position (pixels) | 287.2 |
| timestamp_sec | Time in video (seconds) | 5.0 |

**Use this data to:**
- Count ants per frame
- Track individual ant movements
- Calculate speeds and distances
- Create heatmaps or trajectory plots

### 3. Statistics (if enabled)
- Average number of ants during time period
- Minimum and maximum ants seen
- Useful for behavioral experiments

---

## Training Your Own Model

The included model works for many ant species, but you'll need to train your own if:
- Different ant species
- Very different camera setup
- Need higher accuracy for your specific videos

### Step 1: Create Dataset

**Extract frames from your videos:**

The detector (Yolo) is trained on frames, hence we have to extract frames that we can label from our videos.

```bash
python utils/extract_frames.py --video your_video.mp4 --output dataset/images --interval 30
```

**Annotate frames:**
1. Go to [labelstud.io](https://labelstud.io) and install Label Studio (scope of how to work with Label Studio is out of scope, please refer Label Studio documentation)
2. Upload your extracted frames
3. Draw bounding boxes around each ant
4. Export in YOLO OBB format

**Split into train/val/test:**

We need to split our annotated data into different folders, train - model sees and learns from this data, val - model is evaluated on this data during training and refined if performance is poor, test - unseen data we can test the model on to check it's performance.

```bash
python -m src.training.prepare_dataset --source unprepared_dataset --output dataset
```

### Step 2: Train Model

```bash
python -m src.training.train_yolov8_obb
```

**Training time:**
- GPU: 30 minutes to 4 hours
- CPU: Several hours to days (not recommended)

The best model is saved at `models/trained/best.pt`

### Step 3: Use Your Model

Use it in the GUI or command line just like the default model.

---

## Common Problems and Solutions

### "No module named 'tkinter'"

**macOS with Miniconda/Anaconda:**
```bash
conda install -c conda-forge tk
```

**Linux:**
```bash
sudo apt install python3-tk
```

**Windows:** Reinstall Python and check "tcl/tk" option

### "No ants detected" in video

**Try these in order:**
1. Lower confidence threshold to 0.15 or 0.10
2. Enable image enhancement (Combined mode)
3. Check video format is MP4, AVI, or MOV
4. Verify model file exists at `models/trained/best.pt`

### Processing is very slow

**This is normal without GPU.** Expected speeds:
- CPU: 10-min video → 30-90 min processing
- GPU: 10-min video → 3-10 min processing

**To speed up CPU processing:**
- Test on shorter clips first
- Disable image enhancement
- Use ROI to crop to smaller area

### Ant IDs keep changing

**Solutions:**
1. Increase Track Length to 50-60 frames (in Advanced Options)
2. Lower confidence threshold to 0.15
3. Use BoT-SORT tracker (recommended over ByteTrack)

### "CUDA out of memory" or "device=0 not found"

You're trying to use GPU but:
- Don't have NVIDIA GPU, or
- Installed CPU version of PyTorch

**Solution:** Use CPU version:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### GUI won't start

**Check virtual environment is activated:**
```bash
# You should see (venv) at start of terminal line
# If not, activate it:
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

**Reinstall packages:**
```bash
pip install -r requirements-cpu.txt
```

### "Could not import tracking module"

**Run as module from project root:**
```bash
cd /path/to/ant-tracker
python -m src.gui.ant_tracker_gui
```

Not: `python src/gui/ant_tracker_gui.py`

---

## Using Utility Scripts

### Split Long Videos

Split a long video into smaller chunks for annotation:

```bash
python utils/split_video.py --video long_video.mp4 --chunks 20 --output chunks/
```

### Extract Frames for Dataset

Extract frames at regular intervals:

```bash
python utils/extract_frames.py --video video.mp4 --output frames/ --interval 30
```

Adjust `--interval`:
- Fast-moving ants: 10-15 frames
- Slow-moving ants: 30-60 frames

---

## Advanced Features

### Region of Interest (ROI)

Track only a specific area of the video:

**In GUI:**
1. Check "Enable ROI Cropping"
2. Click "Select ROI"
3. Draw rectangle on first frame

**Command line:**
```bash
--roi "100,100,800,600"  # x1,y1,x2,y2
```

### Image Enhancement

Improve detection on poor quality videos:

**In GUI:**
1. Check "Enable Image Enhancement"
2. Select "Combined" (recommended)

**Warning:** Adds 50-70% processing time

### Statistics for Time Ranges

Calculate ant counts for specific time period:

**In GUI:**
1. Check "Enable Custom Statistics"
2. Set start and end times (in seconds)

Results show average/min/max ant counts

---

## Project Structure

```
ant-tracker/
├── README.md                   # This file - start here
├── ADVANCED.md                 # Advanced configurations
│
├── src/
│   ├── gui/                    # Graphical interface
│   ├── tracking/               # Video tracking code
│   ├── training/               # Model training code
│   └── preprocessing/          # Image enhancement
│
├── models/
│   ├── pretrained/             # Base models
│   └── trained/
│       └── best.pt             # Default model (use this)
│
├── test_videos/                # folder that can be used to store test videos
│
├── utils/                      # Utility scripts
│   ├── split_video.py          # Split long videos
│   └── extract_frames.py       # Extract frames
│
└── outputs/                    # Your results
    ├── training/               # Training results
    └── tracking/               # Tracked videos
```

---
