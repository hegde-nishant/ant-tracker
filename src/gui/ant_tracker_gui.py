#!/usr/bin/env python3
"""
Ant Detection & Tracking System - GUI Application
A simple Tkinter-based interface for running ant tracking with YOLOv8-OBB models.
"""

import os
import platform
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import Canvas, Toplevel, filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# Import tracking functions from enhanced script (all-in-one solution)
try:
    from src.tracking.track_enhanced import (
        frames_to_video_ffmpeg,
        track_video_with_roi_and_stats,
    )
except ImportError as e:
    print(f"Error: Could not import tracking module: {e}")
    print("Make sure src.tracking.track_enhanced is accessible")
    sys.exit(1)


class ToolTip:
    """Create a tooltip for a given widget."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        """Display the tooltip."""
        if self.tooltip_window or not self.text:
            return

        x, y, _, _ = (
            self.widget.bbox("insert") if hasattr(self.widget, "bbox") else (0, 0, 0, 0)
        )
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Helvetica", 9, "normal"),
            wraplength=250,
        )
        label.pack(ipadx=5, ipady=3)

    def hide_tooltip(self, event=None):
        """Hide the tooltip."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class ROISelector:
    """Interactive ROI (Region of Interest) selector for video cropping."""

    def __init__(self, parent, video_path, callback):
        self.parent = parent
        self.video_path = video_path
        self.callback = callback
        self.roi = None

        # Create window
        self.window = Toplevel(parent)
        self.window.title("Select Region of Interest")
        self.window.geometry("900x700")

        # Instructions
        instructions = ttk.Label(
            self.window,
            text="Click and drag to select the region of interest. Click 'Confirm' when done.",
            font=("Helvetica", 10),
        )
        instructions.pack(pady=10)

        # Canvas for image display
        self.canvas = Canvas(self.window, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Buttons
        button_frame = ttk.Frame(self.window)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Confirm", command=self.confirm_roi).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Reset", command=self.reset_roi).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(
            side=tk.LEFT, padx=5
        )

        # Load first frame
        self.load_first_frame()

        # Mouse events
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def load_first_frame(self):
        """Load and display the first frame of the video."""
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Error", "Could not read video file.")
            self.window.destroy()
            return

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.original_frame = frame.copy()
        self.frame_height, self.frame_width = frame.shape[:2]

        # Resize for display
        max_width, max_height = 850, 600
        scale = min(max_width / self.frame_width, max_height / self.frame_height)
        self.display_width = int(self.frame_width * scale)
        self.display_height = int(self.frame_height * scale)

        frame_resized = cv2.resize(frame, (self.display_width, self.display_height))

        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(Image.fromarray(frame_resized))
        self.canvas.config(width=self.display_width, height=self.display_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def on_press(self, event):
        """Handle mouse press."""
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="red",
            width=2,
        )

    def on_drag(self, event):
        """Handle mouse drag."""
        if self.rect_id:
            self.canvas.coords(
                self.rect_id, self.start_x, self.start_y, event.x, event.y
            )

    def on_release(self, event):
        """Handle mouse release."""
        # Get coordinates
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)

        # Convert to original frame coordinates
        scale_x = self.frame_width / self.display_width
        scale_y = self.frame_height / self.display_height

        self.roi = (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y),
        )

    def reset_roi(self):
        """Reset the ROI selection."""
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.roi = None
        self.start_x = None
        self.start_y = None

    def confirm_roi(self):
        """Confirm the ROI selection."""
        if self.roi:
            self.callback(self.roi)
            self.window.destroy()
        else:
            messagebox.showwarning("No Selection", "Please select a region first.")

    def cancel(self):
        """Cancel ROI selection."""
        self.window.destroy()


class AntTrackerGUI:
    """Main GUI application for ant tracking system."""

    def __init__(self, root):
        self.root = root
        self.root.title("Ant Detection & Tracking System")
        self.root.geometry("500x350")
        self.root.resizable(False, False)

        # Set default paths
        self.script_dir = Path(__file__).parent.parent.parent
        self.default_model = self.script_dir / "models/trained/best.pt"
        self.default_output = self.script_dir / "outputs/tracking"

        # Basic variables
        self.model_path = tk.StringVar()
        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(self.default_output))
        self.tracking_mode = tk.StringVar(value="standard")

        # Tracking parameters
        self.conf_threshold = tk.DoubleVar(value=0.25)
        self.iou_threshold = tk.DoubleVar(value=0.7)

        # New advanced parameters
        self.output_fps = tk.IntVar(value=30)
        self.use_original_fps = tk.BooleanVar(value=True)
        self.imgsz = tk.IntVar(value=1280)  # Changed to highest resolution by default
        self.save_txt = tk.BooleanVar(value=False)
        self.save_json = tk.BooleanVar(value=False)
        self.export_csv = tk.BooleanVar(value=True)
        self.line_width = tk.IntVar(value=2)
        self.trail_length = tk.IntVar(value=30)

        # Display options
        self.show_trails = tk.BooleanVar(value=False)
        self.show_ids = tk.BooleanVar(value=False)

        # Tracker parameters
        self.tracker_type = tk.StringVar(value="botsort")
        self.track_buffer = tk.IntVar(value=30)

        # Advanced options
        self.enable_advanced = tk.BooleanVar(value=False)

        # Image enhancement parameters
        self.enable_enhancement = tk.BooleanVar(value=False)
        self.enhancement_type = tk.StringVar(
            value="combined"
        )  # Default to best option when enabled
        self.super_res_scale = tk.DoubleVar(value=1.5)  # Default to recommended scale

        # Statistics parameters
        self.enable_statistics = tk.BooleanVar(value=False)
        self.stats_start_time = tk.DoubleVar(value=0.0)
        self.stats_end_time = tk.DoubleVar(value=0.0)

        # ROI (Region of Interest)
        self.roi_coords = None
        self.use_roi = tk.BooleanVar(value=False)

        # Frame saving mode (more reliable than video writer)
        self.save_frames_mode = tk.BooleanVar(
            value=True
        )  # Default to True for HPC reliability
        self.max_frames = tk.IntVar(value=0)  # 0 = process all frames

        # Set default model if it exists
        if self.default_model.exists():
            self.model_path.set(str(self.default_model))

        # Track output file path and statistics
        self.output_video_path = None
        self.output_frames_dir = None
        self.tracking_statistics = None

        # Show main menu
        self.show_main_menu()

    def show_main_menu(self):
        """Display the main menu with Train/Track options."""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()

        # Main frame
        main_frame = ttk.Frame(self.root, padding="40")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(
            main_frame,
            text="Ant Detection & Tracking System",
            font=("Helvetica", 16, "bold"),
        )
        title.pack(pady=(0, 40))

        # Subtitle
        subtitle = ttk.Label(
            main_frame, text="Choose an option:", font=("Helvetica", 11)
        )
        subtitle.pack(pady=(0, 20))

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        # Train button
        train_btn = ttk.Button(
            button_frame,
            text="Train Model",
            command=self.open_training_instructions,
            width=20,
        )
        train_btn.pack(pady=10)

        # Track button
        track_btn = ttk.Button(
            button_frame,
            text="Track Video",
            command=self.show_tracking_interface,
            width=20,
        )
        track_btn.pack(pady=10)

        # Footer
        footer = ttk.Label(
            main_frame,
            text="YOLOv8-OBB Ant Tracker",
            font=("Helvetica", 9),
            foreground="gray",
        )
        footer.pack(side=tk.BOTTOM, pady=(40, 0))

    def open_training_instructions(self):
        """Open the training instructions and documentation."""
        instructions_file = self.script_dir / "training_detector.pdf"

        if not instructions_file.exists():
            messagebox.showerror(
                "File Not Found",
                f"Documentation file not found:\n{instructions_file}\n\n"
                "Please check your installation.",
            )
            return

        # Open file with default application
        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.call(["open", str(instructions_file)])
            elif platform.system() == "Windows":
                os.startfile(str(instructions_file))
            else:  # Linux
                subprocess.call(["xdg-open", str(instructions_file)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file:\n{e}")

    def show_tracking_interface(self):
        """Display the tracking interface."""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()

        # Resize window for tracking interface
        # Smaller default size for laptop screens
        self.root.geometry("700x600")
        self.root.resizable(True, True)
        self.root.minsize(600, 500)

        # Create main frame with canvas and scrollbar
        self.tracking_canvas = Canvas(self.root)
        scrollbar = ttk.Scrollbar(
            self.root, orient="vertical", command=self.tracking_canvas.yview
        )
        self.scrollable_frame = ttk.Frame(self.tracking_canvas, padding="15")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.tracking_canvas.configure(
                scrollregion=self.tracking_canvas.bbox("all")
            ),
        )

        self.canvas_window = self.tracking_canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )
        self.tracking_canvas.configure(yscrollcommand=scrollbar.set)

        # Bind canvas resize to update scrollable frame width
        def on_canvas_configure(event):
            self.tracking_canvas.itemconfig(self.canvas_window, width=event.width)

        self.tracking_canvas.bind("<Configure>", on_canvas_configure)

        self.tracking_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Make mousewheel work
        def _on_mousewheel(event):
            self.tracking_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.tracking_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        main_frame = self.scrollable_frame

        # Title
        title = ttk.Label(
            main_frame, text="Video Tracking", font=("Helvetica", 14, "bold")
        )
        title.pack(pady=(0, 20))

        # Model selection section
        model_frame = ttk.LabelFrame(main_frame, text="Model Selection", padding="10")
        model_frame.pack(fill=tk.X, pady=5)

        ttk.Label(model_frame, text="Model File (.pt):").pack(anchor=tk.W)
        model_entry_frame = ttk.Frame(model_frame)
        model_entry_frame.pack(fill=tk.X, pady=5)

        model_entry = ttk.Entry(model_entry_frame, textvariable=self.model_path)
        model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        model_btn = ttk.Button(
            model_entry_frame, text="Browse", command=self.browse_model, width=10
        )
        model_btn.pack(side=tk.RIGHT)

        # Video upload section
        video_frame = ttk.LabelFrame(main_frame, text="Video Input", padding="10")
        video_frame.pack(fill=tk.X, pady=5)

        ttk.Label(video_frame, text="Video File:").pack(anchor=tk.W)
        video_entry_frame = ttk.Frame(video_frame)
        video_entry_frame.pack(fill=tk.X, pady=5)

        video_entry = ttk.Entry(video_entry_frame, textvariable=self.video_path)
        video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        video_btn = ttk.Button(
            video_entry_frame, text="Browse", command=self.browse_video, width=10
        )
        video_btn.pack(side=tk.RIGHT)

        # Tracking options section
        options_frame = ttk.LabelFrame(
            main_frame, text="Tracking Options", padding="10"
        )
        options_frame.pack(fill=tk.X, pady=5)

        # Tracking mode
        mode_label = ttk.Label(options_frame, text="Tracking Mode:")
        mode_label.pack(anchor=tk.W, pady=(0, 5))
        ToolTip(
            mode_label,
            "Standard: Shows boxes around each ant\nWith Trail: Also shows movement paths to visualize where ants have traveled",
        )

        mode_frame = ttk.Frame(options_frame)
        mode_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(
            mode_frame,
            text="Standard Tracking",
            variable=self.tracking_mode,
            value="standard",
        ).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Radiobutton(
            mode_frame,
            text="With Trail Visualizations",
            variable=self.tracking_mode,
            value="custom",
        ).pack(side=tk.LEFT)

        # Tracker type
        tracker_label = ttk.Label(options_frame, text="Tracker Algorithm:")
        tracker_label.pack(anchor=tk.W, pady=(10, 5))
        ToolTip(
            tracker_label,
            "BoT-SORT: Better when ants temporarily disappear (behind objects, under other ants). Recommended.\nByteTrack: Faster but simpler - good for clear videos where ants are always visible",
        )

        tracker_frame = ttk.Frame(options_frame)
        tracker_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(
            tracker_frame,
            text="BoT-SORT (Recommended)",
            variable=self.tracker_type,
            value="botsort",
        ).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Radiobutton(
            tracker_frame,
            text="ByteTrack",
            variable=self.tracker_type,
            value="bytetrack",
        ).pack(side=tk.LEFT)

        # Confidence threshold
        conf_label = ttk.Label(options_frame, text="Confidence Threshold:")
        conf_label.pack(anchor=tk.W, pady=(10, 5))
        ToolTip(
            conf_label,
            "How sure we are that a detected object is actually an ant (not dust, shadows, or other objects).\nLower (0.1-0.2) = more detections but more false positives\nHigher (0.4-0.6) = only very confident detections\nDefault 0.25 works well for most ant videos",
        )

        conf_frame = ttk.Frame(options_frame)
        conf_frame.pack(fill=tk.X)

        conf_slider = ttk.Scale(
            conf_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.conf_threshold,
        )
        conf_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.conf_label = ttk.Label(conf_frame, text=f"{self.conf_threshold.get():.2f}")
        self.conf_label.pack(side=tk.RIGHT)
        conf_slider.config(
            command=lambda v: self.conf_label.config(text=f"{float(v):.2f}")
        )

        # IoU threshold
        iou_label = ttk.Label(options_frame, text="IoU Threshold:")
        iou_label.pack(anchor=tk.W, pady=(10, 5))
        ToolTip(
            iou_label,
            "Removes duplicate boxes around the same ant.\nLower values (0.5) are stricter about removing overlaps\nHigher values (0.8) keep more boxes\nDefault 0.7 works well for ants that sometimes cluster together",
        )

        iou_frame = ttk.Frame(options_frame)
        iou_frame.pack(fill=tk.X)

        iou_slider = ttk.Scale(
            iou_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.iou_threshold,
        )
        iou_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.iou_label = ttk.Label(iou_frame, text=f"{self.iou_threshold.get():.2f}")
        self.iou_label.pack(side=tk.RIGHT)
        iou_slider.config(
            command=lambda v: self.iou_label.config(text=f"{float(v):.2f}")
        )

        # --- Advanced Options Section ---
        self.advanced_frame = ttk.LabelFrame(
            main_frame, text="Advanced Options", padding="10"
        )
        self.advanced_frame.pack(fill=tk.X, pady=5)

        # Configure grid columns to expand
        self.advanced_frame.columnconfigure(1, weight=1)

        advanced_check = ttk.Checkbutton(
            self.advanced_frame,
            text="Enable Advanced Options",
            variable=self.enable_advanced,
            command=self.toggle_advanced_controls,
        )
        advanced_check.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        ToolTip(
            advanced_check,
            "Show additional tracking parameters for fine-tuning. Most users can use default settings.",
        )

        # Advanced controls (initially hidden)
        self.advanced_controls_frame = ttk.Frame(self.advanced_frame)
        self.advanced_controls_frame.columnconfigure(1, weight=1)

        # Track buffer (renamed to "Track Length" as requested)
        buffer_label = ttk.Label(
            self.advanced_controls_frame, text="Track Length (frames):"
        )
        buffer_label.grid(row=0, column=0, sticky=tk.W, pady=5)
        ToolTip(
            buffer_label,
            "How many frames to keep tracking after an ant disappears from view. Useful when ants go behind objects or under each other.\nDefault: 30 frames (1 second at 30fps) works for most cases",
        )

        buffer_spin = ttk.Spinbox(
            self.advanced_controls_frame,
            from_=10,
            to=100,
            textvariable=self.track_buffer,
            width=15,
        )
        buffer_spin.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Line width
        line_label = ttk.Label(self.advanced_controls_frame, text="Box Line Width:")
        line_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        ToolTip(
            line_label,
            "Thickness of the box drawn around each ant. Increase if hard to see on small screens.\nDefault: 2 pixels",
        )

        line_spin = ttk.Spinbox(
            self.advanced_controls_frame,
            from_=1,
            to=5,
            textvariable=self.line_width,
            width=15,
        )
        line_spin.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Initially hide advanced controls
        if not self.enable_advanced.get():
            self.advanced_controls_frame.grid_forget()

        # --- Image Enhancement Section ---
        self.enhancement_frame = ttk.LabelFrame(
            main_frame, text="Image Enhancement (Experimental)", padding="10"
        )
        self.enhancement_frame.pack(fill=tk.X, pady=5)

        # Configure grid columns
        self.enhancement_frame.columnconfigure(1, weight=1)

        enh_check = ttk.Checkbutton(
            self.enhancement_frame,
            text="Enable Image Enhancement",
            variable=self.enable_enhancement,
            command=self.toggle_enhancement_controls,
        )
        enh_check.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        ToolTip(
            enh_check,
            "Improve image quality for better ant detection. Useful when ants are small, blend with background, or video quality is low.\nWarning: Increases processing time by 50-70%\nTry this if default tracking misses ants",
        )

        # Enhancement controls (initially hidden)
        self.enhancement_controls_frame = ttk.Frame(self.enhancement_frame)
        self.enhancement_controls_frame.columnconfigure(1, weight=1)

        enh_label = ttk.Label(self.enhancement_controls_frame, text="Enhancement Type:")
        enh_label.grid(row=0, column=0, sticky=tk.W, pady=5)
        ToolTip(
            enh_label,
            "Combined (CLAHE + Sharpen): Best for most ant videos - brightens dark areas and sharpens ant edges\nCLAHE: Adjusts brightness/contrast, good for uneven lighting\nSharpen: Makes ant edges crisper\nDefault: Combined",
        )

        enh_combo = ttk.Combobox(
            self.enhancement_controls_frame,
            textvariable=self.enhancement_type,
            values=[
                "none",
                "sharpen",
                "sharpen_strong",
                "clahe",
                "denoise",
                "combined",
            ],
            width=15,
            state="readonly",
        )
        enh_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Enhancement descriptions
        enh_desc_frame = ttk.Frame(self.enhancement_controls_frame)
        enh_desc_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)

        enh_descriptions = {
            "none": "• None: No enhancement (fastest)",
            "sharpen": "• Sharpen: Moderate sharpening",
            "sharpen_strong": "• Sharpen Strong: Aggressive sharpening",
            "clahe": "• CLAHE: Contrast enhancement",
            "denoise": "• Denoise: Noise reduction",
            "combined": "• Combined: CLAHE + Sharpen (BEST for ants)",
        }

        self.enh_desc_label = ttk.Label(
            enh_desc_frame,
            text=enh_descriptions["combined"],
            foreground="gray",
            font=("Helvetica", 9),
        )
        self.enh_desc_label.pack(anchor=tk.W)

        def update_enh_description(*args):
            self.enh_desc_label.config(
                text=enh_descriptions.get(self.enhancement_type.get(), "")
            )

        self.enhancement_type.trace_add("write", update_enh_description)

        # Super-resolution
        sr_label = ttk.Label(
            self.enhancement_controls_frame, text="Super-Resolution Scale:"
        )
        sr_label.grid(row=2, column=0, sticky=tk.W, pady=5)
        ToolTip(
            sr_label,
            "Makes the image bigger before detecting ants.\n1.0 = no change (fastest)\n1.5 = 50% bigger (recommended for small ants)\n2.0 = 2x bigger (slowest but best for tiny ants)\nUse this for videos where ants are very small (less than 20 pixels)",
        )

        sr_combo = ttk.Combobox(
            self.enhancement_controls_frame,
            textvariable=self.super_res_scale,
            values=[1.0, 1.25, 1.5, 2.0],
            width=10,
            state="readonly",
        )
        sr_combo.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Warning label
        enh_warning = ttk.Label(
            self.enhancement_controls_frame,
            text="WARNING: Enhancement increases processing time. Test on short clips first.",
            foreground="orange",
            font=("Helvetica", 8),
        )
        enh_warning.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        # Initially hide enhancement controls
        if not self.enable_enhancement.get():
            self.enhancement_controls_frame.grid_forget()

        # --- Output Settings Section ---
        output_settings_frame = ttk.LabelFrame(
            main_frame, text="Output Settings", padding="10"
        )
        output_settings_frame.pack(fill=tk.X, pady=5)

        # Configure grid columns
        output_settings_frame.columnconfigure(1, weight=1)

        # Export options
        ttk.Label(output_settings_frame, text="Export Formats:").grid(
            row=0, column=0, sticky=tk.W, pady=(5, 5)
        )

        export_frame = ttk.Frame(output_settings_frame)
        export_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)

        csv_check = ttk.Checkbutton(
            export_frame, text="CSV (Track data)", variable=self.export_csv
        )
        csv_check.pack(side=tk.LEFT, padx=(0, 15))
        ToolTip(
            csv_check,
            "Save ant tracking data to spreadsheet file. Contains frame number, ant ID, position (x,y coordinates), and timestamp for each detected ant.\nOpen with Excel, Google Sheets, or R/Python for analysis",
        )

        txt_check = ttk.Checkbutton(
            export_frame, text="TXT (Per-frame)", variable=self.save_txt
        )
        txt_check.pack(side=tk.LEFT, padx=(0, 15))
        ToolTip(
            txt_check,
            "Save detection coordinates in YOLO format - one text file per frame. For advanced users who want to retrain models or use custom analysis tools",
        )

        json_check = ttk.Checkbutton(export_frame, text="JSON", variable=self.save_json)
        json_check.pack(side=tk.LEFT)
        ToolTip(
            json_check,
            "Save tracking results in JSON format (computer-readable). Useful if you're writing custom analysis scripts",
        )

        # --- ROI Section ---
        self.roi_frame = ttk.LabelFrame(
            main_frame, text="Region of Interest", padding="10"
        )
        self.roi_frame.pack(fill=tk.X, pady=5)

        roi_check = ttk.Checkbutton(
            self.roi_frame,
            text="Enable ROI Cropping",
            variable=self.use_roi,
            command=self.toggle_roi_controls,
        )
        roi_check.pack(anchor=tk.W, pady=5)
        ToolTip(
            roi_check,
            "Region of Interest - process only a specific area of the video.\nSpeeds up tracking significantly and focuses on your experimental arena.\nClick 'Select ROI' to draw a box on the first frame\nExample: Track only the feeding area, ignore the rest",
        )

        # ROI controls (initially hidden)
        self.roi_controls_frame = ttk.Frame(self.roi_frame)

        roi_button_frame = ttk.Frame(self.roi_controls_frame)
        roi_button_frame.pack(fill=tk.X, pady=5)

        self.roi_select_btn = ttk.Button(
            roi_button_frame, text="Select ROI", command=self.select_roi, width=15
        )
        self.roi_select_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.roi_clear_btn = ttk.Button(
            roi_button_frame, text="Clear", command=self.clear_roi, width=10
        )
        self.roi_clear_btn.pack(side=tk.LEFT)

        self.roi_status_label = ttk.Label(
            self.roi_controls_frame, text="No ROI selected", foreground="gray"
        )
        self.roi_status_label.pack(anchor=tk.W, pady=(5, 0))

        # Initially hide ROI controls
        if not self.use_roi.get():
            self.roi_controls_frame.pack_forget()

        # --- Statistics Section ---
        self.stats_frame = ttk.LabelFrame(
            main_frame, text="Statistics & Analysis", padding="10"
        )
        self.stats_frame.pack(fill=tk.X, pady=5)

        stats_check = ttk.Checkbutton(
            self.stats_frame,
            text="Enable Custom Statistics",
            variable=self.enable_statistics,
            command=self.toggle_stats_controls,
        )
        stats_check.pack(anchor=tk.W, pady=5)
        ToolTip(
            stats_check,
            "Calculate average, minimum, and maximum number of ants visible during a specific time period.\nUseful for experiments: count ants at feeding site between seconds 10-60\nResults displayed after tracking completes",
        )

        # Statistics controls (initially hidden)
        self.stats_controls_frame = ttk.Frame(self.stats_frame)

        time_frame = ttk.Frame(self.stats_controls_frame)
        time_frame.pack(fill=tk.X, pady=5)

        # Configure grid columns
        time_frame.columnconfigure(1, weight=1)
        time_frame.columnconfigure(3, weight=1)

        ttk.Label(time_frame, text="Start Time (sec):").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        start_spin = ttk.Spinbox(
            time_frame,
            from_=0,
            to=10000,
            textvariable=self.stats_start_time,
            width=12,
            increment=0.5,
        )
        start_spin.grid(row=0, column=1, sticky=tk.W, padx=(10, 20), pady=5)
        ToolTip(start_spin, "Start time for analysis")

        ttk.Label(time_frame, text="End Time (sec):").grid(
            row=0, column=2, sticky=tk.W, pady=5
        )
        end_spin = ttk.Spinbox(
            time_frame,
            from_=0,
            to=10000,
            textvariable=self.stats_end_time,
            width=12,
            increment=0.5,
        )
        end_spin.grid(row=0, column=3, sticky=tk.W, padx=(10, 0), pady=5)
        ToolTip(end_spin, "End time (0 = end of video)")

        # Initially hide stats controls
        if not self.enable_statistics.get():
            self.stats_controls_frame.pack_forget()

        # Output directory section
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        output_frame.pack(fill=tk.X, pady=5)

        ttk.Label(output_frame, text="Output Directory:").pack(anchor=tk.W)
        output_entry_frame = ttk.Frame(output_frame)
        output_entry_frame.pack(fill=tk.X, pady=5)

        output_entry = ttk.Entry(output_entry_frame, textvariable=self.output_dir)
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        output_btn = ttk.Button(
            output_entry_frame, text="Browse", command=self.browse_output, width=10
        )
        output_btn.pack(side=tk.RIGHT)

        # Progress section
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=10)

        self.progress_bar = ttk.Progressbar(
            progress_frame, mode="indeterminate", length=300
        )
        self.progress_bar.pack(fill=tk.X)

        self.status_label = ttk.Label(
            progress_frame, text="Ready", font=("Helvetica", 9)
        )
        self.status_label.pack(pady=5)

        # Buttons section (centered)
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # Create inner frame to hold buttons
        button_inner_frame = ttk.Frame(button_frame)
        button_inner_frame.pack(anchor=tk.CENTER)

        # Run button
        self.run_btn = ttk.Button(
            button_inner_frame, text="Run Tracker", command=self.run_tracking, width=15
        )
        self.run_btn.pack(side=tk.LEFT, padx=5)

        # Back button
        back_btn = ttk.Button(
            button_inner_frame,
            text="Back to Menu",
            command=self.show_main_menu,
            width=15,
        )
        back_btn.pack(side=tk.LEFT, padx=5)

    def browse_model(self):
        """Open file dialog to select model file."""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            initialdir=self.script_dir,
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")],
        )
        if filename:
            self.model_path.set(filename)

    def browse_video(self):
        """Open file dialog to select video file."""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            initialdir=self.script_dir,
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.MP4 *.AVI *.MOV"),
                ("All Files", "*.*"),
            ],
        )
        if filename:
            self.video_path.set(filename)

    def browse_output(self):
        """Open folder dialog to select output directory."""
        dirname = filedialog.askdirectory(
            title="Select Output Directory", initialdir=self.script_dir
        )
        if dirname:
            self.output_dir.set(dirname)

    def select_roi(self):
        """Open ROI selector window."""
        if not self.video_path.get():
            messagebox.showwarning("No Video", "Please select a video file first.")
            return

        if not os.path.exists(self.video_path.get()):
            messagebox.showerror(
                "Error", f"Video file not found:\n{self.video_path.get()}"
            )
            return

        # Callback to receive ROI coordinates
        def roi_callback(roi):
            self.roi_coords = roi
            self.roi_status_label.config(
                text=f"ROI selected: ({roi[0]}, {roi[1]}) to ({roi[2]}, {roi[3]})",
                foreground="green",
            )
            self.use_roi.set(True)

        # Open ROI selector
        ROISelector(self.root, self.video_path.get(), roi_callback)

    def clear_roi(self):
        """Clear the selected ROI."""
        self.roi_coords = None
        self.use_roi.set(False)
        self.roi_status_label.config(text="No ROI selected", foreground="gray")

    def toggle_roi_controls(self):
        """Show/hide ROI controls based on checkbox."""
        if self.use_roi.get():
            self.roi_controls_frame.pack(fill=tk.X, pady=(5, 0))
        else:
            self.roi_controls_frame.pack_forget()
            # Clear ROI when disabled
            if self.roi_coords:
                self.roi_coords = None
                self.roi_status_label.config(text="No ROI selected", foreground="gray")

    def toggle_stats_controls(self):
        """Show/hide statistics controls based on checkbox."""
        if self.enable_statistics.get():
            self.stats_controls_frame.pack(fill=tk.X, pady=(5, 0))
        else:
            self.stats_controls_frame.pack_forget()

    def toggle_enhancement_controls(self):
        """Show/hide enhancement controls based on checkbox."""
        if self.enable_enhancement.get():
            self.enhancement_controls_frame.grid(
                row=1, column=0, columnspan=2, sticky=tk.W + tk.E, pady=(5, 0)
            )
        else:
            self.enhancement_controls_frame.grid_forget()

    def toggle_advanced_controls(self):
        """Show/hide advanced controls based on checkbox."""
        if self.enable_advanced.get():
            self.advanced_controls_frame.grid(
                row=1, column=0, columnspan=2, sticky=tk.W + tk.E, pady=(5, 0)
            )
        else:
            self.advanced_controls_frame.grid_forget()

    def toggle_fps_controls(self):
        """Show/hide custom FPS controls based on radio button selection."""
        if not self.use_original_fps.get():  # Custom FPS selected
            self.fps_controls_frame.grid(
                row=5, column=0, columnspan=2, sticky=tk.W, pady=(5, 0)
            )
        else:  # Original FPS selected
            self.fps_controls_frame.grid_forget()

    def toggle_frame_mode_controls(self):
        """Show/hide controls based on frame saving mode."""
        if self.save_frames_mode.get():
            # Show frame mode info
            self.frame_mode_info.grid(
                row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 10)
            )
            # Hide FPS controls when saving frames
            self.fps_settings_label.grid_forget()
            self.fps_radio_frame.grid_forget()
            self.fps_controls_frame.grid_forget()
        else:
            # Hide frame mode info
            self.frame_mode_info.grid_forget()
            # Show FPS controls when creating video directly
            self.fps_settings_label.grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
            self.fps_radio_frame.grid(
                row=4, column=0, columnspan=2, sticky=tk.W, pady=5
            )

    def validate_inputs(self):
        """Validate user inputs before running tracking."""
        # Check model file
        if not self.model_path.get():
            messagebox.showerror("Error", "Please select a model file.")
            return False

        if not os.path.exists(self.model_path.get()):
            messagebox.showerror(
                "Error", f"Model file not found:\n{self.model_path.get()}"
            )
            return False

        # Check video file
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file.")
            return False

        if not os.path.exists(self.video_path.get()):
            messagebox.showerror(
                "Error", f"Video file not found:\n{self.video_path.get()}"
            )
            return False

        # Check output directory
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return False

        return True

    def run_tracking(self):
        """Start tracking in a separate thread."""
        if not self.validate_inputs():
            return

        # Disable run button and show progress
        self.run_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Running tracking...")
        self.progress_bar.start(10)

        # Run tracking in separate thread
        tracking_thread = threading.Thread(target=self.execute_tracking, daemon=True)
        tracking_thread.start()

    def create_video_from_frames(self):
        """Create video from previously saved frames."""
        frames_dir = Path(self.output_dir.get()) / "frames"

        if not frames_dir.exists():
            messagebox.showerror(
                "No Frames Found",
                f"Frames directory not found:\n{frames_dir}\n\n"
                "Please run tracking with 'Save Frames' mode first.",
            )
            return

        # Disable button and show progress
        self.create_video_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Creating video from frames...")
        self.progress_bar.start(10)

        # Run in separate thread
        def create_video_thread():
            try:
                output_video = (
                    Path(self.output_dir.get()) / "ant_tracking_from_frames.mp4"
                )

                # Get FPS (use 30 as default)
                fps = 30

                print("\n" + "=" * 70)
                print("CREATING VIDEO FROM FRAMES")
                print("=" * 70)
                print(f"Frames Directory: {frames_dir}")
                print(f"Output Video: {output_video}")
                print(f"FPS: {fps}")
                print("=" * 70)
                print()

                frames_to_video_ffmpeg(frames_dir, output_video, fps=fps)

                self.output_video_path = output_video

                # Update UI on completion
                self.root.after(0, self.video_creation_complete)

            except Exception as e:
                import traceback

                traceback.print_exc()
                self.root.after(0, lambda: self.video_creation_error(str(e)))

        threading.Thread(target=create_video_thread, daemon=True).start()

    def video_creation_complete(self):
        """Handle successful video creation."""
        self.progress_bar.stop()
        self.status_label.config(text="Video created successfully!")
        self.create_video_btn.config(state=tk.NORMAL)

        message = (
            f"Video created successfully!\n\nOutput saved to:\n{self.output_video_path}"
        )
        messagebox.showinfo("Success", message)

    def video_creation_error(self, error_msg):
        """Handle video creation error."""
        self.progress_bar.stop()
        self.status_label.config(text="Video creation failed")
        self.create_video_btn.config(state=tk.NORMAL)

        messagebox.showerror(
            "Video Creation Error",
            f"An error occurred while creating video:\n\n{error_msg}\n\n"
            "Make sure ffmpeg is installed on your system.",
        )

    def execute_tracking(self):
        """Execute the tracking process."""
        try:
            # Get all parameters
            model_path = self.model_path.get()
            video_path = self.video_path.get()
            output_dir = self.output_dir.get()
            mode = self.tracking_mode.get()
            conf = self.conf_threshold.get()
            iou = self.iou_threshold.get()

            # Advanced parameters
            imgsz = self.imgsz.get()
            tracker_type = f"{self.tracker_type.get()}.yaml"
            line_width = self.line_width.get()
            trail_length = self.trail_length.get()

            # Output settings
            use_orig_fps = self.use_original_fps.get()
            fps = None if use_orig_fps else self.output_fps.get()

            # ROI
            roi = self.roi_coords if self.use_roi.get() else None

            # Statistics
            enable_stats = self.enable_statistics.get()
            stats_start = self.stats_start_time.get()
            stats_end = self.stats_end_time.get()

            # Export options
            export_csv = self.export_csv.get()
            export_txt = self.save_txt.get()
            export_json = self.save_json.get()

            # Enhancement options (only if enabled)
            if self.enable_enhancement.get():
                enhancement_type = self.enhancement_type.get()
                super_res_scale = self.super_res_scale.get()
            else:
                enhancement_type = "none"
                super_res_scale = 1.0

            # Display options - enable trails when custom mode is selected
            show_trails = mode == "custom"  # Automatically enable trails in custom mode
            show_ids = self.show_ids.get()

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            video_name = Path(video_path).stem

            # Print all selected options
            print("\n" + "=" * 70)
            print("TRACKING CONFIGURATION")
            print("=" * 70)
            print(f"Model: {Path(model_path).name}")
            print(f"Video: {Path(video_path).name}")
            print(f"Output Directory: {output_dir}")
            print()
            print("TRACKING PARAMETERS:")
            print(f"  Tracking Mode: {mode.capitalize()}")
            print(f"  Tracker Algorithm: {self.tracker_type.get().upper()}")
            print(f"  Confidence Threshold: {conf}")
            print(f"  IoU Threshold: {iou}")
            print()
            print("ADVANCED SETTINGS:")
            print(f"  Processing Resolution: {imgsz}px (High Quality)")
            print(f"  Track Buffer: {self.track_buffer.get()} frames")
            print(f"  Box Line Width: {line_width}px")
            if show_trails or show_ids:
                print(
                    f"  Display: Trails={'Yes' if show_trails else 'No'}, IDs={'Yes' if show_ids else 'No'}"
                )
            else:
                print(f"  Display: Clean (boxes + count only)")
            print()
            print("OUTPUT SETTINGS:")
            if use_orig_fps:
                print(f"  Output FPS: Original video FPS")
            else:
                print(f"  Output FPS: {fps} (custom)")
                # Get original FPS to show warning
                import cv2

                cap = cv2.VideoCapture(video_path)
                orig_fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()

                if fps < orig_fps:
                    print(
                        f"      WARNING: Custom FPS ({fps}) < Original FPS ({orig_fps:.0f})"
                    )
                    print(f"      Video will appear {orig_fps / fps:.1f}x faster!")
                    print(
                        f"      Recommendation: Use original FPS or match original ({int(orig_fps)})"
                    )
                elif fps > orig_fps:
                    print(
                        f"      WARNING: Custom FPS ({fps}) > Original FPS ({orig_fps:.0f})"
                    )
                    print(f"      Video will appear {fps / orig_fps:.1f}x slower!")
                    print(
                        f"      Recommendation: Use original FPS or match original ({int(orig_fps)})"
                    )
            print(f"  Export CSV: {'Yes' if export_csv else 'No'}")
            print(f"  Export TXT: {'Yes' if export_txt else 'No'}")
            print(f"  Export JSON: {'Yes' if export_json else 'No'}")
            print()
            print("IMAGE ENHANCEMENT:")
            if enhancement_type == "none":
                print(f"  Enhancement: None (disabled)")
            else:
                from src.preprocessing.image_enhancement import (
                    get_enhancement_description,
                )

                print(f"  Enhancement: {get_enhancement_description(enhancement_type)}")
                if super_res_scale > 1.0:
                    print(f"  Super-Resolution: {super_res_scale}x upscaling")
                    print(f"     Note: Enhancement increases processing time")
            print()
            if roi:
                print("REGION OF INTEREST:")
                print(f"  ROI Enabled: Yes")
                print(f"  Coordinates: ({roi[0]}, {roi[1]}) to ({roi[2]}, {roi[3]})")
                print(f"  Size: {roi[2] - roi[0]}x{roi[3] - roi[1]} pixels")
                print()
            else:
                print("REGION OF INTEREST: Full frame")
                print()
            if enable_stats:
                print("STATISTICS:")
                print(f"  Statistics Enabled: Yes")
                print(
                    f"  Time Range: {stats_start}s to {stats_end if stats_end > 0 else 'end'}s"
                )
                print()
            else:
                print("STATISTICS: Disabled")
                print()
            print("=" * 70)
            print()

            # Always use the unified tracking function with appropriate parameters
            output_video = Path(output_dir) / f"{video_name}_tracked.mp4"

            result = track_video_with_roi_and_stats(
                model_path=model_path,
                source=video_path,
                output_path=str(output_video),
                conf_threshold=conf,
                iou_threshold=iou,
                imgsz=imgsz,
                tracker=tracker_type,
                roi=roi,
                fps=fps,
                line_width=line_width,
                trail_length=trail_length,
                enable_statistics=enable_stats,
                stats_start_time=stats_start,
                stats_end_time=stats_end,
                enhancement_type=enhancement_type,
                super_res_scale=super_res_scale,
                show_trails=show_trails,
                show_ids=show_ids,
            )

            self.output_video_path = output_video
            self.tracking_statistics = result.get("statistics", {})

            # Export CSV if requested
            if export_csv:
                import csv

                csv_path = Path(output_dir) / f"{video_name}_track_data.csv"

                # Collect all tracking data and sort by frame_id
                all_tracks = []
                for track_id, positions in result["track_history"].items():
                    for frame, x, y, timestamp in positions:
                        all_tracks.append([frame, track_id, x, y, timestamp])

                # Sort by frame_id (first column)
                all_tracks.sort(key=lambda row: row[0])

                # Write to CSV
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["frame", "track_id", "x", "y", "timestamp"])
                    writer.writerows(all_tracks)

                print(f"Track data exported to: {csv_path}")

            # Update UI on completion (use after() for thread safety)
            self.root.after(0, self.tracking_complete)

        except Exception as e:
            import traceback

            traceback.print_exc()
            # Update UI with error (use after() for thread safety)
            self.root.after(0, lambda: self.tracking_error(str(e)))

    def tracking_complete(self):
        """Handle successful tracking completion."""
        # Stop progress bar
        self.progress_bar.stop()
        self.status_label.config(text="Tracking complete!")
        self.run_btn.config(state=tk.NORMAL)

        # Build completion message
        message = f"Tracking completed successfully!\n\nOutput saved to:\n{self.output_video_path}\n"

        # Add statistics if available
        if self.tracking_statistics:
            stats = self.tracking_statistics
            message += f"\n=== Statistics ===\n"
            message += f"Time range: {stats['time_range'][0]:.1f}s - {stats['time_range'][1]:.1f}s\n"
            message += f"Average ants detected: {stats['average_ants']:.2f}\n"
            message += f"Min: {stats['min_ants']} | Max: {stats['max_ants']}\n"
            message += f"Median: {stats['median_ants']:.2f}\n"
            message += f"Total unique tracks: {stats['total_unique_tracks']}\n"

        # Show completion message
        messagebox.showinfo("Tracking Complete", message)

    def tracking_error(self, error_msg):
        """Handle tracking error."""
        # Stop progress bar
        self.progress_bar.stop()
        self.status_label.config(text="Error occurred")
        self.run_btn.config(state=tk.NORMAL)

        # Show error message
        messagebox.showerror(
            "Tracking Error", f"An error occurred during tracking:\n\n{error_msg}"
        )

    def open_video(self):
        """Open the output video in default player."""
        if not self.output_video_path or not os.path.exists(self.output_video_path):
            messagebox.showerror("Error", "Output video file not found.")
            return

        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.call(["open", str(self.output_video_path)])
            elif platform.system() == "Windows":
                os.startfile(str(self.output_video_path))
            else:  # Linux
                subprocess.call(["xdg-open", str(self.output_video_path)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open video:\n{e}")


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = AntTrackerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
