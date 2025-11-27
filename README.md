⚡ Real-Time Pantograph Spark Detection
A high-performance computer vision system designed to detect electrical arcing (sparks) between railway pantographs and overhead catenary wires.

This project was engineered to solve the "FPS vs Accuracy" trade-off, achieving 60-70 FPS on mid-range hardware (NVIDIA GTX 1650) without sacrificing detection reliability.

🚀 Key Features
Hybrid Tracking: Utilizes high-speed Template Matching to track the moving pantograph, dynamically cropping the Region of Interest (ROI) to reduce AI workload.

Intelligent Pixel Gating: Implements a pre-processing "gate" that analyzes pixel intensity and shape before running the AI. This allows the system to skip heavy inference on empty frames (99% of the video), resulting in massive speed gains.

False Positive Filtration: Includes specific algorithms to differentiate between actual fire/sparks and shiny copper overhead wires (sunlight reflections) using morphological erosion and aspect-ratio analysis.

Asynchronous I/O: Decouples image saving from the main processing thread using queues, ensuring that disk writes never cause video stutter or frame drops.

Live & Video Modes: Supports both pre-recorded video analysis and low-latency live camera feeds.

🛠️ Tech Stack
Language: Python 3.9+

Vision: OpenCV (cv2)

AI Model: YOLOv8 (Ultralytics) - Exported to TensorRT (.engine) for max speed.

Hardware: Optimized for NVIDIA GTX 1650 (Laptop).
