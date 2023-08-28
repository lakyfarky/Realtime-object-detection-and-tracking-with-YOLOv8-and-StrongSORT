# Object Detection and Tracking with YOLOv8 and StrongSORT: Drone-Captured Data

This repository showcases my graduate thesis project focused on leveraging YOLOv8 for real-time object detection and integrating StrongSORT for accurate object tracking. By harnessing drone-captured data, this project explores the synergy between advanced computer vision algorithms and aerial imagery, opening up new possibilities for surveillance, mapping, and more.

## Installation

1. Clone this repository: 
```(bash)
https://github.com/lakyfarky/Realtime-object-detection-and-tracking-with-YOLOv8-and-StrongSORT.git
```
2. Install PyTorch (CUDA 11.8, **skip if don't have NVIDIA GPU with CUDA support**):
```(bash)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
3. Install Ultralytics (YOLOv8):
```(bash)
pip install Ultralytics
```
4. Install Boxmot (StrongSort):
```(bash)
pip install boxmot
```