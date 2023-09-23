# Object Detection and Tracking with YOLOv8 and StrongSORT: Drone-Captured Data

This repository showcases my graduate thesis project focused on leveraging YOLOv8 for real-time object detection and integrating StrongSORT for accurate object tracking. By harnessing drone-captured data, this project explores the synergy between advanced computer vision algorithms and aerial imagery, opening up new possibilities for surveillance, mapping, etc.

[Imgur](https://imgur.com/z407DNo)
## Installation

0. Create python virtual enviroment: 
```bash
python -m venv [venv_name]
source [venv_name]/scripts/activate
```
1. Clone this repository: 
```bash
git clone https://github.com/lakyfarky/Realtime-object-detection-and-tracking-with-YOLOv8-and-StrongSORT.git
cd 'Realtime-object-detection-and-tracking-with-YOLOv8-and-StrongSORT'
```
2. Install PyTorch (CUDA 11.8, **skip if don't have NVIDIA GPU with CUDA support**):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
3. Install Ultralytics (YOLOv8):
```bash
pip install Ultralytics
```
4. Install Boxmot (StrongSort):
```bash
pip install boxmot
```
<details>
    <summary> <h2>EL-YOLOv8</h2></summary>
To utilize EL-YOLOv8s model follow next steps:

1. **Copy ESPP.py into ultralytics/nn/modules/block.py** and add ESPP in special attribute:
```python
__all__ = ('ESPP', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3')
```
2. Add ESPP to ultralytics/nn/modules/\_\_init\_\_.py
```python
from .block import (ESPP, C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3)
```
3. Add ESPP to ultralytics/nn/models/task.py
```python
from ultralytics.nn.modules import (ESPP, AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,
                                    RTDETRDecoder, Segment)
```
</details>

## Used Repositories

Here are the repositories that I've used in this project:

1. [Ultralytics](https://github.com/ultralytics/ultralytics)

2. [Boxmot](https://github.com/mikel-brostrom/yolo_tracking)

3. [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset)
