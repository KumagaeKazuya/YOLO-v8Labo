# test_pose.py

import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')  # 姿勢検出用の軽量モデル
results = model('https://ultralytics.com/images/bus.jpg')
results[0].show()
