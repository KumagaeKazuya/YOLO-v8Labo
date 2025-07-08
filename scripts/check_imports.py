try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    import time
    import math
    import os
    from collections import defaultdict, deque
    import logging
    print("すべてのモジュールが正常にインポートされました。")
except ImportError as e:
    print(f"インポートエラーが発生しました: {e}")
