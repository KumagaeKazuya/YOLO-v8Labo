from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque

# 亀岡作成のコードを統合用に調整
class VideoDistortionCorrector:
    """動画の歪み補正クラス（逆バレル補正版）"""

    def __init__(self, k1=-0.1, strength=1.0, zoom_factor=1.2):
        self.k1 = k1
        self.strength = strength
        self.zoom_factor = zoom_factor
        self.map_x = None
        self.map_y = None

    def create_correction_maps(self, width, height):
        """
        逆バレル歪み補正とズーム調整用のマップを作成

        Returns:
        map_x, map_y: 歪み補正用のマップ
        """
        # 画像の中心を計算
        cx, cy = width // 2, height // 2
        max_radius = min(cx, cy)

        # 変換マップを作成
        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)

        # 調整された歪み係数
        adjusted_k1 = self.k1 * self.strength

        # 各ピクセルの補正を計算
        for y in range(height):
            for x in range(width):
                # 中心からの距離
                dx = x - cx
                dy = y - cy
                r = np.sqrt(dx*dx + dy*dy)

                if r > 0:
                    # 正規化された半径
                    r_norm = r / max_radius

                    # 逆バレル歪み補正
                    r_corrected = r * (1 + adjusted_k1 * r_norm * r_norm)

                    # ズーム調整を適用
                    scale = (r_corrected / r) * self.zoom_factor
                    new_x = cx + dx * scale
                    new_y = cy + dy * scale

                    map_x[y, x] = new_x
                    map_y[y, x] = new_y
                else:
                    map_x[y, x] = x
                    map_y[y, x] = y

        self.map_x = map_x
        self.map_y = map_y

        return self.map_x, self.map_y

    def apply_correction(self, frame):
        """フレームに歪み補正を適用"""
        if self.map_x is None or self.map_y is None:
            raise ValueError("補正マップが作成されていません。create_correction_maps()を先に呼び出してください。")

        return cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)