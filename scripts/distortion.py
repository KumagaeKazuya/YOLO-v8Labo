import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from ultralytics import YOLO
from collections import defaultdict, deque
import logging
import csv
from datetime import datetime

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class EnhancedCSVLogger:
    """機械学習用拡張CSVロガー"""

    def __init__(self, csv_path="enhanced_detection_log.csv"):
        self.csv_path = csv_path
        self.csv_file = None
        self.csv_writer = None
        self.start_time = time.time()
        self.prev_keypoints = {}  # track_id -> previous keypoints for motion calculation

        # CSV ヘッダー定義
        self.headers = [
            # 基本情報
            "timestamp", "frame_idx", "relative_time_sec", "frame_interval_ms",

            # 人物識別情報
            "person_id", "track_confidence", "detection_confidence",

            # 位置情報
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", 
            "bbox_width", "bbox_height", "bbox_area",
            "center_x", "center_y", "grid_row", "grid_col",

            # 行動判定
            "using_phone", "phone_detection_confidence", "smoothed_phone_usage",
            "phone_detection_method",  # "front_view", "back_view", "failed"

            # キーポイント座標 (17点 x 3座標 = 51列)
            *[f"kp_{i}_{coord}" for i in range(17) for coord in ["x", "y", "conf"]],

            # 動作特徴量
            "movement_speed_px_per_frame", "pose_change_magnitude", 
            "head_movement_speed", "hand_movement_speed",
            "wrist_to_face_dist_left", "wrist_to_face_dist_right", "min_wrist_face_dist",

            # 姿勢分析
            "head_pose_angle", "shoulder_width", "torso_lean_angle",
            "left_arm_angle", "right_arm_angle",

            # 画像品質・環境要因
            "frame_brightness", "frame_contrast", "blur_score", "noise_level",

            # 時系列特徴
            "consecutive_phone_frames", "phone_state_duration_sec",
            "position_stability", "tracking_quality",

            # アノテーション用
            "manual_label_phone", "manual_label_posture", "manual_label_attention",
            "annotation_confidence", "annotator_id", "review_required",

            # メタデータ
            "video_source", "processing_version", "model_version", "notes"
        ]

        self.init_csv()

    def init_csv(self):
        """CSV ファイルを初期化"""
        try:
            self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(self.headers)
            logger.info(f"拡張CSVログを初期化: {self.csv_path}")
        except Exception as e:
            logger.error(f"CSV初期化エラー: {e}")

    def calculate_keypoint_features(self, keypoints, track_id):
        """キーポイントから特徴量を計算"""
        features = {}

        try:
            # キーポイントのインデックス定義
            kp_idx = {
                'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
                'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
                'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
                'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
            }

            # 有効なキーポイントのみ取得
            def get_valid_point(idx):
                if idx >= len(keypoints) or keypoints[idx][2] < 0.3:  # 信頼度チェック
                    return None
                return keypoints[idx][:2]

            # 基本的な距離計算
            nose = get_valid_point(kp_idx['nose'])
            left_wrist = get_valid_point(kp_idx['left_wrist'])
            right_wrist = get_valid_point(kp_idx['right_wrist'])
            left_shoulder = get_valid_point(kp_idx['left_shoulder'])
            right_shoulder = get_valid_point(kp_idx['right_shoulder'])

            # 手首-顔の距離
            if nose is not None and left_wrist is not None:
                features['wrist_to_face_dist_left'] = np.linalg.norm(nose - left_wrist)
            else:
                features['wrist_to_face_dist_left'] = -1

            if nose is not None and right_wrist is not None:
                features['wrist_to_face_dist_right'] = np.linalg.norm(nose - right_wrist)
            else:
                features['wrist_to_face_dist_right'] = -1

            features['min_wrist_face_dist'] = min(
                features['wrist_to_face_dist_left'] if features['wrist_to_face_dist_left'] > 0 else float('inf'),
                features['wrist_to_face_dist_right'] if features['wrist_to_face_dist_right'] > 0 else float('inf')
            )
            if features['min_wrist_face_dist'] == float('inf'):
                features['min_wrist_face_dist'] = -1

            # 肩幅
            if left_shoulder is not None and right_shoulder is not None:
                features['shoulder_width'] = np.linalg.norm(left_shoulder - right_shoulder)
            else:
                features['shoulder_width'] = -1

            # 動作速度計算（前フレームとの比較）
            if track_id in self.prev_keypoints:
                prev_kp = self.prev_keypoints[track_id]
                movement_diffs = []
                head_movement = 0
                hand_movement = 0

                # 全体的な動き
                for i, (curr, prev) in enumerate(zip(keypoints, prev_kp)):
                    if curr[2] > 0.3 and prev[2] > 0.3:  # 両方とも有効
                        diff = np.linalg.norm(curr[:2] - prev[:2])
                        movement_diffs.append(diff)

                        # 頭部の動き（鼻）
                        if i == kp_idx['nose']:
                            head_movement = diff
                        # 手の動き（手首）
                        elif i in [kp_idx['left_wrist'], kp_idx['right_wrist']]:
                            hand_movement = max(hand_movement, diff)

                features['movement_speed_px_per_frame'] = np.mean(movement_diffs) if movement_diffs else 0
                features['pose_change_magnitude'] = np.sum(movement_diffs) if movement_diffs else 0
                features['head_movement_speed'] = head_movement
                features['hand_movement_speed'] = hand_movement
            else:
                features['movement_speed_px_per_frame'] = 0
                features['pose_change_magnitude'] = 0
                features['head_movement_speed'] = 0
                features['hand_movement_speed'] = 0

            # 現在のキーポイントを保存
            self.prev_keypoints[track_id] = keypoints.copy()

            # 角度計算（簡単な例）
            features['head_pose_angle'] = 0  # TODO: 実装
            features['torso_lean_angle'] = 0  # TODO: 実装
            features['left_arm_angle'] = 0   # TODO: 実装
            features['right_arm_angle'] = 0  # TODO: 実装

        except Exception as e:
            logger.error(f"キーポイント特徴量計算エラー: {e}")
            # デフォルト値で埋める
            features = {
                'wrist_to_face_dist_left': -1, 'wrist_to_face_dist_right': -1,
                'min_wrist_face_dist': -1, 'shoulder_width': -1,
                'movement_speed_px_per_frame': 0, 'pose_change_magnitude': 0,
                'head_movement_speed': 0, 'hand_movement_speed': 0,
                'head_pose_angle': 0, 'torso_lean_angle': 0,
                'left_arm_angle': 0, 'right_arm_angle': 0
            }

        return features

    def calculate_image_quality(self, frame, bbox):
        """画像品質指標を計算"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                return {'brightness': 0, 'contrast': 0, 'blur_score': 0, 'noise_level': 0}

            # グレースケール変換
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) > 2 else roi

            # 明度
            brightness = np.mean(gray_roi)

            # コントラスト（標準偏差）
            contrast = np.std(gray_roi)

            # ブラー検出（ラプラシアン分散）
            blur_score = cv2.Laplacian(gray_roi, cv2.CV_64F).var()

            # ノイズレベル（簡易推定）
            noise_level = np.std(cv2.GaussianBlur(gray_roi, (5,5), 0) - gray_roi)

            return {
                'brightness': brightness,
                'contrast': contrast,
                'blur_score': blur_score,
                'noise_level': noise_level
            }

        except Exception as e:
            logger.error(f"画像品質計算エラー: {e}")
            return {'brightness': 0, 'contrast': 0, 'blur_score': 0, 'noise_level': 0}

    def log_detection(self, frame_idx, track_id, detection_data, frame, 
                    phone_usage, phone_confidence, phone_method,
                    grid_row, grid_col, video_source="unknown"):
        """拡張検出ログを記録"""
        try:
            current_time = time.time()
            timestamp = datetime.now().isoformat()
            relative_time = current_time - self.start_time

            # 基本データ取得
            keypoints = detection_data['keypoints']
            bbox = detection_data['bbox']
            center = detection_data['center']
            confidence = detection_data.get('confidence', 0.0)

            # バウンディングボックス情報
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height

            # キーポイント特徴量計算
            kp_features = self.calculate_keypoint_features(keypoints, track_id)

            # 画像品質計算
            quality_metrics = self.calculate_image_quality(frame, bbox)

            # キーポイント座標をフラット化
            kp_coords = []
            for i in range(17):
                if i < len(keypoints):
                    kp_coords.extend([keypoints[i][0], keypoints[i][1], keypoints[i][2]])
                else:
                    kp_coords.extend([0, 0, 0])  # 欠損値

            # ログデータ準備
            log_data = [
                # 基本情報
                timestamp, frame_idx, relative_time, 33.33,  # 30fps想定

                # 人物識別情報
                track_id, 0.9, confidence,  # track_confidenceは仮値

                # 位置情報
                x1, y1, x2, y2, bbox_width, bbox_height, bbox_area,
                center[0], center[1], grid_row, grid_col,

                # 行動判定
                phone_usage, phone_confidence, phone_usage, phone_method,

                # キーポイント座標
                *kp_coords,

                # 動作特徴量
                kp_features['movement_speed_px_per_frame'],
                kp_features['pose_change_magnitude'],
                kp_features['head_movement_speed'],
                kp_features['hand_movement_speed'],
                kp_features['wrist_to_face_dist_left'],
                kp_features['wrist_to_face_dist_right'],
                kp_features['min_wrist_face_dist'],

                # 姿勢分析
                kp_features['head_pose_angle'],
                kp_features['shoulder_width'],
                kp_features['torso_lean_angle'],
                kp_features['left_arm_angle'],
                kp_features['right_arm_angle'],

                # 画像品質
                quality_metrics['brightness'],
                quality_metrics['contrast'],
                quality_metrics['blur_score'],
                quality_metrics['noise_level'],

                # 時系列特徴（簡易実装）
                0, 0.0, 0.8, 0.9,  # 仮値

                # アノテーション用（空欄）
                "", "", "", 0.0, "", False,

                # メタデータ
                video_source, "v1.0", "yolov8m-pose", ""
            ]

            # CSV書き込み
            if self.csv_writer:
                self.csv_writer.writerow(log_data)

        except Exception as e:
            logger.error(f"ログ記録エラー: {e}")

    def close(self):
        """CSVファイルをクローズ"""
        if self.csv_file:
            self.csv_file.close()
            logger.info(f"拡張CSVログを保存完了: {self.csv_path}")


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


class OrderedIDTracker:
    """左から順にIDを割り振る追跡システム"""

    def __init__(self, distance_threshold=100, max_missing_frames=30):
        self.distance_threshold = distance_threshold
        self.max_missing_frames = max_missing_frames
        self.tracked_persons = {}  # {id: {'center': (x, y), 'missing_count': int, 'bbox': (x1,y1,x2,y2)}}
        self.next_id = 1

    def update_tracks(self, detections):
        """
        検出結果を更新し、左から順にIDを割り振る

        Parameters:
        detections: list of dict with keys: 'center', 'bbox', 'keypoints', 'confidence'

        Returns:
        list of dict with assigned IDs
        """
        if not detections:
            # 検出がない場合、既存の追跡を更新
            self._update_missing_counts()
            return []

        # 検出結果を左から右へソート（x座標順）
        detections_sorted = sorted(detections, key=lambda d: d['center'][0])

        # 既存の追跡対象も左から右へソート
        existing_tracks = sorted(self.tracked_persons.items(), key=lambda t: t[1]['center'][0])

        assigned_detections = []
        used_track_ids = set()

        # 既存の追跡対象とのマッチング
        for detection in detections_sorted:
            best_match_id = None
            best_distance = float('inf')

            detection_center = detection['center']

            # 既存の追跡対象との距離を計算
            for track_id, track_data in existing_tracks:
                if track_id in used_track_ids:
                    continue

                track_center = track_data['center']
                distance = np.linalg.norm(np.array(detection_center) - np.array(track_center))

                if distance < self.distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match_id = track_id

            # マッチした場合は既存IDを使用、そうでなければ新しいIDを割り振り
            if best_match_id is not None:
                assigned_id = best_match_id
                used_track_ids.add(assigned_id)
            else:
                assigned_id = self._get_next_available_id()

            # 追跡データを更新
            self.tracked_persons[assigned_id] = {
                'center': detection_center,
                'missing_count': 0,
                'bbox': detection['bbox']
            }

            # 結果に追加
            detection_with_id = detection.copy()
            detection_with_id['track_id'] = assigned_id
            assigned_detections.append(detection_with_id)

        # 使用されなかった追跡対象のmissing_countを増加
        for track_id in list(self.tracked_persons.keys()):
            if track_id not in used_track_ids:
                self.tracked_persons[track_id]['missing_count'] += 1

                # 長時間見つからない場合は削除
                if self.tracked_persons[track_id]['missing_count'] > self.max_missing_frames:
                    del self.tracked_persons[track_id]
                    logger.debug(f"Track ID {track_id} removed due to long absence")

        return assigned_detections

    def _get_next_available_id(self):
        """次に利用可能なIDを取得（左から順の順序を保つため）"""
        # 既存のIDの中で最小の欠番を探す
        existing_ids = set(self.tracked_persons.keys())

        # 1から順番にチェック
        for i in range(1, max(existing_ids) + 2 if existing_ids else 2):
            if i not in existing_ids:
                return i

        # ここに到達することはないが、念のため
        return max(existing_ids) + 1 if existing_ids else 1

    def _update_missing_counts(self):
        """全ての追跡対象のmissing_countを更新"""
        for track_id in list(self.tracked_persons.keys()):
            self.tracked_persons[track_id]['missing_count'] += 1
            if self.tracked_persons[track_id]['missing_count'] > self.max_missing_frames:
                del self.tracked_persons[track_id]
                logger.debug(f"Track ID {track_id} removed due to long absence")

    def get_active_tracks(self):
        """アクティブな追跡対象を取得"""
        return {tid: data for tid, data in self.tracked_persons.items()}


class PostureDetectionSystem:
    """居眠り検出システムクラス（順序付きID割り振り版）"""

    def __init__(self, model_path="models/yolov8m-pose.pt"):
        self.model = YOLO(model_path)

        # 順序付きIDトラッカーを初期化
        self.id_tracker = OrderedIDTracker(distance_threshold=100, max_missing_frames=30)

        # 人物の状態管理（track_idベース）
        self.person_states = {}

        self.config = {
            "conf_threshold": 0.4,
            "phone_distance_threshold": 100,
            "smoothing_frames": 5,
            "detection_interval": 3,
        }

        # グリッド分割設定を初期化
        self.split_ratios = [0.5, 0.5]  # 上下50%ずつ
        self.split_ratios_cols = [0.5, 0.5]  # 左右50%ずつ

    def smooth_detection(self, track_id, using_phone):
        """検出結果を平滑化（track_idベース）"""
        if track_id not in self.person_states:
            self.person_states[track_id] = {
                "phone_history": deque(maxlen=self.config["smoothing_frames"]),
                "last_seen": 0,
            }

        state = self.person_states[track_id]
        state["phone_history"].append(using_phone)

        # 平滑化：過半数の判定で決定
        if len(state["phone_history"]) == 0:
            return False

        smoothed_phone = sum(state["phone_history"]) > len(state["phone_history"]) // 2
        return smoothed_phone

    def detect_phone_usage(self, keypoints):
        """携帯電話使用を検出（エラーハンドリング強化）"""
        try:
            # キーポイントの基本チェック
            if keypoints is None or keypoints.shape[0] < 17:
                return False

            # 必要なキーポイントのインデックス
            KEYPOINT_INDICES = {
                'nose': 0, 'left_wrist': 9, 'right_wrist': 10,
                'neck': 1, 'left_shoulder': 5, 'right_shoulder': 6
            }

            # キーポイントの取得と検証
            def get_valid_keypoint(idx):
                if idx >= len(keypoints):
                    return None
                pt = keypoints[idx][:2]
                if np.any(np.isnan(pt)) or np.allclose(pt, [0, 0]):
                    return None
                return pt

            nose = get_valid_keypoint(KEYPOINT_INDICES['nose'])
            left_wrist = get_valid_keypoint(KEYPOINT_INDICES['left_wrist'])
            right_wrist = get_valid_keypoint(KEYPOINT_INDICES['right_wrist'])

            # 基本的な携帯使用判定
            if all(pt is not None for pt in [nose, left_wrist, right_wrist]):
                threshold = self.config["phone_distance_threshold"]
                dist_left = np.linalg.norm(nose - left_wrist)
                dist_right = np.linalg.norm(nose - right_wrist)

                if min(dist_left, dist_right) < threshold:
                    return True

            # 背面判定にフォールバック
            return self.detect_phone_usage_back_view(keypoints)

        except Exception as e:
            logger.error(f"携帯使用判定エラー: {e}")
            return False

    def detect_phone_usage_back_view(self, keypoints):
        """背面からの携帯使用検出（エラーハンドリング追加）"""
        try:
            threshold = 60

            # 必要なキーポイントのインデックス
            indices = {'neck': 1, 'left_wrist': 9, 'right_wrist': 10, 
                    'left_shoulder': 5, 'right_shoulder': 6}

            # キーポイントの取得
            points = {}
            for name, idx in indices.items():
                if idx >= len(keypoints):
                    return False
                pt = keypoints[idx][:2]
                if np.any(np.isnan(pt)) or np.allclose(pt, [0, 0]):
                    return False
                points[name] = pt

            # 距離計算
            dist_left = min(
                np.linalg.norm(points['left_wrist'] - points['neck']),
                np.linalg.norm(points['left_wrist'] - points['left_shoulder'])
            )
            dist_right = min(
                np.linalg.norm(points['right_wrist'] - points['neck']),
                np.linalg.norm(points['right_wrist'] - points['right_shoulder'])
            )

            return min(dist_left, dist_right) < threshold

        except Exception as e:
            logger.error(f"背面携帯使用検出エラー: {e}")
            return False

    def draw_monitor_grid(self, img, col_ratios, row_ratios):
        """監視グリッドを描画"""
        h, w = img.shape[:2]
        # 縦線
        x_current = 0
        for ratio in col_ratios[:-1]:
            x_current += int(w * ratio)
            cv2.line(img, (x_current, 0), (x_current, h), (255, 255, 255), 2)
        # 横線
        y_current = 0
        for ratio in row_ratios[:-1]:
            y_current += int(h * ratio)
            cv2.line(img, (0, y_current), (w, y_current), (255, 255, 255), 2)

    def calculate_grid_boundaries(self, w, h, cols, rows):
        """グリッド境界を計算"""
        x_grid = [0]
        for ratio in cols:
            x_grid.append(x_grid[-1] + int(w * ratio))
        y_grid = [0]
        for ratio in rows:
            y_grid.append(y_grid[-1] + int(h * ratio))
        return x_grid, y_grid

    def get_person_region(self, cx, cy, x_grid, y_grid):
        """人物の位置するグリッド領域を取得"""
        col_idx = row_idx = -1
        for i in range(len(x_grid) - 1):
            if x_grid[i] <= cx < x_grid[i + 1]:
                col_idx = i
                break
        for j in range(len(y_grid) - 1):
            if y_grid[j] <= cy < y_grid[j + 1]:
                row_idx = j
                break
        if col_idx == -1 or row_idx == -1:
            return None
        return row_idx, col_idx

    def process_frame(self, frame, frame_idx, csv_writer, enhanced_csv_logger=None):
        """
        フレームを処理して検出結果を返す（順序付きID割り振り版 + 拡張CSV対応）
        """
        height, width = frame.shape[:2]

        # YOLO検出実行（追跡なし）
        try:
            results = self.model(frame, conf=self.config["conf_threshold"], verbose=False)
        except Exception as e:
            logger.error(f"YOLO検出エラー: {e}")
            return frame

        # グリッド境界計算
        x_grid, y_grid = self.calculate_grid_boundaries(
            width, height, self.split_ratios_cols, self.split_ratios
        )

        # 検出結果を整理
        detections = []
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue

            try:
                kps_list = result.keypoints.xy.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for kps, box, conf in zip(kps_list, boxes, confs):
                    if conf < self.config["conf_threshold"]:
                        continue

                    if kps.shape[0] < 17:
                        continue

                    # 中心座標計算
                    valid_points = kps[kps[:, 0] > 0]
                    if len(valid_points) == 0:
                        continue
                    center = np.mean(valid_points[:, :2], axis=0)
                    cx, cy = int(center[0]), int(center[1])

                    detections.append({
                        'center': (cx, cy),
                        'bbox': box,
                        'keypoints': kps,
                        'confidence': conf
                    })

            except Exception as e:
                logger.error(f"検出結果処理エラー: {e}")
                continue

        # 順序付きIDトラッカーで追跡更新
        tracked_detections = self.id_tracker.update_tracks(detections)

        # グリッド描画
        self.draw_monitor_grid(frame, self.split_ratios_cols, self.split_ratios)

        # 検出結果の描画
        for detection in tracked_detections:
            track_id = detection['track_id']
            kps = detection['keypoints']
            box = detection['bbox']
            cx, cy = detection['center']

            # 携帯使用検出
            using_phone = self.detect_phone_usage(kps)
            using_phone_raw = using_phone  # 生の判定結果を保存
            using_phone = self.smooth_detection(track_id, using_phone)

            # 検出方法の判定
            phone_method = "front_view" if using_phone_raw else "back_view" if self.detect_phone_usage_back_view(kps) else "failed"
            phone_confidence = 0.8 if using_phone else 0.2  # 簡易的な信頼度

            # 領域の取得
            region = self.get_person_region(cx, cy, x_grid, y_grid)
            row, col = region if region else (-1, -1)

            # 基本CSVに結果を記録
            if csv_writer:
                csv_writer.writerow([frame_idx, track_id, using_phone, row, col])

            # 拡張CSVに結果を記録
            if enhanced_csv_logger:
                detection_data = {
                    'keypoints': kps,
                    'bbox': box,
                    'center': (cx, cy),
                    'confidence': detection['confidence']
                }
                enhanced_csv_logger.log_detection(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    detection_data=detection_data,
                    frame=frame,
                    phone_usage=using_phone,
                    phone_confidence=phone_confidence,
                    phone_method=phone_method,
                    grid_row=row,
                    grid_col=col,
                    video_source="google_drive"
                )

            # 状態表示
            if using_phone:
                color = (0, 0, 255)  # 赤色
                label = f"ID: {track_id}: Phone"
            else:
                color = (255, 255, 0)  # 黄色
                label = f"ID: {track_id}: Awake"

            if region:
                row, col = region
                label += f" [R{row},C{col}]"

            # バウンディングボックス描画
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

            # キーポイント描画
            for pt in kps.astype(int):
                if len(pt) >= 2 and pt[0] > 0 and pt[1] > 0:
                    cv2.circle(frame, tuple(pt[:2]), 3, (255, 255, 0), -1)

        # デバッグ情報の表示
        active_tracks = self.id_tracker.get_active_tracks()
        active_persons = len(active_tracks)

        # アクティブなIDを左から順にソートして表示
        active_ids = sorted(active_tracks.keys())
        id_info = f"Active IDs (L→R): {active_ids}" if active_ids else "Active IDs: None"

        cv2.putText(
            frame,
            id_info,
            (20, height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Total Persons: {active_persons}",
            (20, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return frame


class IntegratedVideoProcessor:
    """統合動画処理システム（逆バレル補正版 + 順序付きID割り振り）"""

    def __init__(self, k1=-0.1, strength=1.0, zoom_factor=0.8, model_path="yolov8m-pose.pt"):
        self.corrector = VideoDistortionCorrector(k1, strength, zoom_factor)
        self.detector = PostureDetectionSystem(model_path)

        # 拡張CSVロガーを初期化（オプション）
        self.csv_logger = None

    def set_csv_logger(self, csv_path="enhanced_detection_log.csv"):
        """CSVロガーを設定"""
        self.csv_logger = EnhancedCSVLogger(csv_path)

    def process_video(self, input_path, output_path, result_log="frame_results.csv", 
                    show_preview=True, apply_correction=True):
        """
        動画を処理（逆バレル歪み補正 + 居眠り検出 + 順序付きID割り振り）

        Parameters:
        input_path: 入力動画パス
        output_path: 出力動画パス
        result_log: 結果ログのCSVファイル名
        show_preview: プレビュー表示するか
        apply_correction: 歪み補正を適用するか
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"動画ファイルを開けません: {input_path}")
            return

        # 動画情報取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"動画情報: {width}x{height}, {fps:.1f}fps, {total_frames}フレーム")

        # 歪み補正マップ作成
        if apply_correction:
            logger.info("逆バレル歪み補正＋ズーム調整マップを作成中...")
            self.corrector.create_correction_maps(width, height)
            logger.info(f"歪み係数: {self.corrector.k1 * self.corrector.strength:.4f}")
            logger.info(f"ズーム倍率: {self.corrector.zoom_factor:.2f}x ({'引き' if self.corrector.zoom_factor < 1.0 else '寄り'})")

        # 出力動画設定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 結果ログ準備
        os.makedirs(os.path.dirname(result_log), exist_ok=True)
        
        with open(result_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "person_id", "using_phone", "grid_row", "grid_col"])

            frame_idx = 0
            start_time = time.time()

            logger.info("動画処理を開始... (左から順ID割り振りシステム使用)")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # 逆バレル歪み補正適用
                if apply_correction:
                    frame = self.corrector.apply_correction(frame)

                # 居眠り検出処理（順序付きID割り振り使用）
                frame = self.detector.process_frame(frame, frame_idx, writer)

                # フレーム番号表示
                cv2.putText(
                    frame,
                    f"Frame: {frame_idx}/{total_frames}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )

                # プレビュー表示
                if show_preview:
                    display_title = "Integrated Video Processing (Ordered ID Assignment L→R)"
                    if apply_correction:
                        display_title += f" - Zoom: {self.corrector.zoom_factor:.2f}x"
                    display_title += " - Press 'q' to quit"

                    cv2.imshow(display_title, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("ユーザーによって処理が中断されました")
                        break

                # 結果保存
                out.write(frame)

                # 進行状況表示
                if frame_idx % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_idx / elapsed
                    progress = (frame_idx / total_frames) * 100
                    eta = (total_frames - frame_idx) / fps_current if fps_current > 0 else 0

                    active_tracks = self.detector.id_tracker.get_active_tracks()
                    logger.info(
                        f"進行状況: {progress:.1f}% ({frame_idx}/{total_frames}) "
                        f"処理速度: {fps_current:.1f}fps 残り時間: {eta:.1f}秒 "
                        f"アクティブID: {sorted(active_tracks.keys())}"
                    )

        # リソース解放
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # CSVロガーをクローズ
        if self.csv_logger:
            self.csv_logger.close()

        # 処理完了時間
        total_time = time.time() - start_time
        logger.info(f"動画処理完了!")
        logger.info(f"出力ファイル: {output_path}")
        logger.info(f"結果ログ: {result_log}")
        logger.info(f"処理時間: {total_time:.1f}秒")
        logger.info(f"平均処理速度: {frame_idx/total_time:.1f}fps")

        active_tracks = self.detector.id_tracker.get_active_tracks()
        logger.info(f"最終アクティブID: {sorted(active_tracks.keys())}")

    def process_image(self, image_path, output_dir="output", show_comparison=True):
        """
        画像処理（逆バレル歪み補正のみ）

        Parameters:
        image_path: 入力画像パス
        output_dir: 出力ディレクトリ
        show_comparison: 比較表示するか
        """
        # 画像読み込み
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"画像が読み込めません: {image_path}")
            return

        h, w = image.shape[:2]
        logger.info(f"画像を読み込み: {image_path}")
        logger.info(f"画像サイズ: {w}x{h}")

        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)

        # 逆バレル歪み補正マップ作成
        self.corrector.create_correction_maps(w, h)

        # 逆バレル歪み補正適用
        logger.info(f"逆バレル歪み補正＋ズーム調整を適用中... (ズーム: {self.corrector.zoom_factor:.2f}x)")
        corrected = self.corrector.apply_correction(image)

        # 結果保存
        output_path = os.path.join(output_dir, f'corrected_inverse_barrel_zoom_{self.corrector.zoom_factor:.2f}x.png')
        cv2.imwrite(output_path, corrected)
        logger.info(f"補正完了: {output_path}")

        # 比較表示
        if show_comparison:
            self._create_comparison_plot(image, corrected, output_dir)

        return corrected

    def _create_comparison_plot(self, original, corrected, output_dir):
        """比較画像を作成"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # BGR→RGB変換
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

        # 画像表示
        axes[0].imshow(original_rgb)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')

        axes[1].imshow(corrected_rgb)
        axes[1].set_title(f'Inverse Barrel Corrected (Zoom: {self.corrector.zoom_factor:.2f}x)', fontsize=14)
        axes[1].axis('off')

        plt.tight_layout()
        # 比較画像保存
        comparison_path = os.path.join(output_dir, f'comparison_inverse_barrel_zoom_{self.corrector.zoom_factor:.2f}x.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.show()

        logger.info(f"比較画像を保存: {comparison_path}")