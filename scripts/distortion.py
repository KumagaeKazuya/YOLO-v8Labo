import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import hashlib
import json
from ultralytics import YOLO
from collections import defaultdict, deque
import logging
import csv
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class VideoProcessingHistory:
    """動画処理履歴管理クラス"""

    def __init__(self, history_file="data/video_processing_history.json"):
        self.history_file = history_file
        self.history = self.load_history()

    def load_history(self):
        """処理履歴を読み込む"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"処理履歴読み込みエラー: {e}")
            return {}

    def save_history(self):
        """処理履歴を保存"""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"処理履歴保存エラー: {e}")

    def get_video_hash(self, video_path):
        """動画ファイルのハッシュ値を計算"""
        try:
            stat = os.stat(video_path)
            hash_data = {
                "basename": os.path.splitext(os.path.basename(video_path))[0],
                "size": stat.st_size,
                "mtime": stat.st_mtime
            }
            hash_str = json.dumps(hash_data, sort_keys=True)
            return hashlib.md5(hash_str.encode()).hexdigest()[:16]
        except Exception as e:
            logger.error(f"動画ハッシュ計算エラー: {e}")
            return None

    def get_processing_record(self, video_hash):
        """動画の処理記録を取得"""
        return self.history.get(video_hash, None)

    def should_overwrite(self, video_hash, estimated_processing_time, quality_threshold=1.0):
        """上書きすべきかを判定（品質劣化防止機能強化）"""
        record = self.get_processing_record(video_hash)

        if record is None:
            return True, "初回処理"

        previous_time = record.get("processing_time", 0)

        # 推定時間が0以下の場合は安全のため処理実行
        if estimated_processing_time <= 0:
            return True, "推定時間が無効のため安全処理"

        # 前回の処理時間が0以下の場合は処理実行
        if previous_time <= 0:
            return True, "前回処理時間が無効のため再処理"

        # 推定時間と前回処理時間の比較
        time_ratio = estimated_processing_time / previous_time

        # 推定時間が前回より長い場合（処理品質向上の可能性）
        if time_ratio >= quality_threshold:
            improvement_ratio = time_ratio
            return True, f"処理品質向上期待 (推定時間が{improvement_ratio:.1f}倍に増加)"

        # 推定時間が前回より短い場合（処理品質劣化の可能性）
        else:
            degradation_ratio = previous_time / estimated_processing_time
            return False, f"処理品質劣化防止 (推定時間が{degradation_ratio:.1f}倍高速化→品質低下の可能性)"


    def add_processing_record(self, video_hash, video_basename, processing_time, total_frames, output_files):
        """処理記録を追加"""
        record = {
            "video_basename": video_basename,
            "processing_time": processing_time,
            "total_frames": total_frames,
            "fps_average": total_frames / processing_time if processing_time > 0 else 0,
            "timestamp": time.time(),
            "execution_count": self.history.get(video_hash, {}).get("execution_count", 0) + 1,
            "output_files": output_files
        }
        self.history[video_hash] = record
        self.save_history()
        return record

class SmartFileManager:
    """スマートファイル管理クラス"""

    def __init__(self, base_output_dir="videos", base_data_dir="data", quality_threshold=1.0):
        self.base_output_dir = base_output_dir
        self.base_data_dir = base_data_dir
        self.video_history = VideoProcessingHistory()
        self.quality_threshold = quality_threshold  # 品質劣化防止閾値

    def generate_output_paths(self, video_path):
        """動画に基づいて出力パスを生成"""
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        video_hash = self.video_history.get_video_hash(video_path)

        if video_hash is None:
            # ハッシュ計算に失敗した場合は基本ファイル名のみ使用
            suffix = ""
            logger.warning("動画ハッシュ計算失敗、基本ファイル名のみ使用")
        else:
            suffix = f"_{video_hash}"

        # 出力ファイルパス生成（動画別に分離）
        output_video = os.path.join(
            self.base_output_dir,
            f"output_{video_basename}{suffix}_advanced_posture_detection.mp4"
        )
        basic_csv = os.path.join(
            self.base_data_dir,
            f"frame_results_{video_basename}{suffix}.csv"
        )
        enhanced_csv = os.path.join(
            self.base_data_dir,
            f"enhanced_detection_log_{video_basename}{suffix}.csv"
        )

        return {
            "output_video": output_video,
            "basic_csv": basic_csv,
            "enhanced_csv": enhanced_csv,
            "video_hash": video_hash,
            "video_basename": video_basename
        }

    def check_existing_files(self, file_paths):
        """既存ファイルの存在をチェック"""
        existing_files = {}
        for key, path in file_paths.items():
            if key not in ["video_hash", "video_basename"] and os.path.exists(path):
                existing_files[key] = {
                    "path": path,
                    "size": os.path.getsize(path),
                    "mtime": os.path.getmtime(path)
                }
        return existing_files

    def should_process_video(self, video_path, estimated_processing_time, force_process=False):
        """動画を処理すべきかを判定"""
        if force_process:
            return True, "強制処理モード", {}

        file_paths = self.generate_output_paths(video_path)
        existing_files = self.check_existing_files(file_paths)

        if not existing_files:
            return True, "初回処理（ファイルなし）", file_paths

        video_hash = file_paths["video_hash"]
        if video_hash is None:
            return True, "ハッシュ計算失敗（安全のため処理実行）", file_paths

        # 品質劣化防止閾値を使用して判定
        should_overwrite, reason = self.video_history.should_overwrite(
            video_hash, estimated_processing_time, self.quality_threshold
        )

        if should_overwrite:
            return True, reason, file_paths
        else:
            return False, reason, file_paths

    def cleanup_existing_files(self, file_paths):
        """既存ファイルを削除（品質向上時のみ - CSV保護機能付き）"""
        logger.info("🔄 品質向上のため既存ファイルを削除中...")

        deleted_files = []
        for key, path in file_paths.items():
            if key not in ["video_hash", "video_basename"] and os.path.exists(path):
                try:
                    # ファイル情報を記録
                    file_size = os.path.getsize(path)
                    size_mb = file_size / (1024 * 1024)

                    os.remove(path)
                    deleted_files.append((key, os.path.basename(path), size_mb))
                    logger.info(f"品質向上のため削除: {os.path.basename(path)} ({size_mb:.1f}MB)")
                except Exception as e:
                    logger.error(f"ファイル削除エラー {path}: {e}")

        if deleted_files:
            logger.info(f"📁 削除完了: {len(deleted_files)}ファイル")
            for key, filename, size_mb in deleted_files:
                logger.info(f"  {key}: {filename} ({size_mb:.1f}MB)")
        else:
            logger.info("削除対象ファイルなし")

    def log_processing_completion(self, video_path, file_paths, processing_time, total_frames):
        """処理完了をログに記録"""
        video_hash = file_paths["video_hash"]
        video_basename = file_paths["video_basename"]

        if video_hash is None:
            logger.warning("動画ハッシュなしのため履歴記録をスキップ")
            return

        output_files = {
            key: path for key, path in file_paths.items()
            if key not in ["video_hash", "video_basename"]
        }

        record = self.video_history.add_processing_record(
            video_hash, video_basename, processing_time, total_frames, output_files
        )

        # 処理品質情報をログ出力
        fps_average = record.get("fps_average", 0)
        execution_count = record.get("execution_count", 1)

        logger.info(f"📊 処理品質記録更新:")
        logger.info(f"  実行回数: {execution_count}回目")
        logger.info(f"  処理時間: {processing_time:.1f}秒")
        logger.info(f"  平均FPS: {fps_average:.1f}")
        logger.info(f"  総フレーム数: {total_frames}")

class PhoneUsageState(Enum):
    """スマホ使用状態の詳細分類"""
    NOT_USING = "not_using"
    HOLDING_NEAR_FACE = "holding_near_face"
    LOOKING_DOWN = "looking_down"
    BOTH_HANDS_UP = "both_hands_up"
    UNCERTAIN = "uncertain"
    TRANSITIONING = "transitioning"

class PersonOrientation(Enum):
    """人物の向きの分類"""
    FRONT_FACING = "front_facing"
    BACK_FACING = "back_facing"
    SIDE_FACING = "side_facing"
    UNCERTAIN = "uncertain"

@dataclass
class PostureFeatures:
    """姿勢特徴量を格納するデータクラス"""
    head_angle: float
    hand_face_distance_left: float
    hand_face_distance_right: float
    shoulder_hand_angle_left: float
    shoulder_hand_angle_right: float
    head_tilt: float
    neck_forward: float
    confidence_score: float
    visible_keypoints: int
    orientation: PersonOrientation = PersonOrientation.UNCERTAIN

@dataclass
class DetectionResult:
    """検出結果の詳細情報"""
    frame_id: int
    track_id: int
    timestamp: float
    phone_state: PhoneUsageState
    confidence: float
    features: PostureFeatures
    bbox: Tuple[int, int, int, int]
    grid_position: Tuple[int, int]
    keypoints_visible: List[bool]
    orientation: PersonOrientation


class EnhancedCSVLogger:
    """機械学習用拡張CSVロガー"""

    def __init__(self, csv_path="enhanced_detection_log.csv", model_name="yolo11x-pose", overwrite_existing=True):
        self.csv_path = csv_path
        self.csv_file = None
        self.csv_writer = None
        self.start_time = time.time()
        self.log_count = 0
        # モデルを変更した時はここを更新
        self.model_name = model_name
        # self.file_manager = SmartFileManager()  # スマートファイル管理を追加
        self.overwrite_existing = overwrite_existing  # ← この行を追加

        # CSV ヘッダー定義
        self.headers = [
            # 基本情報
            "timestamp", "frame_idx", "relative_time_sec", "frame_interval_ms",

            # 人物識別情報
            "tracking_id",

            # 検出枠座標
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",

            # 検出枠信頼度
            "detection_confidence",

            # キーポイント座標と信頼度 (17点 x 3座標 = 51列)
            *[f"kp_{i}_{coord}" for i in range(17) for coord in ["x", "y", "conf"]],

            # モデル情報（キーポイント座標の後）
            "model_name"
        ]

        self.init_csv()

    def init_csv(self):
        """CSV ファイルを初期化"""
        try:
            # ディレクトリ作成
            csv_dir = os.path.dirname(self.csv_path)
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)

            # 既存ファイルのチェック
            if os.path.exists(self.csv_path) and not self.overwrite_existing:
                file_size = os.path.getsize(self.csv_path)
                mod_time = os.path.getmtime(self.csv_path)
                mod_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod_time))

                logger.warning(f"拡張CSVファイルが既に存在: {os.path.basename(self.csv_path)}")
                logger.warning(f"サイズ: {file_size:,} bytes, 更新: {mod_time_str}")
                logger.warning("既存ファイルを保持します（重複防止）")

                # ダミーファイルハンドルを作成（書き込みは無効化）
                self.csv_file = None
                self.csv_writer = None
                return

            # 新規作成または上書き
            self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(self.headers)
            self.csv_file.flush()

            logger.info(f"拡張CSVログ初期化: {os.path.basename(self.csv_path)}")
            logger.info(f"CSVヘッダー列数: {len(self.headers)}")

        except Exception as e:
            logger.error(f"CSV初期化エラー: {e}")
            raise

    def log_detection_result_simplified(self, detection_result: DetectionResult, keypoints, yolo_confidence=0.0):
        """DetectionResultオブジェクトから拡張CSVにログを記録"""
        # CSVライターが無効化されている場合はスキップ
        if self.csv_writer is None:
            return

        try:
            self.log_count += 1
            current_time = time.time()
            timestamp = datetime.now().isoformat()
            relative_time = current_time - self.start_time

            # バウンディングボックス情報
            x1, y1, x2, y2 = detection_result.bbox

            # キーポイント座標をフラット化
            kp_coords = []
            for i in range(17):
                if i < len(keypoints):
                    x = float(keypoints[i][0]) if len(keypoints[i]) > 0 else 0.0
                    y = float(keypoints[i][1]) if len(keypoints[i]) > 1 else 0.0
                    conf = float(keypoints[i][2]) if len(keypoints[i]) > 2 else 0.0
                    kp_coords.extend([x, y, conf])
                else:
                    kp_coords.extend([0.0, 0.0, 0.0])

            # ログデータ準備
            log_data = [
                # 基本情報
                timestamp, detection_result.frame_id, relative_time, 33.33,

                # トラッキングID
                detection_result.track_id,

                # 検出枠座標
                float(x1), float(y1), float(x2), float(y2),

                # 検出枠信頼度
                yolo_confidence,

                # キーポイント座標と信頼度(51列)
                *kp_coords,

                # モデル情報 (キーポイント座標の後)
                self.model_name,
            ]

            # データ長調整
            while len(log_data) < len(self.headers):
                log_data.append("")
            log_data = log_data[:len(self.headers)]

            # CSV書き込み
            if self.csv_writer:
                self.csv_writer.writerow(log_data)
                self.csv_file.flush()

                if self.log_count % 30 == 0:
                    logger.info(f"CSV記録継続: {self.log_count}行目")

        except Exception as e:
            logger.error(f"拡張ログ記録エラー: {e}")

    def close(self):
        """CSVファイルをクローズ"""
        try:
            if self.csv_file:
                self.csv_file.flush()
                self.csv_file.close()
                logger.info(f"CSVログ保存完了: {os.path.basename(self.csv_path)}")
                logger.info(f"総記録数: {self.log_count}行")
            elif self.csv_path and os.path.exists(self.csv_path):
                logger.info(f"既存CSVファイルを保持: {os.path.basename(self.csv_path)}")
        except Exception as e:
            logger.error(f"ログクローズエラー: {e}")

class VideoDistortionCorrector:
    """動画の歪み補正クラス（改良版 - yolo_checker.py準拠）"""

    def __init__(self, k1=-0.1, k2=0.0, p1=0.0, p2=0.0, k3=0.0, alpha=0.6, focal_scale=0.9):
        """
        歪み補正パラメータを初期化

        Args:
            k1, k2, k3: 放射歪み係数
            p1, p2: 接線歪み係数
            alpha: 新しいカメラマトリックスのスケーリング
            focal_scale: 焦点距離のスケーリング
        """
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.k3 = k3
        self.alpha = alpha
        self.focal_scale = focal_scale
        self.map_x = None
        self.map_y = None
        self.original_camera_matrix = None
        self.new_camera_matrix = None
        self.dist_coeffs = None

        logger.info(f"歪み補正初期化（改良版）:")
        logger.info(f"  k1={k1}, k2={k2}, k3={k3}")
        logger.info(f"  p1={p1}, p2={p2}")
        logger.info(f"  alpha={alpha}, focal_scale={focal_scale}")

    def create_correction_maps(self, width, height):
        """歪み補正用のマップを作成（改良版）"""
        logger.info(f"高精度歪み補正マップ作成開始: {width}x{height}")

        # カメラ内部パラメータの設定
        fx = fy = width * self.focal_scale
        cx, cy = width / 2.0, height / 2.0

        self.original_camera_matrix = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
        )

        # 5つの歪み係数を使用（高精度補正）
        self.dist_coeffs = np.array(
            [self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32
        )

        # 最適な新しいカメラマトリックスを計算
        self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.original_camera_matrix,
            self.dist_coeffs,
            (width, height),
            self.alpha,
            (width, height),
        )

        # 補正マップ作成
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.original_camera_matrix,
            self.dist_coeffs,
            None,
            self.new_camera_matrix,
            (width, height),
            cv2.CV_32FC1,
        )

        logger.info("高精度歪み補正マップ作成完了")
        self._log_map_statistics()

    def _log_map_statistics(self):
        """マップの統計情報をログ出力"""
        if self.map_x is not None and self.map_y is not None:
            x_mean, x_std = np.mean(self.map_x), np.std(self.map_x)
            y_mean, y_std = np.mean(self.map_y), np.std(self.map_y)
            logger.info(f"補正マップ統計: X(平均={x_mean:.2f}, 標準偏差={x_std:.2f}), Y(平均={y_mean:.2f}, 標準偏差={y_std:.2f})")

    def apply_correction(self, frame):
        """フレームに歪み補正を適用"""
        if self.map_x is None or self.map_y is None:
            logger.warning("補正マップが作成されていません")
            return frame

        return cv2.remap(
            frame, self.map_x, self.map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

class OrderedIDTracker:
    """左から順にIDを割り振る追跡システム（改良版）"""

    def __init__(self, distance_threshold=100, max_missing_frames=30):
        self.distance_threshold = distance_threshold
        self.max_missing_frames = max_missing_frames
        self.tracked_persons = {}
        self.next_id = 1

    def update_tracks(self, detections):
        """検出結果を更新し、左から順にIDを割り振る"""
        if not detections:
            self._update_missing_counts()
            return []

        detections_sorted = sorted(detections, key=lambda d: d['center'][0])
        existing_tracks = sorted(self.tracked_persons.items(), key=lambda t: t[1]['center'][0])

        assigned_detections = []
        used_track_ids = set()

        for detection in detections_sorted:
            best_match_id = None
            best_distance = float('inf')
            detection_center = detection['center']

            for track_id, track_data in existing_tracks:
                if track_id in used_track_ids:
                    continue

                track_center = track_data['center']
                distance = np.linalg.norm(np.array(detection_center) - np.array(track_center))

                if distance < self.distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match_id = track_id

            if best_match_id is not None:
                assigned_id = best_match_id
                used_track_ids.add(assigned_id)
            else:
                assigned_id = self._get_next_available_id()

            self.tracked_persons[assigned_id] = {
                'center': detection_center,
                'missing_count': 0,
                'bbox': detection['bbox']
            }

            detection_with_id = detection.copy()
            detection_with_id['track_id'] = assigned_id
            assigned_detections.append(detection_with_id)

        # 未使用の追跡対象を更新
        for track_id in list(self.tracked_persons.keys()):
            if track_id not in used_track_ids:
                self.tracked_persons[track_id]['missing_count'] += 1
                if self.tracked_persons[track_id]['missing_count'] > self.max_missing_frames:
                    del self.tracked_persons[track_id]

        return assigned_detections

    def _get_next_available_id(self):
        """次に利用可能なIDを取得"""
        existing_ids = set(self.tracked_persons.keys())
        for i in range(1, max(existing_ids) + 2 if existing_ids else 2):
            if i not in existing_ids:
                return i
        return max(existing_ids) + 1 if existing_ids else 1

    def _update_missing_counts(self):
        """全ての追跡対象のmissing_countを更新"""
        for track_id in list(self.tracked_persons.keys()):
            self.tracked_persons[track_id]['missing_count'] += 1
            if self.tracked_persons[track_id]['missing_count'] > self.max_missing_frames:
                del self.tracked_persons[track_id]

    def get_active_tracks(self):
        """アクティブな追跡対象を取得"""
        return {tid: data for tid, data in self.tracked_persons.items()}

class AdvancedPostureDetectionSystem:
    """高度な姿勢検出システム（yolo_checker.py準拠）"""

    def __init__(self, model_path="models/yolo11x-pose.pt"):
        self.model = YOLO(model_path)
        self.id_tracker = OrderedIDTracker(distance_threshold=100, max_missing_frames=30)
        self.person_states = {}

        # 基本設定
        self.config = {
            "conf_threshold": 0.4,
            "phone_distance_threshold": 100,
            "head_angle_threshold": 30,
            "neck_forward_threshold": 0.2,
        }

        # 後ろ向き用の設定
        self.back_view_config = {
            "hand_head_distance_threshold": 150,
            "arm_bend_threshold": 90,
            "shoulder_tilt_threshold": 15,
            "forward_lean_threshold": 0.3,
        }

        # スケルトン描画用の接続情報
        self.skeleton_connections = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # 顔周り
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],         # 胴体・腕
            [8, 10], [9, 11], [12, 14], [13, 15], [14, 16], [15, 17]  # 腕・脚
        ]

        # グリッド分割設定
        self.split_ratios = [0.5, 0.5]
        self.split_ratios_cols = [0.5, 0.5]

        # 追跡用の履歴
        self.track_history = defaultdict(lambda: deque(maxlen=10))
        self.frame_count = 0

        # FPS計算用
        self.fps_counter = deque(maxlen=30)  # 過去30フレームの平均FPSを計算
        self.last_frame_time = time.time()

    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int]) -> None:
        """スケルトン（骨格）を描画"""
        try:
            keypoints_2d = keypoints[:, :2] if keypoints.shape[1] >= 2 else keypoints

            # スケルトン線を描画
            for connection in self.skeleton_connections:
                pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1  # 1-indexed to 0-indexed

                if pt1_idx < len(keypoints_2d) and pt2_idx < len(keypoints_2d):
                    pt1 = keypoints_2d[pt1_idx]
                    pt2 = keypoints_2d[pt2_idx]

                    # 両方のキーポイントが有効な場合のみ線を描画
                    if (pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0):
                        cv2.line(frame, (int(pt1[0]), int(pt1[1])),
                                (int(pt2[0]), int(pt2[1])), color, 2)

        except Exception as e:
            logger.warning(f"スケルトン描画エラー: {e}")

    def determine_person_orientation(self, keypoints: np.ndarray) -> PersonOrientation:
        """人物の向きを判定"""
        try:
            if keypoints.shape[0] < 17:
                return PersonOrientation.UNCERTAIN

            # 顔のキーポイント
            nose = keypoints[0][:2]
            left_eye = keypoints[1][:2]
            right_eye = keypoints[2][:2]
            left_ear = keypoints[3][:2]
            right_ear = keypoints[4][:2]

            # 肩のキーポイント
            left_shoulder = keypoints[5][:2]
            right_shoulder = keypoints[6][:2]

            # 可視性をチェック
            face_points_visible = sum(
                1 for p in [nose, left_eye, right_eye, left_ear, right_ear]
                if not np.allclose(p, [0, 0])
            )

            shoulder_points_visible = sum(
                1 for p in [left_shoulder, right_shoulder]
                if not np.allclose(p, [0, 0])
            )

            # 判定ロジック
            if face_points_visible >= 3:
                return PersonOrientation.FRONT_FACING
            elif shoulder_points_visible >= 2 and face_points_visible <= 1:
                return PersonOrientation.BACK_FACING
            elif face_points_visible >= 1 and shoulder_points_visible >= 1:
                return PersonOrientation.SIDE_FACING
            else:
                return PersonOrientation.UNCERTAIN

        except Exception as e:
            logger.error(f"向き判定エラー: {e}")
            return PersonOrientation.UNCERTAIN

    def extract_advanced_features(self, keypoints: np.ndarray) -> Optional[PostureFeatures]:
        """向きを自動判定して適切な特徴量を抽出"""
        orientation = self.determine_person_orientation(keypoints)

        if orientation == PersonOrientation.FRONT_FACING:
            return self._extract_features_front_view(keypoints)
        elif orientation == PersonOrientation.BACK_FACING:
            return self._extract_features_back_view(keypoints)
        else:
            features = self._extract_features_front_view(keypoints)
            if features:
                features.orientation = orientation
            return features

    def _extract_features_front_view(self, keypoints: np.ndarray) -> Optional[PostureFeatures]:
        """前向き映像用の特徴量抽出"""
        try:
            if keypoints.shape[0] < 17:
                return None

            # キーポイントの定義
            nose = keypoints[0][:2]
            left_ear = keypoints[3][:2]
            right_ear = keypoints[4][:2]
            left_shoulder = keypoints[5][:2]
            right_shoulder = keypoints[6][:2]
            left_elbow = keypoints[7][:2]
            right_elbow = keypoints[8][:2]
            left_wrist = keypoints[9][:2]
            right_wrist = keypoints[10][:2]

            visible_count = sum(1 for kp in keypoints if kp[0] > 0 and kp[1] > 0)

            def safe_distance(p1, p2):
                if (np.any(np.isnan(p1)) or np.any(np.isnan(p2)) or
                    np.allclose(p1, [0, 0]) or np.allclose(p2, [0, 0])):
                    return float("inf")
                return np.linalg.norm(p1 - p2)

            def calculate_angle(p1, p2, p3):
                if any(np.allclose(p, [0, 0]) for p in [p1, p2, p3]):
                    return 0.0
                v1 = p1 - p2
                v2 = p3 - p2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                return np.degrees(np.arccos(cos_angle))

            features = PostureFeatures(
                head_angle=calculate_angle(left_ear, nose, right_ear),
                hand_face_distance_left=safe_distance(left_wrist, nose),
                hand_face_distance_right=safe_distance(right_wrist, nose),
                shoulder_hand_angle_left=calculate_angle(left_shoulder, left_elbow, left_wrist),
                shoulder_hand_angle_right=calculate_angle(right_shoulder, right_elbow, right_wrist),
                head_tilt=self._calculate_head_tilt(left_ear, right_ear),
                neck_forward=0.0,  # 前向きでは簡略化
                confidence_score=np.mean(keypoints[:, 2]) if keypoints.shape[1] > 2 else 0.8,
                visible_keypoints=visible_count,
                orientation=PersonOrientation.FRONT_FACING,
            )

            return features

        except Exception as e:
            logger.error(f"前向き特徴量抽出エラー: {e}")
            return None

    def _extract_features_back_view(self, keypoints: np.ndarray) -> Optional[PostureFeatures]:
        """後ろ向き映像用の特徴量抽出"""
        try:
            if keypoints.shape[0] < 17:
                return None

            # 後ろから見える主要なキーポイント
            left_shoulder = keypoints[5][:2]
            right_shoulder = keypoints[6][:2]
            left_elbow = keypoints[7][:2]
            right_elbow = keypoints[8][:2]
            left_wrist = keypoints[9][:2]
            right_wrist = keypoints[10][:2]

            # 頭部の推定位置
            if not (np.allclose(left_shoulder, [0, 0]) or np.allclose(right_shoulder, [0, 0])):
                shoulder_center = (left_shoulder + right_shoulder) / 2
                shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
                head_estimated = shoulder_center - np.array([0, shoulder_width * 0.3])
            else:
                head_estimated = np.array([0, 0])

            visible_count = sum(1 for kp in keypoints if kp[0] > 0 and kp[1] > 0)

            def safe_distance(p1, p2):
                if (np.any(np.isnan(p1)) or np.any(np.isnan(p2)) or 
                    np.allclose(p1, [0, 0]) or np.allclose(p2, [0, 0])):
                    return float("inf")
                return np.linalg.norm(p1 - p2)

            def calculate_angle(p1, p2, p3):
                if any(np.allclose(p, [0, 0]) for p in [p1, p2, p3]):
                    return 0.0
                v1 = p1 - p2
                v2 = p3 - p2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                return np.degrees(np.arccos(cos_angle))

            features = PostureFeatures(
                head_angle=self._calculate_shoulder_levelness(left_shoulder, right_shoulder),
                hand_face_distance_left=safe_distance(left_wrist, head_estimated),
                hand_face_distance_right=safe_distance(right_wrist, head_estimated),
                shoulder_hand_angle_left=calculate_angle(left_shoulder, left_elbow, left_wrist),
                shoulder_hand_angle_right=calculate_angle(right_shoulder, right_elbow, right_wrist),
                head_tilt=self._calculate_shoulder_tilt(left_shoulder, right_shoulder),
                neck_forward=0.0,  # 後ろ向きでは簡略化
                confidence_score=np.mean(keypoints[:, 2]) if keypoints.shape[1] > 2 else 0.8,
                visible_keypoints=visible_count,
                orientation=PersonOrientation.BACK_FACING,
            )

            return features

        except Exception as e:
            logger.error(f"後ろ向き特徴量抽出エラー: {e}")
            return None

    def classify_phone_usage(self, features: PostureFeatures) -> Tuple[PhoneUsageState, float]:
        """向きに応じてスマホ使用状態を分類"""
        if features.orientation == PersonOrientation.FRONT_FACING:
            return self._classify_phone_usage_front_view(features)
        elif features.orientation == PersonOrientation.BACK_FACING:
            return self._classify_phone_usage_back_view(features)
        else:
            return self._classify_phone_usage_front_view(features)

    def _classify_phone_usage_front_view(self, features: PostureFeatures) -> Tuple[PhoneUsageState, float]:
        """前向き映像用のスマホ使用状態分類"""
        confidence = features.confidence_score

        # 両手が顔の近くにある場合
        if (features.hand_face_distance_left < self.config["phone_distance_threshold"] and
            features.hand_face_distance_right < self.config["phone_distance_threshold"]):
            return PhoneUsageState.BOTH_HANDS_UP, confidence * 0.9

        # 片手が顔の近くにある場合
        elif (features.hand_face_distance_left < self.config["phone_distance_threshold"] or
            features.hand_face_distance_right < self.config["phone_distance_threshold"]):
            if features.head_tilt > self.config["head_angle_threshold"]:
                return PhoneUsageState.LOOKING_DOWN, confidence * 0.8
            else:
                return PhoneUsageState.HOLDING_NEAR_FACE, confidence * 0.85

        # 判定困難な場合
        elif features.visible_keypoints < 10:
            return PhoneUsageState.UNCERTAIN, confidence * 0.5
        else:
            return PhoneUsageState.NOT_USING, confidence * 0.9

    def _classify_phone_usage_back_view(self, features: PostureFeatures) -> Tuple[PhoneUsageState, float]:
        """後ろ向き映像用のスマホ使用状態分類"""
        confidence = features.confidence_score

        # 両手が頭部付近にある場合
        if (features.hand_face_distance_left < self.back_view_config["hand_head_distance_threshold"] and
            features.hand_face_distance_right < self.back_view_config["hand_head_distance_threshold"]):
            return PhoneUsageState.BOTH_HANDS_UP, confidence * 0.8

        # 片手が頭部付近にある場合
        elif (features.hand_face_distance_left < self.back_view_config["hand_head_distance_threshold"] or
            features.hand_face_distance_right < self.back_view_config["hand_head_distance_threshold"]):
            if (features.shoulder_hand_angle_left < self.back_view_config["arm_bend_threshold"] or
                features.shoulder_hand_angle_right < self.back_view_config["arm_bend_threshold"]):
                return PhoneUsageState.HOLDING_NEAR_FACE, confidence * 0.7
            else:
                return PhoneUsageState.UNCERTAIN, confidence * 0.5

        # 判定困難な場合
        elif features.visible_keypoints < 8:
            return PhoneUsageState.UNCERTAIN, confidence * 0.4
        else:
            return PhoneUsageState.NOT_USING, confidence * 0.8

    def _calculate_head_tilt(self, left_ear: np.ndarray, right_ear: np.ndarray) -> float:
        """頭部の傾きを計算"""
        if np.allclose(left_ear, [0, 0]) or np.allclose(right_ear, [0, 0]):
            return 0.0

        height_diff = abs(left_ear[1] - right_ear[1])
        width_diff = abs(left_ear[0] - right_ear[0])

        if width_diff == 0:
            return 0.0

        return np.degrees(np.arctan(height_diff / width_diff))

    def _calculate_shoulder_levelness(self, left_shoulder: np.ndarray, right_shoulder: np.ndarray) -> float:
        """肩の水平度を計算"""
        if np.allclose(left_shoulder, [0, 0]) or np.allclose(right_shoulder, [0, 0]):
            return 0.0

        shoulder_vector = right_shoulder - left_shoulder
        horizontal_vector = np.array([1, 0])

        cos_angle = np.dot(shoulder_vector, horizontal_vector) / np.linalg.norm(shoulder_vector)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(abs(cos_angle)))

    def _calculate_shoulder_tilt(self, left_shoulder: np.ndarray, right_shoulder: np.ndarray) -> float:
        """肩の傾きを計算"""
        if np.allclose(left_shoulder, [0, 0]) or np.allclose(right_shoulder, [0, 0]):
            return 0.0

        height_diff = abs(left_shoulder[1] - right_shoulder[1])
        width_diff = abs(left_shoulder[0] - right_shoulder[0])

        if width_diff == 0:
            return 0.0

        return np.degrees(np.arctan(height_diff / width_diff))

    def smooth_detection(self, track_id, phone_state):
        """検出結果を平滑化（track_idベース）"""
        if track_id not in self.person_states:
            self.person_states[track_id] = {
                "phone_history": deque(maxlen=5),
                "last_seen": 0,
            }

        state = self.person_states[track_id]
        state["phone_history"].append(phone_state)

        # 過半数の判定で決定
        if len(state["phone_history"]) == 0:
            return PhoneUsageState.NOT_USING

        # 最も多い状態を採用
        state_counts = defaultdict(int)
        for s in state["phone_history"]:
            state_counts[s] += 1

        return max(state_counts, key=state_counts.get)

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

    def get_grid_position(self, bbox: Tuple[int, int, int, int], frame_width: int, frame_height: int) -> Tuple[int, int]:
        """人物の位置をグリッド座標で取得"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        grid_x = 0 if center_x < frame_width * self.split_ratios_cols[0] else 1
        grid_y = 0 if center_y < frame_height * self.split_ratios[0] else 1

        return (grid_y, grid_x)

    def process_frame(self, frame, frame_idx, csv_writer, enhanced_csv_logger=None):
        """フレームを処理して検出結果を返す（改良版）"""
        self.frame_count = frame_idx
        height, width = frame.shape[:2]

        # YOLO検出実行
        try:
            results = self.model(frame, conf=self.config["conf_threshold"], verbose=False)
        except Exception as e:
            logger.error(f"YOLO検出エラー: {e}")
            return frame

        # グリッド境界計算
        x_grid = [0, int(width * self.split_ratios_cols[0]), width]
        y_grid = [0, int(height * self.split_ratios[0]), height]

            # 検出結果を整理
        detections = []
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue

            try:
                kps_list = result.keypoints.xy.cpu().numpy()
                kps_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for i, (kps, box, conf) in enumerate(zip(kps_list, boxes, confs)):
                    if conf < self.config["conf_threshold"]:
                        continue

                    if kps.shape[0] < 17:
                        continue

                    # キーポイント信頼度を統合
                    if kps_conf is not None and i < len(kps_conf):
                        # YOLOのキーポイント信頼度を使用
                        kps_with_conf = np.column_stack([kps, kps_conf[i]])
                    else:
                        # 信頼度が取得できない場合はYOLO検出信頼度をベースにする
                        default_conf = np.ones(kps.shape[0]) * float(conf) * 0.8
                        kps_with_conf = np.column_stack([kps, default_conf])

                    # 中心座標計算
                    valid_points = kps[kps[:, 0] > 0]
                    if len(valid_points) == 0:
                        continue
                    center = np.mean(valid_points[:, :2], axis=0)
                    cx, cy = int(center[0]), int(center[1])

                    detections.append({
                        'center': (cx, cy),
                        'bbox': box,
                        'keypoints': kps_with_conf,
                        'yolo_confidence': float(conf),
                        'keypoint_confidences': kps_conf[i] if kps_conf is not None and i < len(kps_conf) else None
                    })

            except Exception as e:
                logger.error(f"検出結果処理エラー: {e}")
                continue

        # 順序付きIDトラッカーで追跡更新
        tracked_detections = self.id_tracker.update_tracks(detections)


        # 検出結果の処理
        detection_results = []
        for detection in tracked_detections:
            track_id = detection['track_id']
            kps = detection['keypoints']
            box = detection['bbox']
            cx, cy = detection['center']
            yolo_confidence = detection['yolo_confidence']
            keypoint_confidences = detection.get('keypoint_confidences', None)

            # 高度な特徴量抽出
            features = self.extract_advanced_features(kps)
            if features is None:
                continue

            # スマホ使用状態分類
            phone_state_raw, state_confidence = self.classify_phone_usage(features)
            phone_state = self.smooth_detection(track_id, phone_state_raw)

            # キーポイント信頼度の統計計算
            if keypoint_confidences is not None:
                avg_keypoint_conf = float(np.mean(keypoint_confidences[keypoint_confidences > 0]))
                min_keypoint_conf = float(np.min(keypoint_confidences[keypoint_confidences > 0]))
                max_keypoint_conf = float(np.max(keypoint_confidences))
                visible_keypoints = int(np.sum(keypoint_confidences > 0))
            else:
                avg_keypoint_conf = yolo_confidence * 0.8
                min_keypoint_conf = yolo_confidence * 0.6
                max_keypoint_conf = yolo_confidence
                visible_keypoints = int(np.sum(kps[:, 0] > 0))

            # 総合信頼度（YOLO信頼度 × 平均キーポイント信頼度）
            overall_confidence = yolo_confidence * avg_keypoint_conf

            # グリッド位置
            bbox_tuple = tuple(map(int, box))
            grid_pos = self.get_grid_position(bbox_tuple, width, height)
            row, col = grid_pos

            # DetectionResultオブジェクト作成
            detection_result = DetectionResult(
                frame_id=frame_idx,
                track_id=track_id,
                timestamp=time.time(),
                phone_state=phone_state,
                confidence=overall_confidence,
                features=features,
                bbox=bbox_tuple,
                grid_position=grid_pos,
                keypoints_visible=[kpt[0] > 0 and kpt[1] > 0 for kpt in kps],
                orientation=features.orientation
            )

            detection_results.append(detection_result)

            # 基本CSVに結果を記録（キーポイント信頼度情報を追加）
            if csv_writer:
                using_phone = phone_state not in [PhoneUsageState.NOT_USING, PhoneUsageState.UNCERTAIN]
                csv_writer.writerow([
                    frame_idx, track_id, using_phone, row, col,
                    f"{yolo_confidence:.3f}",  # YOLO検出信頼度
                    f"{avg_keypoint_conf:.3f}",  # 平均キーポイント信頼度
                    f"{overall_confidence:.3f}",  # 総合信頼度
                    visible_keypoints  # 可視キーポイント数
                ])

            # 拡張CSVに結果を記録（キーポイント信頼度を詳細に記録）
            if enhanced_csv_logger is not None:
                try:
                    enhanced_csv_logger.log_detection_result_simplified(
                        detection_result, kps, yolo_confidence
                )
                except Exception as csv_error:
                    logger.error(f"簡略化CSV記録エラー: {csv_error}")



            # 描画処理（状態を表示せず、信頼度のみ表示）
            self._draw_detection_on_frame_with_confidence(frame, detection_result, kps,
                                                    yolo_confidence, avg_keypoint_conf, overall_confidence)


        # デバッグ情報の表示
        active_tracks = self.id_tracker.get_active_tracks()
        active_ids = sorted(active_tracks.keys())
        id_info = f"Active IDs (L→R): {active_ids}" if active_ids else "Active IDs: None"

        cv2.putText(frame, id_info, (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Persons: {len(active_ids)}", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self._draw_fps(frame)
        return frame

    def _calculate_fps(self) -> float:
        """現在のFPSを計算"""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        if frame_time > 0:
            fps = 1.0 / frame_time
            self.fps_counter.append(fps)

        # 過去数フレームの平均FPSを返す
        if len(self.fps_counter) > 0:
            return sum(self.fps_counter) / len(self.fps_counter)
        return 0.0

    def _draw_fps(self, frame: np.ndarray) -> None:
        """FPS情報を画面に描画"""
        current_fps = self._calculate_fps()

        # FPS表示位置（右上角）
        height, width = frame.shape[:2]
        fps_text = f"FPS: {current_fps:.1f}"

        # テキストサイズを測定
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(fps_text, font, font_scale, thickness)

        # 背景矩形を描画（見やすくするため）
        bg_x1 = width - text_width - 20
        bg_y1 = 10
        bg_x2 = width - 10
        bg_y2 = text_height + baseline + 20

        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)  # 黒背景
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 1)  # 白枠

        # FPSテキストを描画
        cv2.putText(frame, fps_text, (bg_x1 + 10, bg_y1 + text_height + 5),
                font, font_scale, (0, 255, 0), thickness)

    def _draw_detection_on_frame_with_confidence(self, frame, detection_result: DetectionResult, keypoints, yolo_conf, keypoint_conf, overall_conf):
        """フレームに検出結果を描画"""
        x1, y1, x2, y2 = detection_result.bbox

        color = (0, 255, 0)  # 全て緑色に統一

        # バウンディングボックス描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # ラベル作成（IDと信頼度のみ）
        label = f"ID:{detection_result.track_id}"

        # ラベル描画
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # キーポイント描画（全て緑色に統一）
        for i, pt in enumerate(keypoints.astype(int)):
            if len(pt) >= 3 and pt[0] > 0 and pt[1] > 0:
                kp_color = (0, 255, 0)  # 緑色固定
                radius = 3  # 固定サイズ
                cv2.circle(frame, tuple(pt[:2]), radius, kp_color, -1)

        # スケルトン描画
        self._draw_skeleton(frame, keypoints, color)

class IntegratedVideoProcessor:
    """統合動画処理システム（改良版）"""

    def __init__(self, k1=-0.1, k2=0.0, p1=0.0, p2=0.0, k3=0.0, alpha=0.6, focal_scale=0.9, model_path="yolo11x-pose.pt"):
        # 改良版歪み補正器
        self.corrector = VideoDistortionCorrector(k1, k2, p1, p2, k3, alpha, focal_scale)
        # 高度な姿勢検出システム
        self.detector = AdvancedPostureDetectionSystem(model_path)
        # 拡張CSVロガー
        self.csv_logger = None
        self.model_path = model_path
        # スマートファイル管理
        self.file_manager = SmartFileManager()

    def estimate_processing_time(self, video_path):
        """動画の処理時間を推定"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # 基本推定式（実際の環境に応じて調整）
            base_time_per_frame = 0.1  # 1フレームあたりの基本処理時間
            resolution_factor = (width * height) / (1920 * 1080)

            estimated_time = total_frames * base_time_per_frame * resolution_factor

            logger.info(f"推定処理時間: {estimated_time:.1f}秒 ({total_frames}フレーム, {width}x{height})")
            return estimated_time

        except Exception as e:
            logger.warning(f"処理時間推定エラー: {e}")
            return 0

    def process_video_with_smart_management(self, input_path, show_preview=True, apply_correction=True, force_process=False):
        """スマートファイル管理付きの動画処理（CSV保護機能強化）"""

        # 処理時間を推定
        estimated_time = self.estimate_processing_time(input_path)

        # 処理すべきかを判定
        should_process, reason, file_paths = self.file_manager.should_process_video(
            input_path, estimated_time, force_process
        )

        if not should_process:
            logger.info(f"⏭️  処理をスキップ: {reason}")

            # 既存ファイル情報を表示（CSV保護情報を含む）
            existing_files = self.file_manager.check_existing_files(file_paths)
            logger.info("=== 🔒 既存ファイル保護情報 ===")
            for key, file_info in existing_files.items():
                size_mb = file_info["size"] / (1024 * 1024)
                mod_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_info["mtime"]))

                # CSVファイルの保護状況を明示
                if "csv" in key.lower():
                    protection_status = "🔒 保護中"
                else:
                    protection_status = "🔒 保護中"

                logger.info(f"{key}: {os.path.basename(file_info['path'])} ({size_mb:.1f}MB, 更新: {mod_time_str}) {protection_status}")

            # ユーザーに確認
            print(f"\n🔒 既存の処理結果が品質劣化防止により保護されています。")
            print(f"スキップ理由: {reason}")
            print("📊 保護対象:")
            print(f"  ・動画ファイル: {os.path.basename(file_paths.get('output_video', 'N/A'))}")
            print(f"  ・基本CSV: {os.path.basename(file_paths.get('basic_csv', 'N/A'))}")
            print(f"  ・拡張CSV: {os.path.basename(file_paths.get('enhanced_csv', 'N/A'))}")
            print("\nそれでも再処理しますか？（既存の全ファイルが削除されます）")
            user_choice = input("再処理する場合は 'y' を入力: ").lower().strip()

            if user_choice != 'y':
                logger.info("⏹️  品質劣化防止により処理をスキップしました")
                logger.info("💡 より長い処理時間が見込まれる場合に自動実行されます")
                return False

            logger.info("🔄 ユーザー選択により品質保護を解除して再処理を実行")

        # 既存ファイルをクリーンアップ（CSV含む）
        if should_process or user_choice == 'y':
            self.file_manager.cleanup_existing_files(file_paths)

        # 実際の動画処理を実行
        logger.info(f"📹 動画処理開始: {reason}")
        logger.info(f"📁 出力ファイル:")
        logger.info(f"  動画: {os.path.basename(file_paths['output_video'])}")
        logger.info(f"  基本CSV: {os.path.basename(file_paths['basic_csv'])}")
        logger.info(f"  拡張CSV: {os.path.basename(file_paths['enhanced_csv'])}")

        # 拡張CSVロガーを設定（動画別ファイル、常に上書き）
        try:
            self.csv_logger = EnhancedCSVLogger(
                file_paths["enhanced_csv"],
                model_name=os.path.basename(self.model_path),
                overwrite_existing=True  # 品質向上時は必ず上書き
            )
            logger.info(f"✅ 拡張CSVロガー初期化成功: {os.path.basename(file_paths['enhanced_csv'])}")
        except Exception as e:
            logger.error(f"❌ 拡張CSVロガー初期化失敗: {e}")
            self.csv_logger = None

        # 従来のprocess_videoメソッドを呼び出し
        success = self.process_video(
            input_path,
            file_paths["output_video"],
            file_paths["basic_csv"],
            show_preview,
            apply_correction
        )

        if success:
            # 処理時間を記録
            processing_time = getattr(self, '_last_processing_time', estimated_time)
            total_frames = getattr(self, '_last_total_frames', 0)

            self.file_manager.log_processing_completion(
                input_path, file_paths, processing_time, total_frames
            )

            logger.info(f"✅ 全ファイル処理完了: {file_paths['video_basename']}")
            logger.info("📊 保護されたファイル:")
            logger.info(f"  ・動画: {os.path.basename(file_paths['output_video'])}")
            logger.info(f"  ・基本CSV: {os.path.basename(file_paths['basic_csv'])}")
            logger.info(f"  ・拡張CSV: {os.path.basename(file_paths['enhanced_csv'])}")

        return success

    def set_csv_logger(self, csv_path="enhanced_detection_log.csv", overwrite_existing=True):
        """CSVロガーを設定（main.py互換性 - 品質保護対応）"""
        try:
            self.csv_logger = EnhancedCSVLogger(
                csv_path,
                model_name=os.path.basename(self.model_path),
                overwrite_existing=overwrite_existing
            )
            logger.info(f"✅ 拡張CSVロガー初期化成功: {os.path.basename(csv_path)}")
        except Exception as e:
            logger.error(f"❌ 拡張CSVロガー初期化失敗: {e}")
            self.csv_logger = None

    def process_video(self, input_path, output_path, result_log="frame_results.csv",
                            show_preview=True, apply_correction=True):
        """動画を処理（main.py互換性メソッド）"""
        start_time = time.time()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"動画ファイルを開けません: {input_path}")
            return False

        # 動画情報取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"動画情報: {width}x{height}, {fps:.1f}fps, {total_frames}フレーム")

        # 歪み補正マップ作成
        if apply_correction:
            logger.info("高精度歪み補正マップを作成中...")
            self.corrector.create_correction_maps(width, height)

        # 出力動画設定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 結果ログ準備
        os.makedirs(os.path.dirname(result_log), exist_ok=True)
        detection_count = 0

        try:
            with open(result_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame", "tracking_id", "using_phone", "grid_row", "grid_col",
                    "yolo_confidence", "avg_keypoint_confidence", "overall_confidence", "visible_keypoints"
                ])

                frame_idx = 0
                logger.info("改良版動画処理を開始...")

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1

                    # 歪み補正適用
                    if apply_correction:
                        frame = self.corrector.apply_correction(frame)

                    # 姿勢検出処理
                    try:
                        frame = self.detector.process_frame(
                            frame, frame_idx, writer, self.csv_logger
                        )

                        # 検出結果をカウント
                        if self.csv_logger and hasattr(self.csv_logger, 'log_count'):
                            detection_count = self.csv_logger.log_count

                    except Exception as detection_error:
                        logger.error(f"フレーム{frame_idx}処理エラー: {detection_error}")

                    # フレーム情報表示
                    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                    # 拡張CSV記録状況表示
                    if self.csv_logger:
                        cv2.putText(frame, f"Enhanced CSV: {detection_count} records", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # プレビュー表示
                    if show_preview:
                        cv2.imshow("Advanced Posture Detection System", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("処理が中断されました")
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
                            f"処理速度: {fps_current:.1f}fps 残り: {eta:.1f}s "
                            f"アクティブID: {sorted(active_tracks.keys())} "
                            f"拡張CSV: {detection_count}行"
                        )

        except Exception as e:
            logger.error(f"動画処理エラー: {e}")
            return False
        finally:
            # リソース解放
            cap.release()
            out.release()
            cv2.destroyAllWindows()

            # CSVロガーをクローズ
            if self.csv_logger:
                self.csv_logger.close()

        # 処理時間と統計を記録
        processing_time = time.time() - start_time
        self._last_processing_time = processing_time
        self._last_total_frames = frame_idx

        # 完了報告
        logger.info(f"🎉 改良版動画処理完了!")
        logger.info(f"処理時間: {processing_time:.1f}秒")
        logger.info(f"平均処理速度: {frame_idx/processing_time:.1f}fps")
        logger.info(f"最終拡張CSV記録数: {detection_count}行")

        return True

    def get_statistics(self) -> dict:
        """処理統計を取得"""
        active_tracks = self.detector.id_tracker.get_active_tracks()
        return {
            'active_tracks': len(active_tracks),
            'total_csv_records': self.csv_logger.log_count if self.csv_logger else 0,
            'track_ids': sorted(active_tracks.keys())
        }