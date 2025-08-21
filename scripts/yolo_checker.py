import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from collections import defaultdict, deque
import logging
import csv
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum

# ===== 設定セクション（ここだけ変更すればOK） =====
DISTORTION_CONFIG = {
    "k1": -0.30,  # 120°の広角レンズなので強めの逆バレル補正
    "k2": 0.03,  # 広角レンズの二次歪みを補正
    "p1": 0.0,  # 接線歪み係数1
    "p2": 0.0,  # 接線歪み係数2
    "k3": 0.01,  # 第3歪み係数
    "alpha": 0.4,  # 広角なので切り抜き重視
    "focal_scale": 0.65,  # 広角効果を少し抑える
    "apply_correction": True,
}

VIDEO_CONFIG = {
    "input_path": "shoot5.mp4",
    "output_path": "output_processed.mp4",
    "result_log": "detection_results.csv",
    "show_preview": True,
    "enable_tracking": True,  # 追跡機能を有効にする
}
# ================================================

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VideoDistortionCorrector:
    """動画の歪み補正クラス（改良版）"""

    def __init__(
        self, k1=-0.1, k2=0.0, p1=0.0, p2=0.0, k3=0.0, alpha=0.6, focal_scale=0.9
    ):
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

        logger.info(f"歪み補正初期化:")
        logger.info(f"  k1={k1}, k2={k2}, k3={k3}")
        logger.info(f"  p1={p1}, p2={p2}")
        logger.info(f"  alpha={alpha}, focal_scale={focal_scale}")

    def create_correction_maps(self, width, height):
        """歪み補正用のマップを作成（改良版）"""
        logger.info(f"歪み補正マップ作成開始: {width}x{height}")

        # カメラ内部パラメータの設定（より適切な値）
        fx = fy = width * self.focal_scale  # 焦点距離
        cx, cy = width / 2.0, height / 2.0  # 主点（画像中心）

        self.original_camera_matrix = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
        )

        # 歪み係数（5つの係数を使用）
        self.dist_coeffs = np.array(
            [self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32
        )

        logger.info(f"元のカメラマトリックス:")
        logger.info(f"  fx={fx:.2f}, fy={fy:.2f}")
        logger.info(f"  cx={cx:.2f}, cy={cy:.2f}")
        logger.info(f"歪み係数: {self.dist_coeffs}")

        # 最適な新しいカメラマトリックスを計算
        self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.original_camera_matrix,
            self.dist_coeffs,
            (width, height),
            self.alpha,  # 0=切り抜き最小、1=全画素表示
            (width, height),
        )

        logger.info(f"新しいカメラマトリックス:")
        logger.info(
            f"  fx={self.new_camera_matrix[0,0]:.2f}, fy={self.new_camera_matrix[1,1]:.2f}"
        )
        logger.info(
            f"  cx={self.new_camera_matrix[0,2]:.2f}, cy={self.new_camera_matrix[1,2]:.2f}"
        )
        logger.info(f"ROI (Region of Interest): {roi}")

        # 補正マップ作成
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.original_camera_matrix,
            self.dist_coeffs,
            None,  # 回転マトリックス（なし）
            self.new_camera_matrix,
            (width, height),
            cv2.CV_32FC1,
        )

        logger.info("歪み補正マップ作成完了")

        # テスト用：マップの統計情報を出力
        self._log_map_statistics()

    def _log_map_statistics(self):
        """マップの統計情報をログ出力（デバッグ用）"""
        if self.map_x is not None and self.map_y is not None:
            x_mean, x_std = np.mean(self.map_x), np.std(self.map_x)
            y_mean, y_std = np.mean(self.map_y), np.std(self.map_y)
            logger.info(f"マップ統計:")
            logger.info(f"  X: mean={x_mean:.2f}, std={x_std:.2f}")
            logger.info(f"  Y: mean={y_mean:.2f}, std={y_std:.2f}")

    def apply_correction(self, frame):
        """フレームに歪み補正を適用"""
        if self.map_x is None or self.map_y is None:
            logger.warning("補正マップが作成されていません")
            return frame

        # リマッピングによる歪み補正
        corrected_frame = cv2.remap(
            frame,
            self.map_x,
            self.map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        return corrected_frame

    def create_comparison_frame(self, original, corrected):
        """元の画像と補正後の画像を並べて表示"""
        height, width = original.shape[:2]

        # 横に並べて表示
        comparison = np.zeros((height, width * 2, 3), dtype=np.uint8)
        comparison[:, :width] = original
        comparison[:, width:] = corrected

        # 区切り線を描画
        cv2.line(comparison, (width, 0), (width, height), (255, 255, 255), 2)

        # ラベルを追加
        cv2.putText(
            comparison,
            "Original",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            comparison,
            "Corrected",
            (width + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        return comparison


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
    detection_id: int  # person_idから変更
    track_id: Optional[int]  # YOLOの追跡ID
    timestamp: float
    phone_state: PhoneUsageState
    confidence: float
    features: PostureFeatures
    bbox: Tuple[int, int, int, int]
    grid_position: Tuple[int, int]
    keypoints_visible: List[bool]
    orientation: PersonOrientation


class AdvancedPostureDetectionSystem:
    """高度な姿勢検出システム（前向き・後ろ向き対応）"""

    def __init__(self, model_path="../models/yolo11n-pose.pt"):
        self.model = YOLO(model_path)

        # 設定
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

        # 詳細ログ用
        self.detailed_results = []

        # グリッド分割設定
        self.split_ratios = [0.5, 0.5]
        self.split_ratios_cols = [0.5, 0.5]

        # 追跡用の履歴
        self.track_history = defaultdict(lambda: deque(maxlen=10))
        self.frame_count = 0

    def determine_person_orientation(self, keypoints: np.ndarray) -> PersonOrientation:
        """人物の向きを判定"""
        try:
            # 顔のキーポイント（鼻、目、耳）
            nose = keypoints[0][:2]
            left_eye = keypoints[1][:2]
            right_eye = keypoints[2][:2]
            left_ear = keypoints[3][:2]
            right_ear = keypoints[4][:2]

            # 肩のキーポイント
            left_shoulder = keypoints[5][:2]
            right_shoulder = keypoints[6][:2]

            # 顔のキーポイントの可視性をチェック
            face_points_visible = sum(
                1
                for p in [nose, left_eye, right_eye, left_ear, right_ear]
                if not np.allclose(p, [0, 0])
            )

            # 肩のキーポイントの可視性をチェック
            shoulder_points_visible = sum(
                1 for p in [left_shoulder, right_shoulder] if not np.allclose(p, [0, 0])
            )

            # 判定ロジック
            if face_points_visible >= 3:
                # 顔のキーポイントが多く見える場合は前向き
                return PersonOrientation.FRONT_FACING
            elif shoulder_points_visible >= 2 and face_points_visible <= 1:
                # 肩は見えるが顔があまり見えない場合は後ろ向き
                return PersonOrientation.BACK_FACING
            elif face_points_visible >= 1 and shoulder_points_visible >= 1:
                # 一部が見える場合は横向き
                return PersonOrientation.SIDE_FACING
            else:
                return PersonOrientation.UNCERTAIN

        except Exception as e:
            logger.error(f"向き判定エラー: {e}")
            return PersonOrientation.UNCERTAIN

    def extract_advanced_features_front_view(
        self, keypoints: np.ndarray
    ) -> Optional[PostureFeatures]:
        """前向き映像用の高度な特徴量を抽出"""
        try:
            if keypoints.shape[0] < 17:
                return None

            # キーポイントの定義（COCO形式）
            nose = keypoints[0][:2]
            left_ear = keypoints[3][:2]
            right_ear = keypoints[4][:2]
            left_shoulder = keypoints[5][:2]
            right_shoulder = keypoints[6][:2]
            left_elbow = keypoints[7][:2]
            right_elbow = keypoints[8][:2]
            left_wrist = keypoints[9][:2]
            right_wrist = keypoints[10][:2]
            neck = keypoints[1][:2]  # 首の近似位置

            # 有効なキーポイント数をカウント
            visible_count = sum(1 for kp in keypoints if kp[0] > 0 and kp[1] > 0)

            # 基本的な距離計算
            def safe_distance(p1, p2):
                if (
                    np.any(np.isnan(p1))
                    or np.any(np.isnan(p2))
                    or np.allclose(p1, [0, 0])
                    or np.allclose(p2, [0, 0])
                ):
                    return float("inf")
                return np.linalg.norm(p1 - p2)

            # 角度計算関数
            def calculate_angle(p1, p2, p3):
                """3点から角度を計算"""
                if any(np.allclose(p, [0, 0]) for p in [p1, p2, p3]):
                    return 0.0
                v1 = p1 - p2
                v2 = p3 - p2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                return np.degrees(np.arccos(cos_angle))

            # 特徴量計算
            features = PostureFeatures(
                head_angle=calculate_angle(left_ear, nose, right_ear),
                hand_face_distance_left=safe_distance(left_wrist, nose),
                hand_face_distance_right=safe_distance(right_wrist, nose),
                shoulder_hand_angle_left=calculate_angle(
                    left_shoulder, left_elbow, left_wrist
                ),
                shoulder_hand_angle_right=calculate_angle(
                    right_shoulder, right_elbow, right_wrist
                ),
                head_tilt=self._calculate_head_tilt(left_ear, right_ear),
                neck_forward=self._calculate_neck_forward(
                    neck, left_shoulder, right_shoulder
                ),
                confidence_score=(
                    np.mean(keypoints[:, 2]) if keypoints.shape[1] > 2 else 0.8
                ),
                visible_keypoints=visible_count,
                orientation=PersonOrientation.FRONT_FACING,
            )

            return features

        except Exception as e:
            logger.error(f"前向き特徴量抽出エラー: {e}")
            return None

    def extract_advanced_features_back_view(
        self, keypoints: np.ndarray
    ) -> Optional[PostureFeatures]:
        """後ろ向き映像用の高度な特徴量を抽出"""
        try:
            if keypoints.shape[0] < 17:
                return None

            # 後ろから見える主要なキーポイント（COCO形式）
            left_shoulder = keypoints[5][:2]  # 左肩
            right_shoulder = keypoints[6][:2]  # 右肩
            left_elbow = keypoints[7][:2]  # 左肘
            right_elbow = keypoints[8][:2]  # 右肘
            left_wrist = keypoints[9][:2]  # 左手首
            right_wrist = keypoints[10][:2]  # 右手首
            left_hip = keypoints[11][:2]  # 左腰
            right_hip = keypoints[12][:2]  # 右腰

            # 首の推定位置（肩の中点から上）
            if not (
                np.allclose(left_shoulder, [0, 0])
                or np.allclose(right_shoulder, [0, 0])
            ):
                shoulder_center = (left_shoulder + right_shoulder) / 2
                # 肩幅の20%上を首と推定
                shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
                neck_estimated = shoulder_center - np.array([0, shoulder_width * 0.2])
            else:
                neck_estimated = np.array([0, 0])

            # 頭部の推定位置（首からさらに上）
            if not np.allclose(neck_estimated, [0, 0]):
                head_estimated = neck_estimated - np.array([0, shoulder_width * 0.3])
            else:
                head_estimated = np.array([0, 0])

            # 有効なキーポイント数をカウント
            visible_count = sum(1 for kp in keypoints if kp[0] > 0 and kp[1] > 0)

            # 距離計算関数
            def safe_distance(p1, p2):
                if (
                    np.any(np.isnan(p1))
                    or np.any(np.isnan(p2))
                    or np.allclose(p1, [0, 0])
                    or np.allclose(p2, [0, 0])
                ):
                    return float("inf")
                return np.linalg.norm(p1 - p2)

            # 角度計算関数
            def calculate_angle(p1, p2, p3):
                """3点から角度を計算"""
                if any(np.allclose(p, [0, 0]) for p in [p1, p2, p3]):
                    return 0.0
                v1 = p1 - p2
                v2 = p3 - p2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                return np.degrees(np.arccos(cos_angle))

            # 後ろ向き用の特徴量計算
            features = PostureFeatures(
                # 頭部角度の代わりに肩の水平度を使用
                head_angle=self._calculate_shoulder_levelness(
                    left_shoulder, right_shoulder
                ),
                # 手首から推定頭部への距離
                hand_face_distance_left=safe_distance(left_wrist, head_estimated),
                hand_face_distance_right=safe_distance(right_wrist, head_estimated),
                # 肩-肘-手首の角度（腕の曲がり具合）
                shoulder_hand_angle_left=calculate_angle(
                    left_shoulder, left_elbow, left_wrist
                ),
                shoulder_hand_angle_right=calculate_angle(
                    right_shoulder, right_elbow, right_wrist
                ),
                # 肩の傾き（頭の傾きの代替）
                head_tilt=self._calculate_shoulder_tilt(left_shoulder, right_shoulder),
                # 首の前傾（後ろ向きでは肩の前傾で代替）
                neck_forward=self._calculate_shoulder_forward_lean(
                    left_shoulder, right_shoulder, left_hip, right_hip
                ),
                confidence_score=(
                    np.mean(keypoints[:, 2]) if keypoints.shape[1] > 2 else 0.8
                ),
                visible_keypoints=visible_count,
                orientation=PersonOrientation.BACK_FACING,
            )

            return features

        except Exception as e:
            logger.error(f"後ろ向き特徴量抽出エラー: {e}")
            return None

    def extract_advanced_features(
        self, keypoints: np.ndarray
    ) -> Optional[PostureFeatures]:
        """向きを自動判定して適切な特徴量を抽出"""
        orientation = self.determine_person_orientation(keypoints)

        if orientation == PersonOrientation.FRONT_FACING:
            return self.extract_advanced_features_front_view(keypoints)
        elif orientation == PersonOrientation.BACK_FACING:
            return self.extract_advanced_features_back_view(keypoints)
        else:
            # 横向きや不明な場合は前向きの処理を使用
            features = self.extract_advanced_features_front_view(keypoints)
            if features:
                features.orientation = orientation
            return features

    def _calculate_head_tilt(
        self, left_ear: np.ndarray, right_ear: np.ndarray
    ) -> float:
        """頭部の傾きを計算"""
        if np.allclose(left_ear, [0, 0]) or np.allclose(right_ear, [0, 0]):
            return 0.0

        # 耳の高さの差から傾きを計算
        height_diff = abs(left_ear[1] - right_ear[1])
        width_diff = abs(left_ear[0] - right_ear[0])

        if width_diff == 0:
            return 0.0

        angle = np.degrees(np.arctan(height_diff / width_diff))
        return angle

    def _calculate_neck_forward(
        self, neck: np.ndarray, left_shoulder: np.ndarray, right_shoulder: np.ndarray
    ) -> float:
        """首の前傾度を計算"""
        if any(np.allclose(p, [0, 0]) for p in [neck, left_shoulder, right_shoulder]):
            return 0.0

        # 肩の中点を計算
        shoulder_center = (left_shoulder + right_shoulder) / 2

        # 首の前傾度（正規化）
        forward_distance = abs(neck[0] - shoulder_center[0])
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

        if shoulder_width == 0:
            return 0.0

        return forward_distance / shoulder_width

    def _calculate_shoulder_levelness(
        self, left_shoulder: np.ndarray, right_shoulder: np.ndarray
    ) -> float:
        """肩の水平度を計算（頭部角度の代替）"""
        if np.allclose(left_shoulder, [0, 0]) or np.allclose(right_shoulder, [0, 0]):
            return 0.0

        # 肩のライン（左右の肩を結ぶ線）の水平からの角度
        shoulder_vector = right_shoulder - left_shoulder
        horizontal_vector = np.array([1, 0])

        cos_angle = np.dot(shoulder_vector, horizontal_vector) / np.linalg.norm(
            shoulder_vector
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(abs(cos_angle)))

        return angle

    def _calculate_shoulder_tilt(
        self, left_shoulder: np.ndarray, right_shoulder: np.ndarray
    ) -> float:
        """肩の傾きを計算（頭の傾きの代替）"""
        if np.allclose(left_shoulder, [0, 0]) or np.allclose(right_shoulder, [0, 0]):
            return 0.0

        # 肩の高さの差から傾きを計算
        height_diff = abs(left_shoulder[1] - right_shoulder[1])
        width_diff = abs(left_shoulder[0] - right_shoulder[0])

        if width_diff == 0:
            return 0.0

        angle = np.degrees(np.arctan(height_diff / width_diff))
        return angle

    def _calculate_shoulder_forward_lean(
        self,
        left_shoulder: np.ndarray,
        right_shoulder: np.ndarray,
        left_hip: np.ndarray,
        right_hip: np.ndarray,
    ) -> float:
        """肩の前傾度を計算（首の前傾の代替）"""
        if any(
            np.allclose(p, [0, 0])
            for p in [left_shoulder, right_shoulder, left_hip, right_hip]
        ):
            return 0.0

        # 肩と腰の中点を計算
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        # 肩の前傾度（腰に対する相対位置）
        forward_distance = abs(shoulder_center[0] - hip_center[0])
        torso_height = abs(shoulder_center[1] - hip_center[1])

        if torso_height == 0:
            return 0.0

        return forward_distance / torso_height

    def classify_phone_usage_front_view(
        self, features: PostureFeatures
    ) -> Tuple[PhoneUsageState, float]:
        """前向き映像用のスマホ使用状態分類"""

        confidence = features.confidence_score

        # 両手が顔の近くにある場合
        if (
            features.hand_face_distance_left < self.config["phone_distance_threshold"]
            and features.hand_face_distance_right
            < self.config["phone_distance_threshold"]
        ):
            return PhoneUsageState.BOTH_HANDS_UP, confidence * 0.9

        # 片手が顔の近くにある場合
        elif (
            features.hand_face_distance_left < self.config["phone_distance_threshold"]
            or features.hand_face_distance_right
            < self.config["phone_distance_threshold"]
        ):

            # 頭部が下向きの場合
            if features.head_tilt > self.config["head_angle_threshold"]:
                return PhoneUsageState.LOOKING_DOWN, confidence * 0.8
            else:
                return PhoneUsageState.HOLDING_NEAR_FACE, confidence * 0.85

        # 首の前傾が大きい場合（下を向いている）
        elif features.neck_forward > self.config["neck_forward_threshold"]:
            return PhoneUsageState.LOOKING_DOWN, confidence * 0.7

        # 判定困難な場合
        elif features.visible_keypoints < 10:
            return PhoneUsageState.UNCERTAIN, confidence * 0.5

        else:
            return PhoneUsageState.NOT_USING, confidence * 0.9

    def classify_phone_usage_back_view(
        self, features: PostureFeatures
    ) -> Tuple[PhoneUsageState, float]:
        """後ろ向き映像用のスマホ使用状態分類"""

        confidence = features.confidence_score

        # 両手が頭部付近にある場合（通話の可能性）
        if (
            features.hand_face_distance_left
            < self.back_view_config["hand_head_distance_threshold"]
            and features.hand_face_distance_right
            < self.back_view_config["hand_head_distance_threshold"]
        ):
            return PhoneUsageState.BOTH_HANDS_UP, confidence * 0.8

        # 片手が頭部付近にある場合
        elif (
            features.hand_face_distance_left
            < self.back_view_config["hand_head_distance_threshold"]
            or features.hand_face_distance_right
            < self.back_view_config["hand_head_distance_threshold"]
        ):

            # 腕が大きく曲がっている場合（スマホを持っている可能性）
            if (
                features.shoulder_hand_angle_left
                < self.back_view_config["arm_bend_threshold"]
                or features.shoulder_hand_angle_right
                < self.back_view_config["arm_bend_threshold"]
            ):
                return PhoneUsageState.HOLDING_NEAR_FACE, confidence * 0.7
            else:
                return PhoneUsageState.UNCERTAIN, confidence * 0.5

        # 前傾姿勢（下を向いてスマホを見ている可能性）
        elif features.neck_forward > self.back_view_config["forward_lean_threshold"]:
            return PhoneUsageState.LOOKING_DOWN, confidence * 0.6

        # 肩の傾きが大きい（首を傾げている可能性）
        elif features.head_tilt > self.back_view_config["shoulder_tilt_threshold"]:
            return PhoneUsageState.UNCERTAIN, confidence * 0.5

        # 判定困難な場合
        elif features.visible_keypoints < 8:
            return PhoneUsageState.UNCERTAIN, confidence * 0.4

        else:
            return PhoneUsageState.NOT_USING, confidence * 0.8

    def classify_phone_usage(
        self, features: PostureFeatures
    ) -> Tuple[PhoneUsageState, float]:
        """向きに応じてスマホ使用状態を分類"""
        if features.orientation == PersonOrientation.FRONT_FACING:
            return self.classify_phone_usage_front_view(features)
        elif features.orientation == PersonOrientation.BACK_FACING:
            return self.classify_phone_usage_back_view(features)
        else:
            # 横向きや不明な場合は前向きの処理を使用
            return self.classify_phone_usage_front_view(features)

    def get_grid_position(self, bbox: Tuple[int, int, int, int], frame_width: int, frame_height: int) -> Tuple[int, int]:
        """人物の位置をグリッド座標で取得"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # グリッド位置を計算
        grid_x = 0 if center_x < frame_width * self.split_ratios_cols[0] else 1
        grid_y = 0 if center_y < frame_height * self.split_ratios[0] else 1

        return (grid_x, grid_y)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        """フレームを処理して姿勢検出とスマホ使用状態判定を実行"""
        self.frame_count += 1
        height, width = frame.shape[:2]

        # YOLO推論
        results = self.model.track(frame, conf=self.config["conf_threshold"], persist=True)

        detection_results = []
        annotated_frame = frame.copy()

        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue

            # 検出されたボックスとキーポイント
            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.xy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            # 追跡IDを取得（利用可能な場合）
            track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else None

            for i, (box, kpts, conf) in enumerate(zip(boxes, keypoints, confidences)):
                if conf < self.config["conf_threshold"]:
                    continue

                # キーポイントから特徴量を抽出
                features = self.extract_advanced_features(kpts)
                if features is None:
                    continue

                # スマホ使用状態を分類
                phone_state, state_confidence = self.classify_phone_usage(features)

                # 追跡ID
                track_id = track_ids[i] if track_ids is not None else None

                # グリッド位置
                bbox_tuple = tuple(map(int, box))
                grid_pos = self.get_grid_position(bbox_tuple, width, height)

                # 検出結果を作成
                detection_result = DetectionResult(
                    frame_id=self.frame_count,
                    detection_id=i,
                    track_id=track_id,
                    timestamp=time.time(),
                    phone_state=phone_state,
                    confidence=state_confidence,
                    features=features,
                    bbox=bbox_tuple,
                    grid_position=grid_pos,
                    keypoints_visible=[kpt[0] > 0 and kpt[1] > 0 for kpt in kpts],
                    orientation=features.orientation
                )

                detection_results.append(detection_result)

                # 追跡履歴を更新
                if track_id is not None:
                    self.track_history[track_id].append(phone_state)

                # 色の設定
                color_map = {
                    PhoneUsageState.NOT_USING: (0, 255, 0),  # 緑
                    PhoneUsageState.HOLDING_NEAR_FACE: (0, 165, 255),  # オレンジ
                    PhoneUsageState.LOOKING_DOWN: (0, 0, 255),  # 赤
                    PhoneUsageState.BOTH_HANDS_UP: (255, 0, 255),  # マゼンタ
                    PhoneUsageState.UNCERTAIN: (128, 128, 128),  # グレー
                    PhoneUsageState.TRANSITIONING: (255, 255, 0),  # シアン
                }
                color = color_map.get(phone_state, (255, 255, 255))

                # キーポイントとスケルトンを描画
                annotated_frame = self._draw_keypoints(annotated_frame, kpts, color)

                # 検出情報を描画
                annotated_frame = self._draw_detection(annotated_frame, detection_result)

        return annotated_frame, detection_results

    def _draw_detection(self, frame: np.ndarray, detection: DetectionResult) -> np.ndarray:
        """検出結果をフレームに描画"""
        x1, y1, x2, y2 = detection.bbox

        # 状態に応じて色を設定
        color_map = {
            PhoneUsageState.NOT_USING: (0, 255, 0),  # 緑
            PhoneUsageState.HOLDING_NEAR_FACE: (0, 165, 255),  # オレンジ
            PhoneUsageState.LOOKING_DOWN: (0, 0, 255),  # 赤
            PhoneUsageState.BOTH_HANDS_UP: (255, 0, 255),  # マゼンタ
            PhoneUsageState.UNCERTAIN: (128, 128, 128),  # グレー
            PhoneUsageState.TRANSITIONING: (255, 255, 0),  # シアン
        }

        color = color_map.get(detection.phone_state, (255, 255, 255))

        # バウンディングボックス
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ラベル

        if detection.track_id is not None:
            label = f" ID:{detection.track_id}"
        label += f"_"
        label += f"{detection.phone_state.value}"
        label += f" ({detection.confidence:.2f})"

        # 向きの情報
        orientation_short = {
            PersonOrientation.FRONT_FACING: "Front",
            PersonOrientation.BACK_FACING: "Back", 
            PersonOrientation.SIDE_FACING: "Side",
            PersonOrientation.UNCERTAIN: "?"
        }
        label += f" [{orientation_short[detection.orientation]}]"

        # ラベルの背景
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1), color, -1)

        # ラベルのテキスト
        cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # グリッド位置の表示
        grid_label = f"Grid: {detection.grid_position}"
        cv2.putText(frame, grid_label, (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def _draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
        """キーポイントとスケルトンを描画"""
        # COCO形式のキーポイント接続情報
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # 顔周り
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],         # 胴体・腕
            [8, 10], [9, 11], [12, 14], [13, 15], [14, 16], [15, 17]  # 腕・脚
        ]

        # キーポイントの名前（COCO形式）
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        # キーポイントを描画
        for i, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0:  # 有効なキーポイントのみ
                # キーポイントの円
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                cv2.circle(frame, (int(x), int(y)), 6, (255, 255, 255), 2)

                # キーポイント番号（デバッグ用）
                cv2.putText(frame, str(i), (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # スケルトン（骨格）を描画
        for connection in skeleton:
            pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1  # 1-indexed to 0-indexed
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1 = keypoints[pt1_idx]
                pt2 = keypoints[pt2_idx]

                # 両方のキーポイントが有効な場合のみ線を描画
                if (pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0):
                    cv2.line(frame, (int(pt1[0]), int(pt1[1])),
                            (int(pt2[0]), int(pt2[1])), color, 2)

        return frame

    def save_results_to_csv(self, results: List[DetectionResult], filename: str):
        """結果をCSVファイルに保存"""
        if not results:
            logger.warning("保存する結果がありません")
            return

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'frame_id', 'detection_id', 'track_id', 'timestamp',
                'phone_state', 'confidence', 'orientation',
                'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                'grid_x', 'grid_y', 'visible_keypoints',
                'head_angle', 'hand_face_distance_left', 'hand_face_distance_right',
                'shoulder_hand_angle_left', 'shoulder_hand_angle_right',
                'head_tilt', 'neck_forward'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    'frame_id': result.frame_id,
                    'detection_id': result.detection_id,
                    'track_id': result.track_id,
                    'timestamp': result.timestamp,
                    'phone_state': result.phone_state.value,
                    'confidence': result.confidence,
                    'orientation': result.orientation.value,
                    'bbox_x1': result.bbox[0],
                    'bbox_y1': result.bbox[1],
                    'bbox_x2': result.bbox[2],
                    'bbox_y2': result.bbox[3],
                    'grid_x': result.grid_position[0],
                    'grid_y': result.grid_position[1],
                    'visible_keypoints': result.features.visible_keypoints,
                    'head_angle': result.features.head_angle,
                    'hand_face_distance_left': result.features.hand_face_distance_left,
                    'hand_face_distance_right': result.features.hand_face_distance_right,
                    'shoulder_hand_angle_left': result.features.shoulder_hand_angle_left,
                    'shoulder_hand_angle_right': result.features.shoulder_hand_angle_right,
                    'head_tilt': result.features.head_tilt,
                    'neck_forward': result.features.neck_forward,
                }
                writer.writerow(row)

        logger.info(f"結果を {filename} に保存しました ({len(results)} 件)")

    def get_statistics(self, results: List[DetectionResult]) -> Dict:
        """検出結果の統計情報を計算"""
        if not results:
            return {}

        # 状態別の統計
        state_counts = defaultdict(int)
        orientation_counts = defaultdict(int)
        confidence_sum = 0

        for result in results:
            state_counts[result.phone_state.value] += 1
            orientation_counts[result.orientation.value] += 1
            confidence_sum += result.confidence

        # 平均信頼度
        avg_confidence = confidence_sum / len(results) if results else 0

        # 統計情報をまとめる
        stats = {
            'total_detections': len(results),
            'average_confidence': avg_confidence,
            'state_distribution': dict(state_counts),
            'orientation_distribution': dict(orientation_counts),
            'phone_usage_ratio': (
                state_counts['holding_near_face'] + 
                state_counts['looking_down'] + 
                state_counts['both_hands_up']
            ) / len(results) if results else 0
        }

        return stats


def main():
    """メイン処理"""
    logger.info("姿勢検出システムを開始します")

    # 設定の表示
    logger.info(f"入力動画: {VIDEO_CONFIG['input_path']}")
    logger.info(f"出力動画: {VIDEO_CONFIG['output_path']}")
    logger.info(f"歪み補正: {'有効' if DISTORTION_CONFIG['apply_correction'] else '無効'}")

    # 動画ファイルの存在確認
    if not os.path.exists(VIDEO_CONFIG['input_path']):
        logger.error(f"入力動画ファイルが見つかりません: {VIDEO_CONFIG['input_path']}")
        return

    # システム初期化
    try:
        posture_system = AdvancedPostureDetectionSystem()

        # 歪み補正器の初期化
        distortion_corrector = None
        if DISTORTION_CONFIG['apply_correction']:
            distortion_corrector = VideoDistortionCorrector(**{
                k: v for k, v in DISTORTION_CONFIG.items() 
                if k != 'apply_correction'
            })

    except Exception as e:
        logger.error(f"システム初期化エラー: {e}")
        return

    # 動画処理
    cap = cv2.VideoCapture(VIDEO_CONFIG['input_path'])
    if not cap.isOpened():
        logger.error("動画ファイルを開けませんでした")
        return

    # 動画情報取得
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"動画情報: {width}x{height}, {fps}fps, {total_frames}フレーム")

    # 歪み補正マップの作成
    if distortion_corrector:
        distortion_corrector.create_correction_maps(width, height)

    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_CONFIG['output_path'], fourcc, fps, (width, height))

    # 処理開始
    all_results = []
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 歪み補正
            if distortion_corrector:
                frame = distortion_corrector.apply_correction(frame)

            # 姿勢検出とスマホ使用状態判定
            processed_frame, frame_results = posture_system.process_frame(frame)
            all_results.extend(frame_results)

            # 動画に書き込み
            out.write(processed_frame)

            # プレビュー表示
            if VIDEO_CONFIG['show_preview']:
                # 表示サイズを調整
                display_frame = cv2.resize(processed_frame, (960, 540))
                cv2.imshow('姿勢検出結果', display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("ユーザーによって処理が中断されました")
                    break

            # 進捗表示
            if frame_count % 30 == 0:  # 30フレームごと
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / frame_count) * (total_frames - frame_count)
                logger.info(f"進捗: {progress:.1f}% ({frame_count}/{total_frames}), "
                        f"検出数: {len(frame_results)}, ETA: {eta:.1f}s")

    except KeyboardInterrupt:
        logger.info("処理が中断されました")
    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}")
    finally:
        # リソースの解放
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # 結果の保存と統計
    if all_results:
        # CSV保存
        posture_system.save_results_to_csv(all_results, VIDEO_CONFIG['result_log'])

        # 統計情報
        stats = posture_system.get_statistics(all_results)
        logger.info("=== 処理結果統計 ===")
        logger.info(f"総検出数: {stats['total_detections']}")
        logger.info(f"平均信頼度: {stats['average_confidence']:.3f}")
        logger.info(f"スマホ使用率: {stats['phone_usage_ratio']:.3f}")
        logger.info(f"状態分布: {stats['state_distribution']}")
        logger.info(f"向き分布: {stats['orientation_distribution']}")

    else:
        logger.warning("検出結果がありませんでした")

    # 処理時間
    total_time = time.time() - start_time
    logger.info(f"処理完了: {total_time:.2f}秒 ({frame_count/total_time:.1f} fps)")
    logger.info(f"出力動画: {VIDEO_CONFIG['output_path']}")


if __name__ == "__main__":
    main()