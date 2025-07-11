import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from ultralytics import YOLO
from collections import defaultdict, deque
import logging
import csv

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


class DrowsinessDetectionSystem:
    """居眠り検出システムクラス"""

    def __init__(self, model_path="models/yolov8m-pose.pt"):
        self.model = YOLO(model_path)
        self.person_states = defaultdict(
            lambda: {
                "phone_history": deque(maxlen=10),
                "last_seen": 0,
            }
        )

        self.config = {
            "conf_threshold": 0.4,
            "phone_distance_threshold": 100,
            "smoothing_frames": 5,
            "detection_interval": 3,
        }

        # グリッド分割設定を初期化
        self.split_ratios = [0.5, 0.5]  # 上下50%ずつ
        self.split_ratios_cols = [0.5, 0.5]  # 左右50%ずつ

    def get_person_id(self, keypoints, existing_persons):
        """人物IDを取得"""
        if not existing_persons:
            return 0

        center = np.mean(keypoints[:, :2], axis=0)
        min_dist = float("inf")
        best_id = len(existing_persons)

        for person_id, last_center in existing_persons.items():
            dist = np.linalg.norm(center - last_center)
            if dist < min_dist and dist < 100:
                min_dist = dist
                best_id = person_id

        return best_id

    def smooth_detection(self, person_id, using_phone):
        """検出結果を平滑化"""
        state = self.person_states[person_id]
        state["phone_history"].append(using_phone)
        smoothed_phone = sum(state["phone_history"]) > len(state["phone_history"]) // 2
        return smoothed_phone

    def detect_phone_usage(self, keypoints):
        """携帯電話使用を検出"""
        try:
            # キーポイントの信頼度チェック
            if keypoints.shape[0] < 17:
                return False

            nose = keypoints[0][:2]
            left_wrist = keypoints[9][:2]
            right_wrist = keypoints[10][:2]

            def valid(pt):
                return not np.any(np.isnan(pt)) and not np.allclose(pt, [0, 0])

            if not all(map(valid, [nose, left_wrist, right_wrist])):
                return False

            threshold = self.config["phone_distance_threshold"]
            dist_left = np.linalg.norm(nose - left_wrist)
            dist_right = np.linalg.norm(nose - right_wrist)

            return min(dist_left, dist_right) < threshold
        except Exception as e:
            logger.error(f"携帯使用判定エラー: {e}")
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

    def process_frame(self, frame, frame_idx, csv_writer):
        """フレームを処理して検出結果を返す"""
        height, width = frame.shape[:2]
        keypoints_all = []
        bboxes = []

        # YOLOで全体フレームを処理
        results = self.model(frame, conf=self.config["conf_threshold"], verbose=False)

        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue

            kps_list = result.keypoints.xy.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for kps, box, conf in zip(kps_list, boxes, confs):
                if conf < self.config["conf_threshold"]:
                    continue
                keypoints_all.append(kps)
                bboxes.append(box)

        # グリッド境界計算
        x_grid, y_grid = self.calculate_grid_boundaries(
            width, height, self.split_ratios_cols, self.split_ratios
        )

        # 検出結果の処理と描画
        if keypoints_all:
            existing_persons = {}
            for kps, box in zip(keypoints_all, bboxes):
                if kps.shape[0] < 17:
                    continue

                person_id = self.get_person_id(kps, existing_persons)
                center = np.mean(kps[:, :2], axis=0)
                existing_persons[person_id] = center

                using_phone = self.detect_phone_usage(kps)
                using_phone = self.smooth_detection(person_id, using_phone)

                # 領域の取得
                cx, cy = int(center[0]), int(center[1])
                region = self.get_person_region(cx, cy, x_grid, y_grid)
                row, col = region if region else (-1, -1)

                # CSVに結果を記録
                if csv_writer:
                    csv_writer.writerow([frame_idx, person_id, using_phone, row, col])

                # 状態表示
                if using_phone:
                    color = (0, 0, 255)  # 赤色
                    label = f"ID: {person_id}: Phone"
                else:
                    color = (255, 255, 0)  # 黄色
                    label = f"ID: {person_id}: Awake"

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
                    if pt[0] > 0 and pt[1] > 0:  # 有効な点のみ描画
                        cv2.circle(frame, tuple(pt[:2]), 3, (255, 255, 0), -1)

        # グリッドを描画（コメントアウト）
        # self.draw_monitor_grid(frame, self.split_ratios_cols, self.split_ratios)

        return frame


class IntegratedVideoProcessor:
    """統合動画処理システム（逆バレル補正版）"""

    def __init__(self, k1=-0.1, strength=1.0, zoom_factor=0.8, model_path="yolov8m-pose.pt"):
        self.corrector = VideoDistortionCorrector(k1, strength, zoom_factor)
        self.detector = DrowsinessDetectionSystem(model_path)

    def process_video(self, input_path, output_path, result_log="frame_results.csv", 
                    show_preview=True, apply_correction=True):
        """
        動画を処理（逆バレル歪み補正 + 居眠り検出）

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
        with open(result_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "person_id", "using_phone", "grid_row", "grid_col"])

            frame_idx = 0
            start_time = time.time()

            logger.info("動画処理を開始...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # 逆バレル歪み補正適用
                if apply_correction:
                    frame = self.corrector.apply_correction(frame)

                # 居眠り検出処理
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
                    display_title = "Integrated Video Processing (Inverse Barrel Correction)"
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

                    logger.info(
                        f"進行状況: {progress:.1f}% ({frame_idx}/{total_frames}) "
                        f"処理速度: {fps_current:.1f}fps 残り時間: {eta:.1f}秒"
                    )

        # リソース解放
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # 処理完了時間
        total_time = time.time() - start_time
        logger.info(f"動画処理完了!")
        logger.info(f"出力ファイル: {output_path}")
        logger.info(f"結果ログ: {result_log}")
        logger.info(f"処理時間: {total_time:.1f}秒")
        logger.info(f"平均処理速度: {frame_idx/total_time:.1f}fps")

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