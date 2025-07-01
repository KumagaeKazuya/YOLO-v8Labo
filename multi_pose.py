from ultralytics import YOLO
import cv2
import numpy as np
import time
import math
from collections import defaultdict, deque
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DrowsinessDetectionSystem:
    def __init__(self, rtsp_url, model_path="yolov8m-pose.pt"):
        # RTSP URLとモデルのパスを設定
        self.rtsp_url = rtsp_url
        self.model = YOLO(model_path)

        # グローバル変数(カメラキャリブレーションパラメータ)の定義
        self.K = None
        self.dist_coeffs = None
        self.new_K = None
        self.map1 = None
        self.map2 = None

        # グリッド分割設定(後でカメラ映像に合わせて調整を行う)
        self.split_ratios = [0.32, 0.08, 0.15, 0.46]
        self.split_ratios_cols = [0.13, 0.1, 0.1, 0.17, 0.15, 0.15, 0.12, 0.13, 0.13, 0.13]

        # 追跡とスムージング
        self.last_keypoints_all = []
        self.last_bboxes = []
        self.person_states = defaultdict(lambda: {
            'sleeping_history': deque(maxlen=10),
            'phone_history': deque(maxlen=10),
            'last_seen': 0
        })

        # 設定パラメータ
        self.config = {
            'conf_threshold': 0.4,
            'phone_distance_threshold': 100,
            'sleeping_angle_threshold': 45,
            'smoothing_frames': 5,
            'detection_interval': 3
        }

    def initialize_camera_params(self, w, h):
        fx = fy = w * 0.8
        cx = w / 2
        cy = h / 2
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([-0.3, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
        self.new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coeffs, (w, h), 1)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, self.dist_coeffs, None, self.new_K, (w, h), cv2.CV_16SC2)

    def correct_frame(self, frame):
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

    def caltulate_angle(self, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return abs(math.degrees(math.atan2(dy, dx)))

    def estimate_head_keypoint(self, keypoints):
        try:
            offset = 67
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            center = (left_shoulder + right_shoulder) / 2
            head_top = center - np.array([0, offset])
            return head_top
        except Exception as e:
            logger.error(f"頭頂点推定エラー: {e}")
            return None

    def detect_phone_usage(self, keypoints):
        try:
            nose = keypoints[0][:2]
            left_wrist = keypoints[9][:2]
            right_wrist = keypoints[10][:2]

            def valid(pt):
                return not np.any(np.isnan(pt)) and not np.allclose(pt, [0, 0])

            if not all(map(valid, [nose, left_wrist, right_wrist])):
                return False

            threshold = self.config['phone_distance_threshold']

            if not np.allclose(left_wrist, [0, 0]):
                dist_left = np.linalg.norm(nose - left_wrist)
                if dist_left < threshold:
                    return True

            if not np.allclose(right_wrist, [0, 0]):
                dist_right = np.linalg.norm(nose - right_wrist)
                if dist_right < threshold:
                    return True

            return False
        except Exception as e:
            logger.error(f"携帯使用判定エラー: {e} - keypoints shape: {keypoints.shape if hasattr(keypoints, 'shape') else 'Unknown'}")
            return False

    def detect_phone_usage_back_view(self, keypoints):
        try:
            threshold = 60
            neck = keypoints[1][:2]
            left_wrist = keypoints[9][:2]
            right_wrist = keypoints[10][:2]
            left_shoulder = keypoints[5][:2]
            right_shoulder = keypoints[6][:2]

            def valid(pt):
                return not np.any(np.isnan(pt)) and not np.allclose(pt, [0, 0])

            if not all(map(valid, [neck, left_wrist, right_wrist, left_shoulder, right_shoulder])):
                return False

            dist_left = min(np.linalg.norm(left_wrist - neck), np.linalg.norm(left_wrist - left_shoulder))
            dist_right = min(np.linalg.norm(right_wrist - neck), np.linalg.norm(right_wrist - right_shoulder))
            return min(dist_left, dist_right) < threshold
        except Exception as e:
            logger.error(f"背後視点携帯使用検出エラー: {e}")
            return False

    def detect_phone_usage_back_view(self, keypoints):
        """背後視点での携帯電話使用検出"""
        try:
            threshold = 60  # 実戦で調整可能な閾値(40-80推奨)

            neck = keypoints[1][:2]
            left_wrist = keypoints[9][:2]
            right_wrist = keypoints[10][:2]
            left_shoulder = keypoints[5][:2]
            right_shoulder = keypoints[6][:2]

            def valid(pt):
                return not np.any(np.isnan(pt)) and not np.allclose(pt, [0, 0])
            if not all(map(valid, [neck, left_wrist, right_wrist, left_shoulder, right_shoulder])):
                return False
            # 左手首と右手首の距離を計算
            dist_left = min(np.linalg.norm(left_wrist - neck), np.linalg.norm(left_wrist - left_shoulder))
            dist_right = min(np.linalg.norm(right_wrist - neck), np.linalg.norm(right_wrist - right_shoulder))
            return min(dist_left, dist_right) < thereshould
        except Exception as e:
            logger.error(f"背後視点携帯使用検出エラー: {e}")
            return False
def get_person_id(self, keypoints, existing_persons):
        if not existing_persons:
            return 0
        center = np.mean(keypoints[:, :2], axis=0)
        min_dist = float('inf')
        best_id = len(existing_persons)

        for person_id, last_center in existing_persons.items():
            dist = np.linalg.norm(center - last_center)
            if dist < min_dist and dist < 100:
                min_dist = dist
                best_id = person_id

        return best_id

def smooth_detection(self, person_id, sleeping, using_phone):
            # 誤検出を防ぐためのスムージング処理
            state = self.person_states[person_id]
            '''state['sleeping_history'].append(sleeping)#コメントアウト箇所の関連処理のため凍結'''
            state['phone_history'].append(using_phone)

            # 多数決による平滑化
            '''smoothed_sleeping = sum(state['sleeping_history']) > len(state['sleeping_history']) // 2'''
            smoothed_phone = sum(state['phone_history']) > len(state['phone_history']) // 2
            return smoothed_phone #睡眠判定を行う際はsmoothed_sleepingを戻り値に記載する

def draw_monitor_grid(self, img, col_ratios):
            # 画像にモニタリンググリッドを描画
            h, w = img.shape[:2]
            x_current = 0
            for ratio in col_ratios[:-1]:
                x_current += int(w * ratio)
                cv2.line(img, (x_current, 0), (x_current, h), color=(255, 255, 255), thickness=2)

def get_grid_boundaries(self, w, h, cols, rows):
            # グリッド境界の計算
            x_grid = [0]
            for ratio in cols:
                x_grid.append(x_grid[-1] + int(w * ratio))
            y_grid = [0]
            for ratio in rows:
                y_grid.append(y_grid[-1] + int(h * ratio))
            return x_grid, y_grid

def get_person_region(self, cx, cy, x_grid, y_grid):
            # 人物のグリッド領域を取得
            col_idx = -1
            row_idx = -1
            for i in range(len(x_grid) - 1):
                if x_grid[i] <= cx < x_grid[i + 1]:
                    col_idx = i
                    return
            for j in range(len(y_grid) - 1):
                if y_grid[j] <= cy < y_grid[j + 1]:
                    row_idx = j
                    return
            if col_idx == -1 or row_idx == -1:
                return None
            return col_idx, row_idx

def draw_detection_results(self, frame, keypoints_all, bboxes, x_grid, y_grid):
            # フレームに全ての検出結果を描画
            existing_persons = {}

            for i, (kps, box) in enumerate(zip(keypoints_all, bboxes)):
                if kps.shape[0] < 17:
                    continue

            # 人物IDの取得・追跡
            person_id = self.get_person_id(kps, existing_persons)
            center = np.mean(kps[:, :2], axis=0)
            existing_persons[person_id] = center

            cx, cy = int(center[0]), int(center[1])
            region = self.get_person_position(cx, cy, x_grid, y_grid)

            # 頭部推定と姿勢検出
            head_top = self.estimate_head_top(kps)
            if head_top is not None:
                return
            #sleeping = self.detect_sleeping_posure(kps, head_top)
            using_phone = self.detect_phone_usage(kps) # 正面向き判定
            #using_phone_back = self.detect_phone_usage_back_view(kps)  # 背後向き判定。後に切り替え

            # スムージング処理
            #sleeping, using_phone = self.smooth_detection(person_id, sleeping, using_phone)睡眠判定の際はこちらのコードを使用
            using_phone = self.smooth_detection(person_id, using_phone)

            # ステータスと色の決定
            if using_phone:
    # if sleeping and using_phone:  # 睡眠と携帯使用の両方を検出
                color = (0, 0, 255)  # 赤色
                label = f"ID: {person_id}: Phone"
    # label = f"ID{person_id}: Sleeping+Phone" 睡眠判定も行う場合はこちらのコードを使用
                status = "DANGER"

            else:
                color = (255, 255, 0)  # 黄色
                label = f"ID: {person_id}: Awake"
                status = "AWAKE"

'''# 睡眠判定も追加するコード
if sleeping and using_phone:
    color = (0, 0, 255)
    label = f"ID{person_id}: Sleeping+Phone"
    status = "DANGER"
elif sleeping:
    color = (0, 255, 0)  # 緑色
    label = f"ID: {person_id}: Sleeping"
    status = "SLEEPING"
elif using_phone:
    color = (255, 0, 0)  # 青色
    label = f"ID: {person_id}: Phone"
    status = "PHONE"
    else:
    color = (0, 255, 0)
    label = f"ID{person_id}: Awake"
    status = "NORMAL"
'''
# バウンディングボックスとラベルの描画
x1, y1, x2, y2 = map(int, box)
cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# キーポイントの描画
for j, (px, py) in enumerate(kps, astype(int)):
            if j == 1: # 首 - 青
                cv2.circle(frame, (px, py), 6, (255, 0, 0), -1)
            else: # そのほかのキーポイント - シアン
                cv2.circle(frame, (px, py), 4, (255, 255, 0), -1)

# 推定頭頂部 - 白
head_px, head_py = head_top.astype(int)
cv2.circle(frame, (head_px, head_py), 6, (255, 255, 255), -1)

# 問題のあるケースの領域情報描画
if sleeping and region is not None:
    row, col = region
    shoulder_center = ((kps[5] + kps[6]) / 2).astype(int)
    cv2.putText(frame, f"{status} ({row}, {col})", (shoulder_center[0], shoulder_center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def run(self):
    # メイン検出ループ
    cap = cv2.VideoCapture(self.rtsp_url)
    if not cap.isOpened():
        logger.error("RTSPカメラに接続できませんでした")
        return

    # GUI作成
    window_name = "Drowsiness Detection"
    cv2.namedWindow(window_name)
    # ヘッドオフセットは固定のためトラックバーは不要。必要な時はコメントアウトを外す
    # cv2.createTrackbar("Head Y-Offset", window_name, 40, 100, lambda x: None)
    cv2.createTrackbar("Phone Distance", window_name, self.config['phone_distance_threshold'], 200, lambda x: None)
    cv2.createTrackbar("Debug Mode", window_name, 0, 1, lambda x: None)

    prev_time = time.time()
    frame_count = 0
    last_keypoints_all = []

    logger.info("検出システム開始")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("フレーム取得失敗")
                return

            height, width = frame.shape[:2]
            if self.K is None:
                self.initialize_camera_params(width, height)

            # 歪み補正適用
            frame_undist = self.correct_distortion(frame)
            frame_count += 1

            # トラックバーから設定更新
            self.config['phone_distance_theresold'] = cv2.getTrackbarPos("Phone Distance", window_name)
            debug_mode = cv2.getTrackbarPos("Debug Mode", window_name)

            keypoints_all = []
            bboxes = []

            # フレーム区間処理
            if frame_count % self.config['detection_interval'] == 0:
                y_current = 0
                for ratio in self.split_ratios:
                    y_start = int(y_current)
                    y_end = int(y_current + height * ratio)
                    section = frame_undist[y_start:y_end, :]

            # YOLO検出実行
            results = self.model(section, conf=self.config['conf_threshold'], verbose=False)
            for result in results:
                if result.keypoints is None or result.boxes is None:
                    continue

                keypoints_list = result.keypoints.xy.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for kps, box, conf in zip(keypoints_list, boxes, confs):
                    if conf < self.config['conf_threshold']:
                        continue

                    # デバッグ情報(初回のみ)
                    if debug_mode and not debug_printed:
                        logger.info(f"キーポイント形状: {kps.shape}")
                        logger.info(f"キーポイント数: len{kps}")
                        if len(kps) > 10:
                            logger.info(f"鼻 (0): {kps[0]}")
                            logger.info(f"左手首 (9): {kps[9]}")
                            logger.info(f"右手首 (10): {kps[10]}")
                        debug_printed = True

                    # 区間オフセットに対する座標調整
                    kps[:, 1] += y_start
                    box[1::2] += y_start

                    keypoints_all.append(kps)
                    bboxes.append(box)

            y_current = y_end

            # キャッシュ結果更新
            if keypoints_all:
                self.last_keypoints_all, self.last_bboxes = keypoints_all, bboxes
            else:
                keypoints_all, bboxes = self.last_keypoints_all, self.last_bboxes
        else:
            keypoints_all, bboxes = self.last_keypoints_all, self.last_bboxes

            # グリッド描画
            y_sum = 0
            for ratio in self.split_ratios[:-1]:
                y_sum += int(height * ratio)
                cv2.line(frame_undist, (0, y_sum), (width, y_sum), (255, 255, 255), 2)

            self.draw_monitor_grid(frame_undist, self.split_ratios_cols)
            x_grid, y_grid = self.calculate_grid_boundaries(width, height,
                                                        self.split_ratios_cols, self.split_ratios)

# FPS計算
            now = time.time()
            fps = 1 / (now - prev_time)
            prev_time = now
            cv2.putText(frame_undist, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

# 検出結果描画
            if keypoints_all:
                self.draw_detection_results(frame_undist, keypoints_all, bboxes, x_grid, y_grid)

            cv2.imshow(window_name, frame_undist)

            if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
                break

    except KeyboardInterrupt:
        logger.info("ユーザーによる中断")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("終了")

def main():
    # 設定
    RTSP_URL = "rtsp://6199:4003@192.168.100.183/live"
    MODEL_PATH = "yolov8m-pose.pt"

    # 検出システム作成・実行
    detector = DrowsinessDetectionSystem(RTSP_URL, MODEL_PATH)
    detector.run()

if __name__ == "__main__":
    main()