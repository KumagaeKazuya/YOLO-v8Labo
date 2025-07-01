# image_pose.pyの動画推論用テストファイル

# 動作可能・機能維持版 DrowsinessDetectionSystem
# ファイル名例: drowsiness_system.py

from ultralytics import YOLO
import cv2
import numpy as np
import time
import math
from collections import defaultdict, deque
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DrowsinessDetectionSystem:
    def __init__(self, rtsp_url, model_path="yolov8m-pose.pt"):
        self.rtsp_url = rtsp_url
        self.model = YOLO(model_path)

        self.K = self.dist_coeffs = self.new_K = self.map1 = self.map2 = None

        self.split_ratios = [0.32, 0.08, 0.15, 0.46]
        self.split_ratios_cols = [0.13, 0.1, 0.1, 0.17, 0.15, 0.15, 0.12, 0.13, 0.13, 0.13]

        self.last_keypoints_all = []
        self.last_bboxes = []
        self.person_states = defaultdict(lambda: {
            'sleeping_history': deque(maxlen=10),
            'phone_history': deque(maxlen=10),
            'last_seen': 0
        })

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

    def smooth_detection(self, person_id, using_phone):
        state = self.person_states[person_id]
        state['phone_history'].append(using_phone)
        smoothed_phone = sum(state['phone_history']) > len(state['phone_history']) // 2
        return smoothed_phone

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
            return min(np.linalg.norm(nose - left_wrist), np.linalg.norm(nose - right_wrist)) < threshold
        except Exception as e:
            logger.error(f"携帯使用判定エラー: {e}")
            return False

    def draw_monitor_grid(self, img, col_ratios):
        h, w = img.shape[:2]
        x_current = 0
        for ratio in col_ratios[:-1]:
            x_current += int(w * ratio)
            cv2.line(img, (x_current, 0), (x_current, h), (255, 255, 255), 2)

    def calculate_grid_boundaries(self, w, h, cols, rows):
        x_grid = [0]
        for ratio in cols:
            x_grid.append(x_grid[-1] + int(w * ratio))
        y_grid = [0]
        for ratio in rows:
            y_grid.append(y_grid[-1] + int(h * ratio))
        return x_grid, y_grid

    def get_person_region(self, cx, cy, x_grid, y_grid):
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

    def draw_detection_results(self, frame, keypoints_all, bboxes, x_grid, y_grid):
        existing_persons = {}
        for kps, box in zip(keypoints_all, bboxes):
            if kps.shape[0] < 17:
                continue
            person_id = self.get_person_id(kps, existing_persons)
            center = np.mean(kps[:, :2], axis=0)
            existing_persons[person_id] = center
            cx, cy = int(center[0]), int(center[1])
            region = self.get_person_region(cx, cy, x_grid, y_grid)
            head_top = self.estimate_head_keypoint(kps)
            using_phone = self.detect_phone_usage(kps)
            using_phone = self.smooth_detection(person_id, using_phone)

            # 状態表示
            if using_phone:
                color = (0, 0, 255)
                label = f"ID: {person_id}: Phone"
            else:
                color = (255, 255, 0)
                label = f"ID: {person_id}: Awake"

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            for pt in kps.astype(int):
                cv2.circle(frame, tuple(pt[:2]), 3, (255, 255, 0), -1)
            if head_top is not None:
                cv2.circle(frame, tuple(head_top.astype(int)), 6, (255, 255, 255), -1)

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logger.error("RTSPカメラに接続できません")
            return

        window_name = "Drowsiness Detection"
        cv2.namedWindow(window_name)
        cv2.createTrackbar("Phone Distance", window_name, self.config['phone_distance_threshold'], 200, lambda x: None)
        prev_time = time.time()
        frame_count = 0
        debug_printed = False

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("フレーム取得失敗")
                break

            h, w = frame.shape[:2]
            if self.K is None:
                self.initialize_camera_params(w, h)

            frame_undist = self.correct_frame(frame)
            frame_count += 1
            self.config['phone_distance_threshold'] = cv2.getTrackbarPos("Phone Distance", window_name)
            keypoints_all = []
            bboxes = []

            if frame_count % self.config['detection_interval'] == 0:
                y_current = 0
                for ratio in self.split_ratios:
                    y_start = int(y_current)
                    y_end = int(y_current + h * ratio)
                    section = frame_undist[y_start:y_end, :]

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
                            kps[:, 1] += y_start
                            box[1::2] += y_start
                            keypoints_all.append(kps)
                            bboxes.append(box)
                    y_current = y_end

            if keypoints_all:
                self.last_keypoints_all, self.last_bboxes = keypoints_all, bboxes
            else:
                keypoints_all, bboxes = self.last_keypoints_all, self.last_bboxes

            y_sum = 0
            for ratio in self.split_ratios[:-1]:
                y_sum += int(h * ratio)
                cv2.line(frame_undist, (0, y_sum), (w, y_sum), (255, 255, 255), 2)

            self.draw_monitor_grid(frame_undist, self.split_ratios_cols)
            x_grid, y_grid = self.calculate_grid_boundaries(w, h, self.split_ratios_cols, self.split_ratios)

            now = time.time()
            fps = 1 / (now - prev_time)
            prev_time = now
            cv2.putText(frame_undist, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if keypoints_all:
                self.draw_detection_results(frame_undist, keypoints_all, bboxes, x_grid, y_grid)

            cv2.imshow(window_name, frame_undist)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    rtsp_url = "rtsp://6199:4003@192.168.100.183/live"
    model_path = "yolov8m-pose.pt"
    detector = DrowsinessDetectionSystem(rtsp_url, model_path)
    detector.run()


if __name__ == "__main__":
    main()