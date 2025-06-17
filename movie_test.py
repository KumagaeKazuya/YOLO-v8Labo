#test_pose.pyのテスト用ファイル
from ultralytics import YOLO
import cv2
import numpy as np
import time
import math
from scipy.interpolate import splprep, splev

# === 定数定義 ===
RTSP_URL = "rtsp://6199:4003@192.168.100.183/live"
model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0
prev_time = time.time()
last_keypoints_all = []
last_centers = []
last_bboxes = []

# === 関数定義 ===
def get_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(math.degrees(math.atan2(dy, dx)))

def is_lying_down(keypoints):
    try:
        shoulder_center = (keypoints[5] + keypoints[6]) / 2
        hip_center = (keypoints[11] + keypoints[12]) / 2
        angle = get_angle(shoulder_center, hip_center)
        return angle < 30 or angle > 150
    except:
        return False

def smooth_curve(points, smoothness=0):
    x = points[:, 0]
    y = points[:, 1]
    tck, _ = splprep([x, y], s=smoothness)
    unew = np.linspace(0, 1.0, 100)
    out = splev(unew, tck)
    return np.array([out[0], out[1]]).T.astype(np.int32)

# === 分割線の事前計算 ===
def create_smooth_lines(frame_width, frame_height):
    line1 = np.array([
    [0, int(frame_height * 0.425)],
    [int(frame_width * 0.25), int(frame_height * 0.385)],
    [int(frame_width * 0.5), int(frame_height * 0.375)],
    [int(frame_width * 0.75), int(frame_height * 0.385)],
    [frame_width, int(frame_height * 0.425)]
    ], np.int32)
    line2 = np.array([
    [0, int(frame_height * 0.56)],
    [int(frame_width * 0.25), int(frame_height * 0.575)],
    [int(frame_width * 0.5), int(frame_height * 0.59)],
    [int(frame_width * 0.75), int(frame_height * 0.5825)],
    [frame_width, int(frame_height * 0.57)]
    ], np.int32)

    smooth_line1 = smooth_curve(line1)
    smooth_line2 = smooth_curve(line2)

    return smooth_line1, smooth_line2

# === メインループ ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    height, width = frame.shape[:2]

    # 分割線の生成を毎フレームのサイズに合わせて更新
    smooth_line1, smooth_line2 = create_smooth_lines(width, height)

    keypoints_all = []
    centers = []
    bboxes = []

    # === 推論：3フレームごとに実行 ===
    if frame_count % 3 == 0:
        results = model(frame)
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue
            keypoints_list = result.keypoints.xy.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for kps, box, conf in zip(keypoints_list, boxes, confs):
                if conf < 0.5:
                    continue
                selected = [kps[i] for i in [0, 5, 6, 11, 12] if i < len(kps)]
                if not selected:
                    continue
                center = np.mean(selected, axis=0)
                keypoints_all.append(kps)
                centers.append(center)
                bboxes.append(box)

        # 前回保持データ更新
        if keypoints_all:
            last_keypoints_all = keypoints_all
            last_centers = centers
            last_bboxes = bboxes
        else:
            keypoints_all = last_keypoints_all
            centers = last_centers
            bboxes = last_bboxes
    else:
        keypoints_all = last_keypoints_all
        centers = last_centers
        bboxes = last_bboxes

    # === FPS表示 ===
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # === 検出描画 ===
    for bbox, kps in zip(bboxes, keypoints_all):
        x1, y1_box, x2, y2_box = map(int, bbox)

        label = "Sleeping" if is_lying_down(kps) else "Awake"
        color = (0, 0, 255) if label == "Sleeping" else (0, 255, 0)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(frame, (x1, y1_box - h - 10), (x1 + w, y1_box), color, -1)
        cv2.putText(frame, label, (x1, y1_box - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.rectangle(frame, (x1, y1_box), (x2, y2_box), color, 3)

        for point in kps:
            px, py = int(point[0]), int(point[1])
            cv2.circle(frame, (px, py), 4, color, -1)

    # === 分割線の描画 ===
    cv2.polylines(frame, [smooth_line1], isClosed=False, color=(0, 0, 255), thickness=3)
    cv2.polylines(frame, [smooth_line2], isClosed=False, color=(0, 0, 255), thickness=3)

    # === 表示と終了処理 ===
    cv2.imshow("Pose Sleep Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
