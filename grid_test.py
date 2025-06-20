# 歪み補正を行い、3分割した領域で推論を行う
# 分割エリア毎の推論は実装ずみ、ID表示などはなし
from ultralytics import YOLO
import cv2
import numpy as np
import time
import math

RTSP_URL = "rtsp://6199:4003@192.168.100.183/live"
model = YOLO("yolov8m-pose.pt")

cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time = time.time()
frame_count = 0
last_keypoints_all, last_centers, last_bboxes = [], [], []

# カメラ補正用
K = None
dist_coeffs = None
new_K = None
map1 = None
map2 = None

def init_camera_params(w, h):
    global K, dist_coeffs, new_K, map1, map2
    fx = fy = w * 0.8
    cx = w / 2
    cy = h / 2
    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-0.3, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2)

def undistort_frame(frame):
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

def get_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(math.degrees(math.atan2(dy, dx)))

def is_lying_down(keypoints):
    try:
        shoulder_center = (keypoints[5] + keypoints[6]) / 2
        hip_center = (keypoints[11] + keypoints[12]) / 2
        angle = get_angle(shoulder_center, hip_center)
        return angle < 35 or angle > 145
    except:
        return False

def draw_horizontal_line(img, y_pos, color=(255, 255, 255), thickness=2):
    height, width = img.shape[:2]
    cv2.line(img, (0, y_pos), (width, y_pos), color, thickness)

# 分割位置（固定：2本の水平線 → 3分割）
split_positions = [430, 630]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width = frame.shape[:2]
    if K is None:
        init_camera_params(width, height)

    frame_undist = undistort_frame(frame)
    frame_count += 1

    # --- 分割線（直線）を描画 ---
    for y in split_positions:
        draw_horizontal_line(frame_undist, y)

    keypoints_all, centers, bboxes = [], [], []
    frames_to_process = []

    # --- 分割領域でフレームを分ける ---
    y_starts = [0] + split_positions
    y_ends = split_positions + [height]
    for y_start, y_end in zip(y_starts, y_ends):
        split_frame = frame_undist[y_start:y_end, :]
        frames_to_process.append((split_frame, 0, y_start))  # x_offset, y_offset

    # --- 推論処理（3フレームに1回） ---
    if frame_count % 3 == 0:
        for f, x_off, y_off in frames_to_process:
            results = model(f)
            for result in results:
                if result.keypoints is None or result.boxes is None:
                    continue
                keypoints_list = result.keypoints.xy.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for kps, box, conf in zip(keypoints_list, boxes, confs):
                    if conf < 0.4:
                        continue
                    kps[:, 0] += x_off
                    kps[:, 1] += y_off
                    box[0::2] += x_off
                    box[1::2] += y_off

                    selected_indices = [0, 5, 6, 11, 12]
                    pts = [kps[i] for i in selected_indices if i < len(kps)]
                    if not pts:
                        continue

                    keypoints_all.append(kps)
                    centers.append(np.mean(pts, axis=0))
                    bboxes.append(box)

        if centers:
            last_keypoints_all, last_centers, last_bboxes = keypoints_all, centers, bboxes
        else:
            keypoints_all, centers, bboxes = last_keypoints_all, last_centers, last_bboxes
    else:
        keypoints_all, centers, bboxes = last_keypoints_all, last_centers, last_bboxes

    # --- FPS 表示 ---
    fps = 1 / (time.time() - prev_time)
    prev_time = time.time()
    cv2.putText(frame_undist, f"FPS: {fps:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # --- 描画 ---
    for bbox, kps in zip(bboxes, keypoints_all):
        x1, y1, x2, y2 = map(int, bbox)
        label = "Sleeping" if is_lying_down(kps) else "Awake"
        color = (0, 0, 255) if label == "Sleeping" else (0, 255, 0)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(frame_undist, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(frame_undist, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.rectangle(frame_undist, (x1, y1), (x2, y2), color, 3)
        for point in kps:
            px, py = int(point[0]), int(point[1])
            cv2.circle(frame_undist, (px, py), 4, color, -1)

    cv2.imshow("Pose Sleep Detection (Undistorted)", frame_undist)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

