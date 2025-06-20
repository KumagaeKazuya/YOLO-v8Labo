# 歪み補正をした上で、3分割のエリアで推論を行う
# 分割エリア毎の推論は実装ずみ、ID表示などもありの完成版に近い
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
last_keypoints_all, last_centers, last_bboxes, last_area_indices = [], [], [], []

# 歪み補正パラメータ
K = dist_coeffs = new_K = map1 = map2 = None

def init_camera_params(w, h):
    global K, dist_coeffs, new_K, map1, map2
    fx = fy = w * 0.8
    cx = w / 2
    cy = h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
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
        # 信頼度でフィルタ（雑音除去）
        if keypoints.shape[1] == 3:
            confs = keypoints[:, 2]
            if np.any(confs[[5, 6, 11, 12]] < 0.3):  # 肩・腰
                return False

        # 中心点と角度
        shoulder_center = (keypoints[5][:2] + keypoints[6][:2]) / 2
        hip_center = (keypoints[11][:2] + keypoints[12][:2]) / 2
        angle = get_angle(shoulder_center, hip_center)

        # 補助：鼻のy座標の高さ（寝てると顔が低くなる）
        nose_y = keypoints[0][1]
        head_height = min(shoulder_center[1], hip_center[1])
        face_diff = abs(nose_y - head_height)

        return (angle < 35 or angle > 145) and face_diff < 80  # 高さ条件追加
    except:
        return False

def draw_horizontal_line(img, y_pos, color=(255, 255, 255), thickness=2):
    height, width = img.shape[:2]
    cv2.line(img, (0, y_pos), (width, y_pos), color, thickness)

# 分割位置（エリア3分割）
split_positions = [430, 630]
area_colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255)]  # エリア色

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

    # 分割線描画
    for y in split_positions:
        draw_horizontal_line(frame_undist, y)

    keypoints_all, centers, bboxes, area_indices = [], [], [], []
    frames_to_process = []

    # 分割領域切り出し
    y_starts = [0] + split_positions
    y_ends = split_positions + [height]
    for y_start, y_end in zip(y_starts, y_ends):
        split_frame = frame_undist[y_start:y_end, :]
        frames_to_process.append((split_frame, 0, y_start))  # offset記録

    if frame_count % 3 == 0:
        for i, (f, x_off, y_off) in enumerate(frames_to_process):
            results = model(f)
            h_split, w_split = f.shape[:2]
            for result in results:
                if result.keypoints is None or result.boxes is None:
                    continue
                kpts = result.keypoints.xyn.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for kps, box, conf in zip(kpts, boxes, confs):
                    if conf < 0.4:
                        continue

                    kps[:, 0] *= w_split
                    kps[:, 1] *= h_split
                    kps[:, 0] += x_off
                    kps[:, 1] += y_off
                    box[0::2] += x_off
                    box[1::2] += y_off

                    keypoints_all.append(kps)
                    centers.append(np.mean(kps[:, :2], axis=0))
                    bboxes.append(box)
                    area_indices.append(i)

        if centers:
            last_keypoints_all = keypoints_all
            last_centers = centers
            last_bboxes = bboxes
            last_area_indices = area_indices
        else:
            keypoints_all = last_keypoints_all
            centers = last_centers
            bboxes = last_bboxes
            area_indices = last_area_indices
    else:
        keypoints_all = last_keypoints_all
        centers = last_centers
        bboxes = last_bboxes
        area_indices = last_area_indices

    # FPS計測
    fps = 1 / (time.time() - prev_time)
    prev_time = time.time()
    cv2.putText(frame_undist, f"FPS: {fps:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # エリアごとの睡眠者数
    area_sleep_counts = [0, 0, 0]

    for bbox, kps, area_idx in zip(bboxes, keypoints_all, area_indices):
        x1, y1, x2, y2 = map(int, bbox)
        sleeping = is_lying_down(kps)
        label = "Sleeping" if sleeping else "Awake"
        color = (0, 0, 255) if sleeping else area_colors[area_idx]

        if sleeping:
            area_sleep_counts[area_idx] += 1

        cv2.rectangle(frame_undist, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_undist, f"{label}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        for point in kps:
            px, py = int(point[0]), int(point[1])
            cv2.circle(frame_undist, (px, py), 3, color, -1)

    # エリア別表示
    for i, count in enumerate(area_sleep_counts):
        msg = f"Area {i+1}: {count} Sleeping"
        cv2.putText(frame_undist, msg, (20, 70 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, area_colors[i], 2)

    cv2.imshow("Pose Sleep Detection", frame_undist)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
