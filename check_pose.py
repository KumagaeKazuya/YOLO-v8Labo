from ultralytics import YOLO
import cv2
import numpy as np
import time
import math
RTSP_URL = "rtsp://6199:4003@192.168.100.183/live"
model = YOLO("yolov8m-pose.pt")  # mediumモデルに変更
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
prev_time = time.time()
frame_count = 0
last_keypoints_all, last_centers, last_bboxes = [], [], []
def undistort_frame(frame):
    h, w = frame.shape[:2]
    # 画像サイズに合わせて仮カメラ行列をスケーリング
    fx = fy = w * 0.8
    cx = w / 2
    cy = h / 2
    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-0.3, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2)
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
        # 閾値を少し緩和
        return angle < 35 or angle > 145
    except:
        return False
def draw_curved_line(img, y_base, amplitude=50, color=(255, 255, 255), thickness=2):
    height, width = img.shape[:2]
    points = []
    for x in range(0, width, 5):
        norm_x = (2 * x / width) - 1
        offset = amplitude * (norm_x**4 - norm_x**2)
        y = int(y_base + offset)
        points.append((x, y))
    for i in range(1, len(points)):
        cv2.line(img, points[i-1], points[i], color, thickness)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame = undistort_frame(frame)
    frame_count += 1
    height, width = frame.shape[:2]
    draw_curved_line(frame, int(height * 0.25), amplitude=20)
    draw_curved_line(frame, int(height * 0.45), amplitude=50)
    keypoints_all, centers, bboxes = [], [], []
    frames_to_process = []
    split_ratios = [0.4, 0.35, 0.25]
    y_current = 0
    for ratio in split_ratios:
        y_start = int(y_current)
        y_end = int(y_current + height * ratio)
        split_frame = frame[y_start:y_end, :]
        frames_to_process.append((split_frame, 0, y_start))
        y_current = y_end
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
                    if conf < 0.4:  # 閾値を0.5→0.4に下げた
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
    fps = 1 / (time.time() - prev_time)
    prev_time = time.time()
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    for bbox, kps in zip(bboxes, keypoints_all):
        x1, y1, x2, y2 = map(int, bbox)
        label = "Sleeping" if is_lying_down(kps) else "Awake"
        color = (0, 0, 255) if label == "Sleeping" else (0, 255, 0)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        for point in kps:
            px, py = int(point[0]), int(point[1])
            cv2.circle(frame, (px, py), 4, color, -1)
    cv2.imshow("Pose Sleep Detection (Undistorted)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()