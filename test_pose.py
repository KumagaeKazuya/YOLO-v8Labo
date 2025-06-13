from ultralytics import YOLO
import cv2
import numpy as np
import time
import math
from scipy.spatial.distance import cdist

RTSP_URL = "rtsp://6199:4003@192.168.100.183/live"
model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time = time.time()
prev_centers = []
frame_count = 0

last_keypoints_all = []
last_centers = []
last_bboxes = []

STABILITY_THRESHOLD = 3  # 安定判定のフレーム数
stability_counters = {}

def get_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return abs(angle)

def is_lying_down(keypoints):
    try:
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        angle = get_angle(shoulder_center, hip_center)
        return angle < 30 or angle > 150
    except:
        return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    keypoints_all = []
    centers = []
    bboxes = []

    # 3フレームごとに検出
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

                selected_indices = [0, 5, 6, 11, 12]
                pts = [kps[i] for i in selected_indices if i < len(kps)]
                if len(pts) == 0:
                    continue

                center = np.mean(pts, axis=0)
                keypoints_all.append(kps)
                centers.append(center)
                bboxes.append(box)

        if len(centers) > 0:
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

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    for bbox, kps in zip(bboxes, keypoints_all):
        x1, y1, x2, y2 = map(int, bbox)

        if is_lying_down(kps):
            label = "Sleeping"
            color = (0, 0, 255)  # 赤
        else:
            label = "Awake"
            color = (0, 255, 0)  # 緑

        # ラベル背景描画
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        for point in kps:
            px, py = int(point[0]), int(point[1])
            cv2.circle(frame, (px, py), 4, color, -1)

    cv2.imshow("Pose Sleep Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
