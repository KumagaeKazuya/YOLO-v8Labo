from ultralytics import YOLO
import cv2
import numpy as np
import time
from scipy.spatial.distance import cdist

# RTSP URL（例）
RTSP_URL = "rtsp://6199:4003@192.168.100.183/live"

# モデル読み込み
model = YOLO("yolov8n-pose.pt")

# 解像度設定 & RTSP接続
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# FPS & ID追跡用
prev_time = time.time()
prev_centers = []
prev_ids = []
next_id = 0
frame_count = 0

# 前回の検出結果を保存
last_keypoints_all = []
last_centers = []
last_ids = []

# ------------------------
# IDの割り当て関数（距離しきい値を拡大）
# ------------------------
def assign_ids(current_centers, previous_centers, previous_ids, threshold=80):
    global next_id
    ids = []

    if len(previous_centers) == 0:
        for _ in current_centers:
            ids.append(next_id)
            next_id += 1
        return ids, current_centers

    distances = cdist(current_centers, previous_centers)
    for i, row in enumerate(distances):
        min_idx = np.argmin(row)
        if row[min_idx] < threshold:
            ids.append(previous_ids[min_idx])
        else:
            ids.append(next_id)
            next_id += 1

    return ids, current_centers

# ------------------------
# 姿勢の角度判定関数（寝ている判定など任意）
# ------------------------
def is_lying_down(keypoints, threshold=30):
    try:
        ls, rs = keypoints[5], keypoints[6]
        lh, rh = keypoints[11], keypoints[12]
        shoulder_mid = (ls + rs) / 2
        hip_mid = (lh + rh) / 2
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        angle = abs(np.arctan2(dy, dx) * 180 / np.pi)
        return angle < threshold
    except:
        return False

# ------------------------
# メインループ
# ------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    keypoints_all = []
    centers = []
    ids = []

    # 推論は 3フレームに1回のみ実施（疑似FPS向上）
    if frame_count % 3 == 0:
        results = model(frame)
        for result in results:
            if result.keypoints is None:
                continue
            keypoints_list = result.keypoints.xy.cpu().numpy()
            for kps in keypoints_list:
                # 改善案③：中心点を安定した複数キーポイントから計算
                selected_indices = [0, 5, 6, 11, 12]
                pts = [kps[i] for i in selected_indices if i < len(kps)]
                if len(pts) == 0:
                    continue
                center = np.mean(pts, axis=0)
                keypoints_all.append(kps)
                centers.append(center)
        
        if len(centers) > 0:
            ids, prev_centers = assign_ids(centers, prev_centers, prev_ids)
            prev_ids = ids
            last_keypoints_all = keypoints_all
            last_centers = centers
            last_ids = ids
        else:
            # 推論失敗時は前フレームを保持
            keypoints_all = last_keypoints_all
            centers = last_centers
            ids = last_ids
    else:
        # 推論スキップ時は前回の推論結果を使い回す
        keypoints_all = last_keypoints_all
        centers = last_centers
        ids = last_ids

    # FPS計測
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # 表示：関節 + ID + 判定
    for kps, pid in zip(keypoints_all, ids):
        is_lying = is_lying_down(kps)
        color = (0, 0, 255) if is_lying else (0, 255, 0)

        for x, y in kps:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)

        center_x, center_y = np.mean(kps[:, 0]), np.mean(kps[:, 1])
        cv2.putText(frame, f"ID:{pid}", (int(center_x), int(center_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Pose Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
