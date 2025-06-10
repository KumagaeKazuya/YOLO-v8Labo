from ultralytics import YOLO
import cv2
import numpy as np

# RTSPストリームのURL（必要に応じて /live_hd などに変更）
rtsp_url = "rtsp://6199:4003@192.168.100.183/live"

# モデル読み込み
model = YOLO("yolov8n-pose.pt")

# RTSPストリームから映像取得
cap = cv2.VideoCapture(rtsp_url)

# 解像度表示（最初のフレーム取得時）
ret, test_frame = cap.read()
if not ret:
    print("初回フレームの取得に失敗しました。RTSP URLやネットワーク設定を確認してください。")
    exit()

height, width = test_frame.shape[:2]
print(f"[INFO] 現在の解像度: {width}x{height}")

# ★ 解像度をソフト的に上げたい場合（例：1920x1080に拡大）
resize_target = (1920, 1080)

def is_lying_down(keypoints, threshold=30):
    if keypoints.shape[0] < 17:
        return False

    ls, rs = keypoints[5], keypoints[6]
    lh, rh = keypoints[11], keypoints[12]

    shoulder_mid = (ls + rs) / 2
    hip_mid = (lh + rh) / 2

    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]

    angle = np.arctan2(dy, dx) * 180 / np.pi
    angle = abs(angle)

    return angle < threshold

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレーム取得失敗。RTSP接続やネットワークを確認してください。")
        break

    # 解像度変更（リサイズ）したい場合はこちら
    frame = cv2.resize(frame, resize_target)

    results = model(frame)

    for result in results:
        keypoints_list = result.keypoints.xy.cpu().numpy()

        if keypoints_list.shape[0] == 0:
            continue

        for keypoints in keypoints_list:
            if keypoints.shape[0] < 17:
                continue

            person_is_lying = is_lying_down(keypoints)
            color = (0, 0, 255) if person_is_lying else (0, 255, 0)

            for x, y in keypoints:
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    cv2.imshow("Pose Detection (Lie Detection)", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
