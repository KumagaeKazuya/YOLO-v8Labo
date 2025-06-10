from ultralytics import YOLO
import cv2
import numpy as np

# RTSPストリームのURL（環境に合わせて書き換えてください）
rtsp_url = "rtsp://6199:4003@192.168.100.183/live"

# モデル読み込み
model = YOLO("yolov8n-pose.pt")

# RTSPストリームから映像取得
cap = cv2.VideoCapture(rtsp_url)

def is_lying_down(keypoints, threshold=30):
    if keypoints.shape[0] < 17:
        return False

    # COCOキーポイント: 左肩5, 右肩6, 左腰11, 右腰12
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

    results = model(frame)

    for result in results:
        keypoints_list = result.keypoints.xy.cpu().numpy()

        if keypoints_list.shape[0] == 0:
            continue  # 検出なしスキップ

        for keypoints in keypoints_list:
            if keypoints.shape[0] < 17:
                continue  # キーポイント不足スキップ

            person_is_lying = is_lying_down(keypoints)
            color = (0, 0, 255) if person_is_lying else (0, 255, 0)

            for x, y in keypoints:
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    cv2.imshow("Pose Detection (Lie Detection)", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()