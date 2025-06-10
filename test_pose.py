from ultralytics import YOLO
import cv2
import numpy as np
import threading

# ✅ RTSPのURL（自分のカメラに置き換えてください）
RTSP_URL = "rtsp://6199:4003@192.168.100.183/live"

# ✅ カメラをスレッドで読み続けるクラス（常に最新フレームを保持）
class VideoCaptureThread:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# ✅ 人が「寝ている」か判定する関数
def is_lying_down(keypoints, threshold=30):
    try:
        ls, rs = keypoints[5], keypoints[6]   # 左右の肩
        lh, rh = keypoints[11], keypoints[12] # 左右の腰

        # 中心点を計算
        shoulder_mid = (ls + rs) / 2
        hip_mid = (lh + rh) / 2

        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        angle = abs(np.arctan2(dy, dx) * 180 / np.pi)

        return angle < threshold
    except IndexError:
        return False

# ✅ モデル読み込み（nanoモデルが高速）
model = YOLO("yolov8n-pose.pt")

# ✅ カメラ起動（スレッド）
stream = VideoCaptureThread(RTSP_URL)

while True:
    ret, frame = stream.read()
    if not ret:
        continue

    # YOLOv8 Pose 推論（入力サイズ小さめで高速化）
    results = model(frame, imgsz=384)

    for result in results:
        if result.keypoints is None:
            continue
        keypoints_list = result.keypoints.xy.cpu().numpy()  # (人数, 17, 2)

        for keypoints in keypoints_list:
            if keypoints.shape[0] < 13:  # 必須キーポイントが不足していたらスキップ
                continue

            is_lying = is_lying_down(keypoints)
            color = (0, 0, 255) if is_lying else (0, 255, 0)

            # キーポイント描画
            for x, y in keypoints:
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    cv2.imshow("YOLOv8 Pose RTSP View", frame)
    if cv2.waitKey(1) == ord("q"):
        break

stream.stop()
cv2.destroyAllWindows()
