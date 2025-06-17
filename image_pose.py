#Atomcamの映像をそのまま撮ってくるだけのコード
import cv2
# RTSPストリームのURL（必要に応じて変更）
rtsp_url = "rtsp://6199:4003@192.168.100.183/live"

# 映像取得
cap = cv2.VideoCapture(rtsp_url)

# フレーム取得確認
ret, frame = cap.read()
if not ret:
    print("カメラ映像を取得できませんでした。")
    cap.release()
    exit()

# 解像度を確認（任意）
height, width = frame.shape[:2]
print(f"[INFO] 解像度: {width}x{height}")

# ループで映像表示
while True:
    ret, frame = cap.read()
    if not ret:
        print("映像取得に失敗しました。")
        break

    cv2.imshow("Camera Feed Only", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()
