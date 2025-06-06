import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# 日本語ラベル
keypoint_labels_ja = [
    '鼻', '左目', '右目', '左耳', '右耳',
    '左肩', '右肩', '左ひじ', '右ひじ',
    '左手首', '右手首', '左腰', '右腰',
    '左ひざ', '右ひざ', '左足首', '右足首'
]

# フォント（macOS用）
font_path = "/Library/Fonts/Arial.ttf"  # Arial フォントのパスを指定
# フォントファイルの存在確認
import os
if not os.path.exists(font_path):
    raise FileNotFoundError(f"指定されたフォントファイルが見つかりません: {font_path}")
font = ImageFont.truetype(font_path, size=16)

# YOLOv8 Pose モデル
model = YOLO('yolov8n-pose.pt')

# 動画ファイルの読み込み
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# 出力動画設定
save_output = True
output_path = "pose_output.mp4"
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if save_output:
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO推論（フレームごとに検出）
    results = model(frame)

    # PIL形式で描画処理
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    for result in results:
        if result.keypoints is None:
            continue
        for person_kps in result.keypoints.xy:
            for i, (x, y) in enumerate(person_kps):
                x, y = int(x), int(y)
                if x > 0 and y > 0:
                    draw.ellipse((x-3, y-3, x+3, y+3), fill=(0, 255, 0))
                    draw.text((x+5, y-10), keypoint_labels_ja[i], font=font, fill=(255, 0, 0))

    # OpenCVに変換して表示・保存
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("YOLOv8 Pose Detection (Video)", frame)
    if save_output:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
print(f"検出完了: {frame_count} フレームを処理しました。保存先: {output_path}")
