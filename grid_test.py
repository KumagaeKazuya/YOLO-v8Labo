import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os

# 日本語ラベル
keypoint_labels_ja = [
    '鼻', '左目', '右目', '左耳', '右耳',
    '左肩', '右肩', '左ひじ', '右ひじ',
    '左手首', '右手首', '左腰', '右腰',
    '左ひざ', '右ひざ', '左足首', '右足首'
]

# フォント設定（macOS用）
font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W5.ttc"
if not os.path.exists(font_path):
    raise FileNotFoundError(f"指定されたフォントファイルが見つかりません: {font_path}")
font = ImageFont.truetype(font_path, size=16)

# モデルと動画読み込み
model = YOLO('yolov8n-pose.pt')
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

# グリッド設定
cols = 3
rows = 2
cell_w = w // cols
cell_h = h // rows

# フレーム処理ループ
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # グリッド線の描画
    for i in range(rows):
        for j in range(cols):
            x0 = j * cell_w
            y0 = i * cell_h
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline=(100, 100, 100))
            draw.text((x0 + 5, y0 + 5), f"{i},{j}", fill=(100, 100, 100), font=font)

    # 人体キーポイント描画と位置特定
    for result in results:
        if result.keypoints is None:
            continue
        for person_kps in result.keypoints.xy:
            for i, (x, y) in enumerate(person_kps):
                if x <= 0 or y <= 0:
                    continue
                col = int(x) // cell_w
                row = int(y) // cell_h
                draw.ellipse((x-3, y-3, x+3, y+3), fill=(0, 255, 0))
                if 0 <= col < cols and 0 <= row < rows:
                    draw.text((x+5, y-10), f"{keypoint_labels_ja[i]} ({row},{col})", fill=(255, 0, 0), font=font)

    # 表示と保存
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
