# 現在使われているYOLOモデルのレイヤー構造を確認するスクリプト
# レイヤー構造を確認して、モデルのバージョンを確認
from ultralytics import YOLO

# モデルファイルを指定（yolov8n.ptやyolov11n.ptなど）
model = YOLO("yolov8m-pose.pt")

# モデル構造（nn.Module）のクラス名一覧を表示
print("🔍 使用レイヤー一覧（ユニーク名）:")
layer_types = set(type(layer).__name__ for layer in model.model.modules())
for layer_type in sorted(layer_types):
    print(" -", layer_type)

# フル構造も確認したい場合（コメントアウトを外すと大量出力）
print(model.model)
