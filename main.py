# main.py

import os
from scripts.distortion import IntegratedVideoProcessor
from scripts.downloader import download_file_from_google_drive
import logging

# ログ設定（再定義しておくと便利）
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """メイン関数"""
    # 設定
    google_drive_file_id = "1QaYIFAlXRqcThZU9aLGWWQTCUXs6WJCU"
    video_path = download_file_from_google_drive(google_drive_file_id)
    #video_path = "shoot5.mp4"  # 入力動画パス
    output_dir = "videos"

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # 統合処理システムを初期化（逆バレル補正版）
    processor = IntegratedVideoProcessor(
        k1=-0.1,           # 歪み係数（負の値で樽型歪みを補正）
        strength=1.0,      # 補正強度
        zoom_factor=1.3,   # ズーム倍率（1.0未満で引き、1.0超で寄り）
        model_path="models/yolov8m-pose.pt"
    )

    # 動画ファイルの存在確認
    if os.path.exists(video_path):
        logger.info(f"動画ファイル '{video_path}' を処理中...")

        output_video = os.path.join(output_dir, "videos/output.mp4")
        result_log = os.path.join("data", "frame_results.csv")

        processor.process_video(
            input_path=video_path,
            output_path=output_video,
            result_log=result_log,
            show_preview=True,
            apply_correction=True
        )

        logger.info("=== 処理完了 ===")
        logger.info(f"出力動画: {output_video}")
        logger.info(f"検出結果: {result_log}")

    else:
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        logger.info("現在のディレクトリにshoot5.mp4があることを確認してください。")


if __name__ == "__main__":
    main()
