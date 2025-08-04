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
    data_dir = "data"

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # 統合処理システムを初期化（逆バレル補正版）
    processor = IntegratedVideoProcessor(
        k1=-0.1,           # 歪み係数（負の値で樽型歪みを補正）
        strength=1.0,      # 補正強度
        zoom_factor=1.3,   # ズーム倍率（1.0未満で引き、1.0超で寄り）
        model_path="models/yolov8m-pose.pt"
    )

    # 拡張CSVロガーを有効化（機械学習用詳細データ）
    enhanced_csv_path = os.path.join(data_dir, "enhanced_detection_log.csv")
    processor.set_csv_logger(enhanced_csv_path)
    logger.info(f"拡張CSVロガーを有効化: {enhanced_csv_path}")

    # 動画ファイルの存在確認
    if os.path.exists(video_path):
        logger.info(f"動画ファイル '{video_path}' を処理中...")

        # 出力パスの設定（二重ディレクトリを修正）
        output_video = os.path.join(output_dir, "output.mp4")
        result_log = os.path.join(data_dir, "frame_results.csv")

        logger.info("=== 処理開始 ===")
        logger.info(f"入力動画: {video_path}")
        logger.info(f"出力動画: {output_video}")
        logger.info(f"基本結果ログ: {result_log}")
        logger.info(f"拡張結果ログ: {enhanced_csv_path}")
        logger.info("両方のCSVファイルが生成されます:")
        logger.info("  - frame_results.csv: 基本的な検出結果（5列）")
        logger.info("  - enhanced_detection_log.csv: 機械学習用詳細データ（69列）")

        processor.process_video(
            input_path=video_path,
            output_path=output_video,
            result_log=result_log,
            show_preview=True,
            apply_correction=True
        )

        logger.info("=== 処理完了 ===")
        logger.info(f"出力動画: {output_video}")
        logger.info(f"基本検出結果: {result_log}")
        logger.info(f"拡張検出結果: {enhanced_csv_path}")

        # ファイルサイズ情報も表示
        if os.path.exists(result_log):
            size_basic = os.path.getsize(result_log) / 1024  # KB
            logger.info(f"基本CSVファイルサイズ: {size_basic:.1f} KB")

        if os.path.exists(enhanced_csv_path):
            size_enhanced = os.path.getsize(enhanced_csv_path) / 1024  # KB
            logger.info(f"拡張CSVファイルサイズ: {size_enhanced:.1f} KB")

        logger.info("生成されたファイル:")
        logger.info(f"  📹 動画: {output_video}")
        logger.info(f"  📊 基本CSV: {result_log}")
        logger.info(f"  📈 拡張CSV: {enhanced_csv_path}")

    else:
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        logger.info("現在のディレクトリにshoot5.mp4があることを確認してください。")

if __name__ == "__main__":
    main()