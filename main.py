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
    logger.info("CSV修正開始")
    # 設定
    google_drive_file_id = "1QaYIFAlXRqcThZU9aLGWWQTCUXs6WJCU"

    try:
        video_path = download_file_from_google_drive(google_drive_file_id)
        logger.info(f"動画ダウンロード成功: {video_path}")
    except Exception as e:
        logger.error(f"動画ダウンロード失敗: {e}")
        # フォールバック: ローカルファイルを使用
        video_path = "input.mp4"
        if not os.path.exists(video_path):
            logger.error(f"ローカル動画ファイルも見つかりません: {video_path}")
            return
        logger.info(f"ローカル動画ファイルを使用: {video_path}")
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
        model_path="models/yolo11m-pose.pt"
    )

    # 🔧 拡張CSVロガーを有効化（修正版）
    enhanced_csv_path = os.path.join(data_dir, "enhanced_detection_log.csv")

    # 既存のファイルを削除（クリーンスタート）
    if os.path.exists(enhanced_csv_path):
        os.remove(enhanced_csv_path)
        logger.info(f"既存の拡張CSVファイルを削除: {enhanced_csv_path}")

    # 拡張CSVロガーを設定
    processor.set_csv_logger(enhanced_csv_path)

    # CSVロガーの初期化確認
    if processor.csv_logger is not None:
        logger.info(f"拡張CSVロガー有効化成功: {enhanced_csv_path}")
        logger.info(f"ヘッダー列数: {len(processor.csv_logger.headers)}")
    else:
        logger.error("拡張CSVロガー有効化失敗")
        return

    # 動画ファイルの存在確認
    if not os.path.exists(video_path):
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        return
    logger.info(f"動画ファイル確認済み: {video_path}")

    # 出力パスの設定
    output_video = os.path.join(output_dir, "output_fixed.mp4")
    result_log = os.path.join(data_dir, "frame_results.csv")

    # 既存のファイルを削除（クリーンスタート）
    for file_path in [output_video, result_log]:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"既存ファイルを削除: {file_path}")

    logger.info("=== 処理開始（修正版） ===")
    logger.info(f"入力動画: {video_path}")
    logger.info(f"出力動画: {output_video}")
    logger.info(f"基本結果ログ: {result_log}")
    logger.info(f"拡張結果ログ: {enhanced_csv_path}")
    logger.info("修正内容:")
    logger.info("  - EnhancedCSVLoggerのエラーハンドリング強化")
    logger.info("  - データ長チェック機能追加")
    logger.info("  - リアルタイム記録カウンター追加")
    logger.info("  - 即座にフラッシュしてディスク書き込み")

    try:
        # 動画処理実行
        processor.process_video(
            input_path=video_path,
            output_path=output_video,
            result_log=result_log,
            show_preview=True,   # プレビュー表示
            apply_correction=True # 歪み補正適用
        )

        logger.info("=== 処理完了 ===")

        # 結果ファイルの確認
        files_to_check = [
            (output_video, "出力動画"),
            (result_log, "基本CSV"),
            (enhanced_csv_path, "拡張CSV")
        ]

        for file_path, description in files_to_check:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                logger.info(f"{description}: {file_path} ({file_size:,} bytes)")

                # CSVファイルの行数チェック
                if file_path.endswith('.csv'):
                    try:
                        with open(file_path, 'r') as f:
                            line_count = sum(1 for _ in f) - 1  # ヘッダーを除く
                        logger.info(f"データ行数: {line_count}")

                        if line_count == 0:
                            logger.warning(f"{description}にデータが記録されていません")
                        else:
                            logger.info(f"{description}に{line_count}行のデータが記録されました")

                    except Exception as e:
                        logger.error(f"{description}の行数確認エラー: {e}")
            else:
                logger.error(f"{description}ファイルが生成されていません: {file_path}")

        # 最終確認メッセージ
        if (os.path.exists(enhanced_csv_path) and
            os.path.getsize(enhanced_csv_path) > 1000):  # 1KB以上なら成功
            logger.info("拡張CSVシステム修正成功: データが正常に記録されました。")
        else:
            logger.warning("拡張CSVの記録に問題がある可能性があります。")

    except Exception as e:
        logger.error(f"処理中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()