# main.py

import os
from scripts.distortion import IntegratedVideoProcessor
from scripts.downloader import download_file_from_google_drive
import logging
import json

# ===== 設定セクション（ここで全てを調整可能） =====
DISTORTION_CONFIG = {
    # 歪み補正パラメータ（yolo_checker.py準拠の高精度補正）
    "k1": -0.30,        # 120°広角レンズ用の強めの逆バレル補正
    "k2": 0.03,         # 広角レンズの二次歪みを補正
    "p1": 0.0,          # 接線歪み係数1
    "p2": 0.0,          # 接線歪み係数2
    "k3": 0.01,         # 第3歪み係数
    "alpha": 0.4,       # 広角なので切り抜き重視
    "focal_scale": 0.65, # 広角効果を少し抑える
    "apply_correction": True,
}

VIDEO_CONFIG = {
    # 動画処理設定
    "show_preview": True,       # プレビュー表示
    "enable_enhanced_csv": True, # 拡張CSV機能
    "zoom_factor": 1.3,         # ズーム倍率（下位互換性のため残す）
}

MODEL_CONFIG = {
    # YOLOモデル設定
    "model_path": "models/yolo11x-pose.pt",
    "conf_threshold": 0.4,      # 検出信頼度閾値
    "phone_distance_threshold": 100, # スマホ使用判定距離
}
# ===================================================

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """メイン関数（改良版）"""
    logger.info("=== 改良版姿勢検出システム開始 ===")

    # 設定情報の表示
    logger.info("設定情報:")
    logger.info(f"  歪み補正: {'有効' if DISTORTION_CONFIG['apply_correction'] else '無効'}")
    logger.info(f"  k1={DISTORTION_CONFIG['k1']}, k2={DISTORTION_CONFIG['k2']}")
    logger.info(f"  alpha={DISTORTION_CONFIG['alpha']}, focal_scale={DISTORTION_CONFIG['focal_scale']}")
    logger.info(f"  プレビュー: {'有効' if VIDEO_CONFIG['show_preview'] else '無効'}")
    logger.info(f"  拡張CSV: {'有効' if VIDEO_CONFIG['enable_enhanced_csv'] else '無効'}")

    # Google Drive設定
    google_drive_file_id = "1QaYIFAlXRqcThZU9aLGWWQTCUXs6WJCU"

    # 動画ダウンロード
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

    # 出力ディレクトリ設定
    output_dir = "videos"
    data_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # 統合処理システムを初期化（改良版）
    logger.info("改良版統合処理システムを初期化中...")
    processor = IntegratedVideoProcessor(
        k1=DISTORTION_CONFIG["k1"],
        k2=DISTORTION_CONFIG["k2"],
        p1=DISTORTION_CONFIG["p1"],
        p2=DISTORTION_CONFIG["p2"],
        k3=DISTORTION_CONFIG["k3"],
        alpha=DISTORTION_CONFIG["alpha"],
        focal_scale=DISTORTION_CONFIG["focal_scale"],
        model_path=MODEL_CONFIG["model_path"]
    )

    # 拡張CSVロガーを有効化
    enhanced_csv_path = None
    if VIDEO_CONFIG["enable_enhanced_csv"]:
        enhanced_csv_path = os.path.join(data_dir, "enhanced_detection_log_v2.csv")

        # 既存ファイル削除（クリーンスタート）
        if os.path.exists(enhanced_csv_path):
            os.remove(enhanced_csv_path)
            logger.info(f"既存の拡張CSVファイルを削除: {enhanced_csv_path}")

        # 拡張CSVロガーを設定
        processor.set_csv_logger(enhanced_csv_path)

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
    output_video = os.path.join(output_dir, "output_advanced_posture_detection.mp4")
    result_log = os.path.join(data_dir, "frame_results_v2.csv")

    # 既存ファイル削除（クリーンスタート）
    for file_path in [output_video, result_log]:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"既存ファイルを削除: {file_path}")

    logger.info("=== 改良版処理開始 ===")
    logger.info(f"入力動画: {video_path}")
    logger.info(f"出力動画: {output_video}")
    logger.info(f"基本結果ログ: {result_log}")
    if enhanced_csv_path:
        logger.info(f"拡張結果ログ: {enhanced_csv_path}")
    logger.info("改良内容:")
    logger.info("  - 詳細な状態分類（PhoneUsageState）")
    logger.info("  - 前向き・後ろ向きの自動判定")
    logger.info("  - 高度な特徴量抽出システム")
    logger.info("  - 5係数を使用した高精度歪み補正")
    logger.info("  - 向き別の最適化されたスマホ検出")

    try:
        # 動画処理実行
        processor.process_video(
            input_path=video_path,
            output_path=output_video,
            result_log=result_log,
            show_preview=VIDEO_CONFIG["show_preview"],
            apply_correction=DISTORTION_CONFIG["apply_correction"]
        )

        logger.info("=== 改良版処理完了 ===")

        # 結果ファイルの確認と統計
        files_to_check = [
            (output_video, "出力動画"),
            (result_log, "基本CSV"),
        ]

        if enhanced_csv_path:
            files_to_check.append((enhanced_csv_path, "拡張CSV"))

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

        # 統計情報取得
        try:
            stats = processor.get_statistics()
            logger.info("=== 最終統計情報 ===")
            logger.info(f"アクティブトラック数: {stats['active_tracks']}")
            logger.info(f"総CSV記録数: {stats['total_csv_records']}")
            logger.info(f"追跡ID一覧: {stats['track_ids']}")
        except Exception as e:
            logger.warning(f"統計情報取得エラー: {e}")

        # 改良版の成果判定
        success_criteria = [
            (os.path.exists(output_video), "動画出力"),
            (os.path.exists(result_log), "基本CSV出力"),
        ]

        if enhanced_csv_path:
            success_criteria.append((
                os.path.exists(enhanced_csv_path) and os.path.getsize(enhanced_csv_path) > 1000,
                "拡張CSV出力"
            ))

        all_success = all(criteria for criteria, _ in success_criteria)

        if all_success:
            logger.info("✅ 改良版処理が正常に完了しました")
            logger.info("主な改善点の確認:")
            logger.info("  ✓ 詳細な姿勢状態分類システム")
            logger.info("  ✓ 向き判定による最適化")
            logger.info("  ✓ 高精度歪み補正")
            logger.info("  ✓ 拡張データロギング")
        else:
            logger.warning("⚠️ 一部の処理で問題が発生しました")
            for success, name in success_criteria:
                status = "✓" if success else "✗"
                logger.info(f"    {status} {name}")

        # 設定情報の保存（デバッグ・再現用）
        config_file = os.path.join(data_dir, "processing_config.json")
        config_data = {
            "distortion_config": DISTORTION_CONFIG,
            "video_config": VIDEO_CONFIG,
            "model_config": MODEL_CONFIG,
            "processing_timestamp": str(logger.handlers[0].formatter.formatTime(logger.handlers[0], logging.LogRecord("", 0, "", 0, "", (), None))),
            "input_video": video_path,
            "output_video": output_video,
            "result_log": result_log,
            "enhanced_csv": enhanced_csv_path
        }

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            logger.info(f"設定ファイル保存: {config_file}")
        except Exception as e:
            logger.warning(f"設定ファイル保存エラー: {e}")

    except KeyboardInterrupt:
        logger.info("⏹️ ユーザーによって処理が中断されました")
    except Exception as e:
        logger.error(f"❌ 処理中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # クリーンアップ
        try:
            if processor.csv_logger:
                processor.csv_logger.close()
        except:
            pass
        logger.info("リソースのクリーンアップ完了")

def validate_environment():
    """環境の検証"""
    logger.info("=== 環境検証 ===")

    required_dirs = ["models", "videos", "data"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"ディレクトリ作成: {dir_name}")
        else:
            logger.info(f"ディレクトリ確認: {dir_name}")

    # モデルファイルの確認
    model_path = MODEL_CONFIG["model_path"]
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path)
        logger.info(f"YOLOモデル確認: {model_path} ({model_size:,} bytes)")
    else:
        logger.warning(f"YOLOモデルが見つかりません: {model_path}")
        logger.info("モデルは初回実行時に自動ダウンロードされます")

def show_system_info():
    """システム情報の表示"""
    logger.info("=== システム情報 ===")

    try:
        import cv2
        logger.info(f"OpenCV バージョン: {cv2.__version__}")
    except:
        logger.warning("OpenCVが利用できません")

    try:
        import torch
        logger.info(f"PyTorch バージョン: {torch.__version__}")
        logger.info(f"CUDA利用可能: {torch.cuda.is_available()}")
    except:
        logger.warning("PyTorchが利用できません")

    try:
        from ultralytics import YOLO
        logger.info("Ultralytics YOLO: 利用可能")
    except:
        logger.warning("Ultralytics YOLOが利用できません")

def print_usage_help():
    """使用方法のヘルプ"""
    print("""
=== 改良版姿勢検出システム 使用方法 ===

【基本実行】
python main.py

【設定のカスタマイズ】
main.py の設定セクションを編集してください：

DISTORTION_CONFIG = {
    "k1": -0.30,      # 歪み補正の強さ（負の値で逆バレル補正）
    "alpha": 0.4,     # 切り抜き vs 画質のバランス
    "apply_correction": True,  # 歪み補正の有効/無効
}

VIDEO_CONFIG = {
    "show_preview": True,      # プレビュー表示
    "enable_enhanced_csv": True,  # 詳細CSV出力
}

【主な改良点】
✓ 詳細な状態分類（6つの状態）
✓ 前向き・後ろ向き自動判定
✓ 高精度5係数歪み補正
✓ 機械学習用拡張CSV出力
✓ 統計情報とデバッグ支援

【操作方法】
- プレビュー表示中は 'q' キーで終了
- 全ての結果は videos/ と data/ ディレクトリに保存されます
""")

if __name__ == "__main__":
    import sys

    # ヘルプ表示
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage_help()
        sys.exit(0)

    # システム情報表示
    show_system_info()

    # 環境検証
    validate_environment()

    # メイン処理実行
    main()