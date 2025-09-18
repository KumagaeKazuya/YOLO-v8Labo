import os
import glob
import time
import json
from scripts.distortion import IntegratedVideoProcessor
import logging

# ===== 設定セクション（ここで全てを調整可能） =====
DISTORTION_CONFIG = {
    # 歪み補正パラメータ（yolo_checker.py準拠の高精度補正）
    "k1": -0.20,        # 120°広角レンズ用の強めの逆バレル補正
    "k2": -0.001,       # 広角レンズの二次歪みを補正
    "p1": 0.0,          # 接線歪み係数1
    "p2": 0.0,          # 接線歪み係数2
    "k3": 0.012,        # 第3歪み係数
    "alpha": 0.8,       # 広角なので切り抜き重視
    "focal_scale": 0.85, # 広角効果を少し抑える
    "apply_correction": True,
    "use_opencv_method": True,
    "zoom_factor": 1.0,
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

# スマートファイル管理設定
SMART_FILE_MANAGEMENT_CONFIG = {
    "enabled": True,                 # スマートファイル管理の有効/無効
    "force_process": False,          # 強制処理フラグ
    "quality_threshold": 1.0,        # 品質劣化防止閾値（1.0 = 同等時間以上で処理）
    "backup_existing": False,        # 既存ファイルのバックアップ
    "detailed_logging": True,        # 詳細ログ出力
}
# ===================================================

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def find_video_files(video_dir="videos"):
    """videosフォルダ内の動画ファイルを検索"""
    video_extensions = [
        "*.mp4", "*.avi", "*.mov", "*.mkv", "*.flv", "*.wmv",
        "*.MP4", "*.AVI", "*.MOV", "*.MKV", "*.FLV", "*.WMV"
    ]

    video_files = []
    for ext in video_extensions:
        pattern = os.path.join(video_dir, ext)
        video_files.extend(glob.glob(pattern))

    # ファイル名でソート
    video_files.sort()
    return video_files

def estimate_processing_time(video_path):
    """動画の処理時間を推定"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # 基本推定式（実際の環境に応じて調整してください）
        base_time_per_frame = 0.1  # 1フレームあたりの基本処理時間
        resolution_factor = (width * height) / (1920 * 1080)  # 解像度係数

        estimated_time = total_frames * base_time_per_frame * resolution_factor
        return estimated_time

    except Exception as e:
        logger.warning(f"処理時間推定エラー: {e}")
        return 0

def select_video_file():
    """動画ファイルを手動選択"""
    video_dir = "videos"

    # videosディレクトリが存在しない場合は作成
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
        logger.info(f"videosディレクトリを作成: {video_dir}")

    # 動画ファイルを検索
    video_files = find_video_files(video_dir)

    if not video_files:
        logger.error(f"{video_dir}フォルダに動画ファイルが見つかりません")
        logger.error("対応形式: .mp4, .avi, .mov, .mkv, .flv, .wmv")
        logger.error(f"動画ファイルを{video_dir}フォルダに配置してから再実行してください。")
        return None

    # 動画ファイル一覧を表示
    logger.info(f"\n{video_dir}フォルダ内の動画ファイル:")
    for i, video_file in enumerate(video_files, 1):
        file_size = os.path.getsize(video_file)
        file_name = os.path.basename(video_file)
        # ファイルサイズを見やすい単位で表示
        if file_size > 1024 * 1024:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size} bytes"

        # 推定処理時間も表示
        estimated_time = estimate_processing_time(video_file)
        time_str = f"約{estimated_time:.1f}秒" if estimated_time > 0 else "不明"

        logger.info(f"  {i}. {file_name} ({size_str}, 推定処理時間: {time_str})")

    # ユーザーのファイル選択
    while True:
        try:
            print(f"\n処理する動画を選択してください (1-{len(video_files)}):")
            choice = input("選択番号を入力: ").strip()

            if choice.isdigit():
                choice_num = int(choice)

                if 1 <= choice_num <= len(video_files):
                    selected_video = video_files[choice_num - 1]
                    logger.info(f"✅ 選択された動画: {selected_video}")
                    return selected_video
                else:
                    print(f"❌ 1から{len(video_files)}の数字を入力してください")
            else:
                print("❌ 数字を入力してください")

        except KeyboardInterrupt:
            logger.info("選択がキャンセルされました")
            return None
        except Exception as e:
            logger.error(f"選択エラー: {e}")

def process_with_smart_management(processor, video_path):
    """スマートファイル管理による動画処理（品質劣化防止機能強化）"""
    logger.info("=== スマートファイル管理による処理開始 ===")
    logger.info("主な機能:")
    logger.info("  ✓ 動画別ファイル管理（ハッシュベース）")
    logger.info("  ✓ 推論時間による自動上書き制御")
    logger.info("  ✓ 処理履歴の永続化")
    logger.info("  ✓ 重複処理の自動検出・防止")
    logger.info("  ✓ 品質劣化防止機能（推定時間短縮時の保護）")
    logger.info(f"  ✓ 品質閾値: {SMART_FILE_MANAGEMENT_CONFIG['quality_threshold']:.1f}倍")

    # スマートファイル管理のインスタンスに品質閾値を設定
    processor.file_manager.quality_threshold = SMART_FILE_MANAGEMENT_CONFIG["quality_threshold"]

    # スマートファイル管理による動画処理実行
    success = processor.process_video_with_smart_management(
        input_path=video_path,
        show_preview=VIDEO_CONFIG["show_preview"],
        apply_correction=DISTORTION_CONFIG["apply_correction"],
        force_process=SMART_FILE_MANAGEMENT_CONFIG["force_process"]
    )

    # 生成されたファイルパスを取得（スマートファイル管理から）
    file_paths = {}
    if hasattr(processor, 'file_manager'):
        try:
            estimated_time = processor.estimate_processing_time(video_path)
            _, _, file_paths = processor.file_manager.should_process_video(
                video_path, estimated_time, False
            )
        except Exception as e:
            logger.warning(f"ファイルパス取得エラー: {e}")

    return success, file_paths

# process_with_traditional_method 関数を追加
def process_with_traditional_method(processor, video_path):
    """従来方式での動画処理"""
    logger.info("=== 従来方式による処理開始 ===")

    # 出力ファイル名生成（従来方式）
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = "videos"
    data_dir = "data"

    output_video = os.path.join(output_dir, f"output_{video_basename}_advanced_posture_detection.mp4")
    result_log = os.path.join(data_dir, f"frame_results_{video_basename}.csv")
    enhanced_csv_path = os.path.join(data_dir, f"enhanced_detection_log_{video_basename}.csv")

    # 既存ファイルの削除（従来方式）
    for file_path in [output_video, result_log, enhanced_csv_path]:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"既存ファイルを削除: {os.path.basename(file_path)}")

    # 拡張CSVロガーを設定
    if VIDEO_CONFIG["enable_enhanced_csv"]:
        processor.set_csv_logger(enhanced_csv_path, overwrite_existing=True)

    # 動画処理実行
    success = processor.process_video(
        input_path=video_path,
        output_path=output_video,
        result_log=result_log,
        show_preview=VIDEO_CONFIG["show_preview"],
        apply_correction=DISTORTION_CONFIG["apply_correction"]
    )

    return success, {
        "output_video": output_video,
        "result_log": result_log,
        "enhanced_csv": enhanced_csv_path
    }

def display_processing_results(success, file_paths, total_processing_time, processor):
    """処理結果の表示"""
    if success:
        logger.info("=== 処理完了 ===")
        logger.info(f"総処理時間: {total_processing_time:.1f}秒")

        # ファイル情報の表示
        logger.info("=== 生成ファイル情報 ===")
        for key, path in file_paths.items():
            if key not in ["video_hash", "video_basename"] and os.path.exists(path):
                file_size = os.path.getsize(path)
                size_mb = file_size / (1024 * 1024)
                logger.info(f"{key}: {os.path.basename(path)} ({size_mb:.1f}MB)")

        # 統計情報を取得
        try:
            stats = processor.get_statistics()
            logger.info("=== 最終統計情報 ===")
            logger.info(f"アクティブトラック数: {stats['active_tracks']}")
            logger.info(f"総CSV記録数: {stats['total_csv_records']}")
            logger.info(f"追跡ID一覧: {stats['track_ids']}")
        except Exception as e:
            logger.warning(f"統計情報取得エラー: {e}")

        # 処理履歴の確認（スマートファイル管理の場合）
        if SMART_FILE_MANAGEMENT_CONFIG["enabled"]:
            try:
                history_file = "data/video_processing_history.json"
                if os.path.exists(history_file):
                    with open(history_file, 'r', encoding='utf-8') as f:
                        history = json.load(f)

                    logger.info("=== 処理履歴サマリー ===")
                    logger.info(f"記録済み動画数: {len(history)}動画")

                    for video_hash, record in history.items():
                        logger.info(f"  {record['video_basename']}: {record['execution_count']}回実行, "
                                f"平均{record['fps_average']:.1f}fps")

            except Exception as e:
                logger.warning(f"処理履歴確認エラー: {e}")

        logger.info("✅ 処理が正常に完了しました")

        if SMART_FILE_MANAGEMENT_CONFIG["enabled"]:
            logger.info("主な改善点の確認:")
            logger.info("  ✓ 動画別の完全分離ファイル管理")
            logger.info("  ✓ 推論時間による品質保護")
            logger.info("  ✓ 自動重複防止システム")
            logger.info("  ✓ 処理履歴の永続化")

    else:
        logger.warning("⚠️ 処理が完了しませんでした（スキップまたはエラー）")

def save_processing_config(video_path, total_processing_time, smart_management_used):
    """処理設定の保存"""
    data_dir = "data"
    config_file = os.path.join(data_dir, "processing_config.json")

    config_data = {
        "distortion_config": DISTORTION_CONFIG,
        "video_config": VIDEO_CONFIG,
        "model_config": MODEL_CONFIG,
        "smart_file_management_config": SMART_FILE_MANAGEMENT_CONFIG,
        "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_video": video_path,
        "total_processing_time": total_processing_time,
        "smart_file_management_used": smart_management_used
    }

    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        logger.info(f"設定ファイル保存: {config_file}")
    except Exception as e:
        logger.warning(f"設定ファイル保存エラー: {e}")


def main():
    """メイン関数（スマートファイル管理対応版）"""
    logger.info("=== 改良版姿勢検出システム（スマートファイル管理版）開始 ===")

    # 設定情報の表示
    logger.info("設定情報:")
    logger.info(f"  歪み補正: {'有効' if DISTORTION_CONFIG['apply_correction'] else '無効'}")
    logger.info(f"  k1={DISTORTION_CONFIG['k1']}, k2={DISTORTION_CONFIG['k2']}")
    logger.info(f"  alpha={DISTORTION_CONFIG['alpha']}, focal_scale={DISTORTION_CONFIG['focal_scale']}")
    logger.info(f"  プレビュー: {'有効' if VIDEO_CONFIG['show_preview'] else '無効'}")
    logger.info(f"  拡張CSV: {'有効' if VIDEO_CONFIG['enable_enhanced_csv'] else '無効'}")
    logger.info(f"  スマートファイル管理: {'有効' if SMART_FILE_MANAGEMENT_CONFIG['enabled'] else '無効'}")

    # 動画ファイル選択
    logger.info("=== 動画ファイル選択 ===")
    video_path = select_video_file()

    if not video_path:
        logger.error("動画ファイルが選択されませんでした。処理を終了します。")
        return

    if not os.path.exists(video_path):
        logger.error(f"選択された動画ファイルが存在しません: {video_path}")
        return

    # 出力ディレクトリ設定
    output_dir = "videos"
    data_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # 統合処理システムを初期化（スマートファイル管理対応）
    logger.info("スマートファイル管理対応統合処理システムを初期化中...")
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

    logger.info("=== スマートファイル管理による処理開始 ===")
    logger.info("主な機能:")
    logger.info("  ✓ 動画別ファイル管理（ハッシュベース）")
    logger.info("  ✓ 推論時間による自動上書き制御")
    logger.info("  ✓ 処理履歴の永続化")
    logger.info("  ✓ 重複処理の自動検出・防止")
    logger.info("  ✓ 品質劣化防止機能")

    try:
        # 処理開始時間記録
        processing_start_time = time.time()

        # 処理方式の選択と実行
        if SMART_FILE_MANAGEMENT_CONFIG["enabled"]:
            success, file_paths = process_with_smart_management(processor, video_path)
        else:
            success, file_paths = process_with_traditional_method(processor, video_path)

        # 処理時間計算
        total_processing_time = time.time() - processing_start_time

        # 処理結果の表示
        display_processing_results(success, file_paths, total_processing_time, processor)

        # 設定情報の保存（デバッグ・再現用）
        save_processing_config(video_path, total_processing_time, SMART_FILE_MANAGEMENT_CONFIG["enabled"])

    except KeyboardInterrupt:
        logger.info("⏹️ ユーザーによって処理が中断されました")
    except Exception as e:
        logger.error(f"❌ 処理中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # クリーンアップ
        try:
            if hasattr(processor, 'csv_logger') and processor.csv_logger:
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

    # videosフォルダの動画ファイル確認
    video_files = find_video_files("videos")
    if video_files:
        logger.info(f"動画ファイル発見: {len(video_files)}個")
        for i, video_file in enumerate(video_files, 1):
            file_name = os.path.basename(video_file)
            file_size = os.path.getsize(video_file)
            if file_size > 1024 * 1024:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{file_size / 1024:.1f} KB"
            logger.info(f"  {i}. {file_name} ({size_str})")
    else:
        logger.warning("動画ファイルが見つかりません")
        logger.info("videosフォルダに以下の形式の動画ファイルを配置してください:")
        logger.info("  対応形式: .mp4, .avi, .mov, .mkv, .flv, .wmv")

    # モデルファイルの確認
    model_path = MODEL_CONFIG["model_path"]
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path)
        logger.info(f"YOLOモデル確認: {model_path} ({model_size:,} bytes)")
    else:
        logger.warning(f"YOLOモデルが見つかりません: {model_path}")
        logger.info("モデルは初回実行時に自動ダウンロードされます")

    # 処理履歴ファイルの確認（スマートファイル管理有効時のみ）
    if SMART_FILE_MANAGEMENT_CONFIG["enabled"]:
        history_file = "data/video_processing_history.json"
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                logger.info(f"処理履歴ファイル確認: {len(history)}件の記録")
            except Exception as e:
                logger.warning(f"処理履歴ファイル読み込みエラー: {e}")
        else:
            logger.info("処理履歴ファイルは初回実行時に作成されます")

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

    try:
        import hashlib
        import json
        logger.info("スマートファイル管理: 利用可能")
    except:
        logger.warning("スマートファイル管理が利用できません")

def print_usage_help():
    """使用方法のヘルプ"""
    print(f"""
=== 改良版姿勢検出システム（統合版） 使用方法 ===

【基本実行】
python main.py

【処理方式の選択】
スマートファイル管理: {'有効' if SMART_FILE_MANAGEMENT_CONFIG['enabled'] else '無効'}

【スマートファイル管理の特徴】（有効時）
✓ 動画別ファイル管理：異なる動画は必ず別ファイルに保存
✓ 推論時間制御：処理時間が向上する場合のみ上書き
✓ 重複防止：同じ動画・同じ設定での無駄な再処理を防止
✓ 履歴管理：処理履歴をJSONファイルで永続化

【従来方式の特徴】（無効時）
✓ シンプルな処理フロー
✓ 毎回クリーンスタート
✓ 固定ファイル名での出力

【ファイル構成例】
スマートファイル管理有効時:
videos/
├── output_sample_video_a1b2c3d4_advanced_posture_detection.mp4
├── output_test_data_x1y2z3w4_advanced_posture_detection.mp4

data/
├── enhanced_detection_log_sample_video_a1b2c3d4.csv
├── enhanced_detection_log_test_data_x1y2z3w4.csv
├── frame_results_sample_video_a1b2c3d4.csv
├── frame_results_test_data_x1y2z3w4.csv
└── video_processing_history.json

従来方式:
videos/
├── output_sample_video_advanced_posture_detection.mp4
├── output_test_data_advanced_posture_detection.mp4

data/
├── enhanced_detection_log_sample_video.csv
├── enhanced_detection_log_test_data.csv
├── frame_results_sample_video.csv
└── frame_results_test_data.csv

【設定のカスタマイズ】
main.py の設定セクションを編集してください：

# スマートファイル管理の有効/無効
SMART_FILE_MANAGEMENT_CONFIG = {{
    "enabled": True,  # True: スマート管理, False: 従来方式
    "force_process": False,  # 強制処理
}}

DISTORTION_CONFIG = {{
    "k1": -0.20,      # 歪み補正の強さ
    "alpha": 0.8,     # 切り抜き vs 画質のバランス
    "apply_correction": True,  # 歪み補正の有効/無効
}}

VIDEO_CONFIG = {{
    "show_preview": True,      # プレビュー表示
    "enable_enhanced_csv": True,  # 詳細CSV出力
}}

【操作方法】
- プレビュー表示中は 'q' キーで終了
- 再処理確認で 'y' を入力すると強制実行
- 全ての結果は設定に応じて自動保存されます
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