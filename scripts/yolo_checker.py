class OrderedIDTracker:
    """左から順にIDを割り振る追跡システム（YOLOv11のtrack機能と併用）"""

    def __init__(self, distance_threshold=100, max_missing_frames=30):
        self.distance_threshold = distance_threshold
        self.max_missing_frames = max_missing_frames
        self.tracked_persons = {}
        self.next_id = 1

    def get_active_tracks(self):
        """アクティブな追跡対象を取得"""
        return {tid: data for tid, data in self.tracked_persons.items()}


class IntegratedVideoProcessor:
    """統合動画処理システム（YOLOv11版）"""

    def __init__(self, k1=-0.1, k2=0.0, p1=0.0, p2=0.0, k3=0.0, alpha=0.6, focal_scale=0.9, strength=1.0, zoom_factor=0.8, model_path="yolov11n-pose.pt"):
        # 改良された歪み補正器
        self.corrector = VideoDistortionCorrector(k1, k2, p1, p2, k3, alpha, focal_scale, strength, zoom_factor)
        
        # YOLOv11版姿勢検出システム
        self.detector = AdvancedPostureDetectionSystem(model_path)

        # 拡張CSVロガー
        self.csv_logger = None

        logger.info("YOLOv11統合動画処理システムを初期化しました")

    def set_csv_logger(self, csv_path="enhanced_detection_log.csv"):
        """CSVロガーを設定"""
        try:
            self.csv_logger = EnhancedCSVLogger(csv_path)
            logger.info(f"拡張CSVロガー初期化成功: {csv_path}")
        except Exception as e:
            logger.error(f"拡張CSVロガー初期化失敗: {e}")
            self.csv_logger = None

    def process_video(self, input_path, output_path, result_log="frame_results.csv",
                     show_preview=True, apply_correction=True):
        """
        動画を処理（YOLOv11版）

        Parameters:
        input_path: 入力動画パス
        output_path: 出力動画パス
        result_log: 結果ログのCSVファイル名
        show_preview: プレビュー表示するか
        apply_correction: 歪み補正を適用するか
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"動画ファイルを開けません: {input_path}")
            return

        # 動画情報取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"動画情報: {width}x{height}, {fps:.1f}fps, {total_frames}フレーム")

        # 歪み補正マップ作成
        if apply_correction:
            logger.info("改良版歪み補正マップを作成中...")
            self.corrector.create_correction_maps(width, height)

        # 出力動画設定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 結果ログ準備
        os.makedirs(os.path.dirname(result_log), exist_ok=True)

        all_results = []
        with open(result_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "track_id", "phone_state", "grid_row", "grid_col"])

            frame_idx = 0
            start_time = time.time()

            logger.info("YOLOv11動画処理を開始...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # 歪み補正適用
                if apply_correction:
                    frame = self.corrector.apply_correction(frame)

                # YOLOv11姿勢検出処理
                try:
                    processed_frame, frame_results = self.detector.process_frame(
                        frame, frame_idx, writer, self.csv_logger
                    )
                    all_results.extend(frame_results)

                except Exception as detection_error:
                    logger.error(f"フレーム{frame_idx}処理エラー: {detection_error}")
                    processed_frame = frame

                # フレーム番号表示
                cv2.putText(
                    processed_frame,
                    f"Frame: {frame_idx}/{total_frames} (YOLOv11)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )

                # 拡張CSV記録状況表示
                if self.csv_logger and hasattr(self.csv_logger, 'log_count'):
                    cv2.putText(
                        processed_frame,
                        f"Enhanced CSV: {self.csv_logger.log_count} records",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                # プレビュー表示
                if show_preview:
                    display_title = "YOLOv11 Integrated Video Processing"
                    cv2.imshow(display_title, processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("ユーザーによって処理が中断されました")
                        break

                # 結果保存
                out.write(processed_frame)

                # 進行状況表示
                if frame_idx % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_idx / elapsed
                    progress = (frame_idx / total_frames) * 100
                    eta = (total_frames - frame_idx) / fps_current if fps_current > 0 else 0

                    csv_records = self.csv_logger.log_count if self.csv_logger else 0

                    logger.info(
                        f"進行状況: {progress:.1f}% ({frame_idx}/{total_frames}) "
                        f"処理速度: {fps_current:.1f}fps 残り時間: {eta:.1f}秒 "
                        f"検出数: {len(frame_results)} "
                        f"拡張CSV記録: {csv_records}行"
                    )

        # リソース解放
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # CSVロガーをクローズ
        if self.csv_logger:
            self.csv_logger.close()

        # 統計情報の表示
        if all_results:
            stats = self._get_statistics(all_results)
            logger.info("=== YOLOv11処理結果統計 ===")
            logger.info(f"総検出数: {stats['total_detections']}")
            logger.info(f"平均信頼度: {stats['average_confidence']:.3f}")
            logger.info(f"スマホ使用率: {stats['phone_usage_ratio']:.3f}")
            logger.info(f"状態分布: {stats['state_distribution']}")
            logger.info(f"向き分布: {stats['orientation_distribution']}")

        # 処理完了時間
        total_time = time.time() - start_time
        logger.info(f"🎉 YOLOv11動画処理完了!")
        logger.info(f"出力ファイル: {output_path}")
        logger.info(f"結果ログ: {result_log}")
        logger.info(f"処理時間: {total_time:.1f}秒")
        logger.info(f"平均処理速度: {frame_idx/total_time:.1f}fps")

    def _get_statistics(self, results: List[DetectionResult]) -> Dict:
        """検出結果の統計情報を計算"""
        if not results:
            return {}

        # 状態別の統計
        state_counts = defaultdict(int)
        orientation_counts = defaultdict(int)
        confidence_sum = 0

        for result in results:
            state_counts[result.phone_state.value] += 1
            orientation_counts[result.orientation.value] += 1
            confidence_sum += result.confidence

        # 平均信頼度
        avg_confidence = confidence_sum / len(results) if results else 0

        # スマホ使用率
        phone_usage_count = (
            state_counts['holding_near_face'] + 
            state_counts['looking_down'] + 
            state_counts['both_hands_up']
        )

        stats = {
            'total_detections': len(results),
            'average_confidence': avg_confidence,
            'state_distribution': dict(state_counts),
            'orientation_distribution': dict(orientation_counts),
            'phone_usage_ratio': phone_usage_count / len(results) if results else 0
        }

        return stats

    def process_image(self, image_path, output_dir="output", show_comparison=True):
        """
        画像処理（改良版歪み補正）

        Parameters:
        image_path: 入力画像パス
        output_dir: 出力ディレクトリ
        show_comparison: 比較表示するか
        """
        # 画像読み込み
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"画像が読み込めません: {image_path}")
            return

        h, w = image.shape[:2]
        logger.info(f"画像を読み込み: {image_path}")
        logger.info(f"画像サイズ: {w}x{h}")

        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)

        # 改良版歪み補正マップ作成
        self.corrector.create_correction_maps(w, h)

        # 歪み補正適用
        logger.info(f"改良版歪み補正を適用中...")
        corrected = self.corrector.apply_correction(image)

        # 結果保存
        output_path = os.path.join(output_dir, f'corrected_yolov11_enhanced.png')
        cv2.imwrite(output_path, corrected)
        logger.info(f"補正完了: {output_path}")

        # 比較表示
        if show_comparison:
            self._create_comparison_plot(image, corrected, output_dir)

        return corrected

    def _create_comparison_plot(self, original, corrected, output_dir):
        """比較画像を作成"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # BGR→RGB変換
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

        # 画像表示
        axes[0].imshow(original_rgb)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')

        axes[1].imshow(corrected_rgb)
        axes[1].set_title(f'Enhanced Distortion Corrected (YOLOv11)', fontsize=14)
        axes[1].axis('off')

        plt.tight_layout()
        # 比較画像保存
        comparison_path = os.path.join(output_dir, f'comparison_yolov11_enhanced.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.show()

        logger.info(f"比較画像を保存: {comparison_path}")