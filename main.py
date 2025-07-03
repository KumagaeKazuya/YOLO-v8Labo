# 実行を行うファイル

from comp_pose import DrowsinessDetectionSystem

def main():
    rtsp_url = "rtsp://6199:4003@192.168.100.183/live"
    model_path = "yolov8m-pose.pt"
    detector = DrowsinessDetectionSystem(rtsp_url, model_path)
    video_path = "627lab1.mp4"  # 動画ファイルのパス
    save_path = "output.mp4"
    log_path = "frame_results.csv"

    detector = DrowsinessDetectionSystem("", model_path)
    detector.run_on_video(video_path, save_path, log_path)


if __name__ == "__main__":
    main()