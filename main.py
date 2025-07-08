# 実行を行うファイル

from comp_pose import DrowsinessDetectionSystem
from downloader import download_file_from_google_drive

def main():
    rtsp_url = "rtsp://6199:4003@192.168.100.183/live"
    model_path = "models/yolov8m-pose.pt"
    detector = DrowsinessDetectionSystem(rtsp_url, model_path)
    google_drive_file_id = "1QaYIFAlXRqcThZU9aLGWWQTCUXs6WJCU"
    video_path = download_file_from_google_drive(google_drive_file_id)
    save_path = "output.mp4"
    log_path = "data/frame_results.csv"

    detector = DrowsinessDetectionSystem("", model_path)
    detector.run_on_video(video_path, save_path, log_path)

if __name__ == "__main__":
    main()