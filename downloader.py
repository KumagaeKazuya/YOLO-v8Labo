import gdown
import os

def download_file_from_google_drive(file_id: str, save_dir="input_videos", file_name=None):
    """
    Google Driveからファイルをダウンロードする関数。
    :param file_id: Google DriveのファイルID
    :param save_dir: 保存先ディレクトリ
    :param file_name: 保存するファイル名。Noneの場合は元のファイル名を使用
    :return: ダウンロードしたファイルのパス
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if file_name is None:
        file_name = f"input.mp4"

    file_path = os.path.join(save_dir, file_name)

    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
    return file_path