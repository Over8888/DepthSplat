import glob
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pytube import YouTube
from skimage import io


NETWORK_ERROR_MARKERS = (
    "urlopen error",
    "cannot assign requested address",
    "temporary failure in name resolution",
    "connection reset",
    "timed out",
    "timeout",
    "remote end closed connection",
    "connection aborted",
    "connection refused",
    "network is unreachable",
    "name or service not known",
    "bad request",
    "http error 400",
    "http error 403",
    "http error 429",
    "too many requests",
)

TERMINAL_VIDEO_ERROR_MARKERS = (
    "private video",
    "video unavailable",
    "members-only",
    "not available",
    "copyright",
    "age-restricted",
    "sign in to confirm your age",
)


class Data:
    def __init__(self, url, seqname, list_timestamps):
        self.url = url
        self.list_seqnames = [seqname]
        self.list_list_timestamps = [list_timestamps]

    def add(self, seqname, list_timestamps):
        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def __len__(self):
        return len(self.list_seqnames)


def to_ffmpeg_timestamp(timestamp_ms: int) -> str:
    hours = str(int(timestamp_ms / 3_600_000)).zfill(2)
    minutes = str(int((timestamp_ms % 3_600_000) / 60_000)).zfill(2)
    seconds = str(int((timestamp_ms % 60_000) / 1000)).zfill(2)
    millis = str(int(timestamp_ms % 1000)).zfill(3)
    return f"{hours}:{minutes}:{seconds}.{millis}"


def extract_frames_to_npz(data, seq_id, videoname, output_root):
    seqname = data.list_seqnames[seq_id]
    sequence_dir = Path(output_root) / seqname
    save_path = sequence_dir / "data.npz"

    if save_path.exists():
        return False

    if sequence_dir.exists():
        print(f"[WARN] Sequence directory already exists, skip: {sequence_dir}")
        return True

    sequence_dir.mkdir(parents=True, exist_ok=False)

    for timestamp in data.list_list_timestamps[seq_id]:
        frame_path = sequence_dir / f"{timestamp}.png"
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            to_ffmpeg_timestamp(int(timestamp)),
            "-i",
            str(videoname),
            "-vframes",
            "1",
            "-f",
            "image2",
            str(frame_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[WARN] ffmpeg failed for {seqname} @ {timestamp}: {result.stderr.strip()}")

    frame_list = sorted(sequence_dir.glob("*.png"))
    images = {}
    for frame_path in frame_list:
        image = io.imread(frame_path)
        images[frame_path.name] = image
        frame_path.unlink()

    np.savez(save_path, **images)
    return False


def wrap_process(args):
    return extract_frames_to_npz(*args)


class DataDownloader:
    def __init__(self, dataroot, mode="test"):
        print("[INFO] Loading data list ... ", end="")
        self.dataroot = Path(dataroot).resolve()
        self.mode = mode
        self.data_root = self.dataroot.parent
        self.txt_files = sorted(self.dataroot.glob("*.txt"))
        self.output_root = self.data_root / "realestate" / self.mode
        self.failure_log_path = self.data_root / f"failed_videos_{self.mode}.txt"
        self.download_dir = self.data_root
        self.download_stem = f"current_{self.mode}"

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.reset_failure_log()
        print(f"Directory {self.output_root} created." if not any(self.output_root.iterdir()) else "")

        self.list_data = []
        for txt_file in self.txt_files:
            seq_name = txt_file.stem
            with txt_file.open("r", encoding="utf-8") as seq_file:
                lines = seq_file.readlines()

            youtube_url = ""
            list_timestamps = []
            for idx, line in enumerate(lines):
                if idx == 0:
                    youtube_url = line.strip()
                else:
                    timestamp = int(line.split(" ")[0])
                    list_timestamps.append(timestamp)

            registered = False
            for item in self.list_data:
                if youtube_url == item.url:
                    item.add(seq_name, list_timestamps)
                    registered = True
                    break

            if not registered:
                self.list_data.append(Data(youtube_url, seq_name, list_timestamps))

        print(" Done! ")
        print(f"[INFO] {len(self.list_data)} movies are used in {self.mode} mode")

    def reset_failure_log(self):
        if self.failure_log_path.exists():
            self.failure_log_path.unlink()
        self.failure_log_path.touch()

    def cleanup_download_artifacts(self):
        pattern = str(self.download_dir / f"{self.download_stem}.*")
        for path in glob.glob(pattern):
            path_obj = Path(path)
            if path_obj.is_file():
                path_obj.unlink()

    def record_failure(self, data):
        with self.failure_log_path.open("a", encoding="utf-8") as failure_log:
            for seqname in data.list_seqnames:
                failure_log.write(seqname + "\n")

    def is_terminal_video_error(self, exc):
        message = str(exc).lower()
        return any(marker in message for marker in TERMINAL_VIDEO_ERROR_MARKERS)

    def should_try_ytdlp(self, exc):
        if self.is_terminal_video_error(exc):
            return False
        message = str(exc).lower()
        if any(marker in message for marker in NETWORK_ERROR_MARKERS):
            return True
        return True

    def download_with_pytube(self, url):
        yt = YouTube(url)
        stream = yt.streams.filter(res="720p", progressive=True).first()
        if stream is None:
            stream = yt.streams.filter(res="720p").first()
        if stream is None:
            stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
        if stream is None:
            raise RuntimeError("pytube could not find a usable 720p/mp4 stream")
        return Path(stream.download(str(self.download_dir), self.download_stem))

    def find_downloaded_video(self):
        candidates = sorted(self.download_dir.glob(f"{self.download_stem}.*"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"download completed but no file matched {self.download_stem}.*")

    def download_with_ytdlp(self, url):
        output_template = str(self.download_dir / f"{self.download_stem}.%(ext)s")
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
            "--merge-output-format",
            "mp4",
            "--no-playlist",
            "-o",
            output_template,
            url,
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stdout.strip():
            print(result.stdout)
        if result.stderr.strip():
            print(result.stderr)
        return self.find_downloaded_video()

    def download_video(self, url):
        self.cleanup_download_artifacts()
        try:
            videoname = self.download_with_pytube(url)
            print("[INFO] Download completed with pytube")
            return videoname
        except Exception as pytube_exc:
            print(f"[WARN] pytube download failed: {pytube_exc}")
            if not self.should_try_ytdlp(pytube_exc):
                raise RuntimeError(f"pytube failed with terminal error: {pytube_exc}") from pytube_exc

        print("[INFO] Falling back to yt-dlp")
        self.cleanup_download_artifacts()
        try:
            videoname = self.download_with_ytdlp(url)
            print("[INFO] Download completed with yt-dlp fallback")
            return videoname
        except Exception as ytdlp_exc:
            raise RuntimeError(f"yt-dlp fallback failed after pytube error: {ytdlp_exc}") from ytdlp_exc

    def run(self):
        print(f"[INFO] Start downloading {len(self.list_data)} movies")
        for data in self.list_data:
            print(f"[INFO] Downloading {data.url}")
            try:
                videoname = self.download_video(data.url)
            except Exception as exc:
                print(f"[ERROR] Download failed for {data.url}: {exc}")
                self.record_failure(data)
                continue

            if len(data) == 1:
                extract_frames_to_npz(data, 0, videoname, self.output_root)
            else:
                with Pool(processes=4) as pool:
                    pool.map(
                        wrap_process,
                        [(data, seq_id, videoname, self.output_root) for seq_id in range(len(data))],
                    )

            if videoname.is_file():
                videoname.unlink()

        return True

    def show(self):
        print("########################################")
        global_count = 0
        for data in self.list_data:
            print(f" URL : {data.url}")
            for idx in range(len(data)):
                print(f" SEQ_{idx} : {data.list_seqnames[idx]}")
                print(f" LEN_{idx} : {len(data.list_list_timestamps[idx])}")
                global_count += 1
            print("----------------------------------------")
        print(f"TOTAL : {global_count} sequnces")


def resolve_input_directory(input_arg):
    dataroot = Path(input_arg).resolve()
    if not dataroot.is_dir():
        raise ValueError(f"input directory does not exist: {dataroot}")
    mode = dataroot.name
    if mode not in {"test", "train"}:
        raise ValueError(f"input directory must end with 'test' or 'train', got: {dataroot}")
    return dataroot, mode


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: this.py <input_dir>")
        sys.exit(1)

    try:
        dataroot, mode = resolve_input_directory(sys.argv[1])
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    downloader = DataDownloader(dataroot, mode)
    downloader.show()
    ok = downloader.run()
    print("Done!" if ok else "Failed")
