import base64
import logging
import os
import tempfile
import math
import cv2
import numpy as np
from moviepy import VideoFileClip
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_video_info(file_path):
    """Retrieves media metadata efficiently using a context manager."""
    with VideoFileClip(file_path) as video:
        return {
            "path": file_path,
            "name": os.path.basename(file_path),
            "format": os.path.splitext(file_path)[1][1:].lower(),
            "fps": video.fps,
            "duration": video.duration,
            "frames": int(video.fps * video.duration),
            "width": video.size[0],
            "height": video.size[1]
        }

def extract_frames(video, sample_fps=10):
    """
    Extracts frames using iter_frames for significantly better performance 
    compared to repeated get_frame calls.
    """
    frames = []
    # iter_frames handles sampling logic internally
    for frame in video.iter_frames(fps=sample_fps, dtype="uint8"):
        # Convert RGB (moviepy) to BGR (opencv)
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", bgr_frame)
        frames.append(base64.b64encode(buffer).decode("utf-8"))
    return frames

def process_video_clip(video_path, fps=5, audio_fps=16000):
    """Processes video and audio into base64 strings in a single pass."""
    try:
        results = {}
        with VideoFileClip(video_path) as video:
            # 1. Read raw video bytes
            with open(video_path, "rb") as f:
                results["video"] = base64.b64encode(f.read()).decode("utf-8")

            # 2. Extract frames (Efficiently)
            results["frames"] = extract_frames(video, sample_fps=fps)

            # 3. Handle Audio
            results["audio"] = None
            if video.audio:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    audio_temp_path = tmp.name
                
                try:
                    video.audio.write_audiofile(
                        audio_temp_path, codec="pcm_s16le", 
                        fps=audio_fps, logger=None
                    )
                    with open(audio_temp_path, "rb") as f:
                        results["audio"] = base64.b64encode(f.read()).decode("utf-8")
                finally:
                    if os.path.exists(audio_temp_path):
                        os.remove(audio_temp_path)

        return results["video"], results["frames"], results["audio"]
    except Exception as e:
        logger.error(f"Failed to process clip {video_path}: {e}")
        raise

def has_media_streams(file_path):
    """Checks for video and audio streams using a single ffprobe call."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
        "-of", "csv=p=0", file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    streams = result.stdout.strip().split('\n')
    return "video" in streams and "audio" in streams

def verify_video_processing(video_path, output_dir, interval, strict=False):
    """Verifies splitting results with improved stream and static checks."""
    if not os.path.exists(video_path):
        return False

    info = get_video_info(video_path)
    expected_count = math.ceil(info["duration"] / interval)
    
    if not os.path.exists(output_dir):
        return False
        
    actual_files = [f for f in os.listdir(output_dir) if f.lower().endswith(('.mp4', '.webm', '.mov'))]
    if len(actual_files) != expected_count:
        logger.error(f"Count mismatch: expected {expected_count}, found {len(actual_files)}")
        return False

    if strict:
        for filename in actual_files:
            full_path = os.path.join(output_dir, filename)
            if not has_media_streams(full_path):
                logger.error(f"Missing streams in {filename}")
                return False
    return True