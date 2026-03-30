import os
import json
import logging
import argparse
import glob
import re

# Fix NumExpr warning
os.environ["NUMEXPR_MAX_THREADS"] = "16"
# Fix ONNX Thread Affinity errors
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from mmagent.utils.video_processing import process_video_clip
from mmagent.face_processing import process_faces
from mmagent.audio_processing import process_voices

# Configure logger
logger = logging.getLogger(__name__)

processing_config = json.load(open("configs/processing_config.json"))
memory_config = json.load(open("configs/memory_config.json"))

def process_segment(
    video_graph,
    base64_video,
    base64_frames,
    base64_audio,
    clip_id,
    sample
):
    save_path = sample["intermediate_path"]

    process_voices(
        video_graph,
        base64_audio,
        base64_video,
        save_path=os.path.join(save_path, f"clip_{clip_id}_voices.json"),
        preprocessing=["voice"],
    )

    process_faces(
        video_graph,
        base64_frames,
        save_path=os.path.join(save_path, f"clip_{clip_id}_faces.json"),
        preprocessing=["face"],
    )

def _streaming_process_video(sample):
    """Process video segments at specified intervals with given fps.

    Args:
        video_graph (VideoGraph): Graph object to store video information
        video_path (str): Path to the video file or directory containing clips
        interval_seconds (float): Time interval between segments in seconds
        fps (float): Frames per second to extract from each segment

    Returns:
        None: Updates video_graph in place with processed segments
    """

    # Process each interval
    clips = glob.glob(sample["clip_path"] + "/*")
    for clip_path in clips:
        clip_id = int(os.path.basename(clip_path).split(".")[0].split("_")[-1])
        base64_video, base64_frames, base64_audio = process_video_clip(clip_path)

        # Process frames for this interval
        if base64_frames:
            process_segment(
                None,
                base64_video,
                base64_frames,
                base64_audio,
                clip_id,
                sample,
            )

def streaming_process_video(sample):
    print(f"DEBUG: Starting process for {sample.get('clip_path')}")
    clips = glob.glob(sample["clip_path"] + "/*")
    
    if not clips:
        print(f"WARNING: No clips found in {sample['clip_path']}")
        return

    for clip_path in clips:
        print(f"DEBUG: Processing clip: {clip_path}")
        try:
            # Safer way to get ID on Windows
            clip_name = os.path.basename(clip_path)
            clip_id = int(clip_name.split(".")[0].split("_")[-1])
            
            base64_video, base64_frames, base64_audio = process_video_clip(clip_path)
            print(f"DEBUG: Successfully extracted data for clip {clip_id}")

            if base64_frames:
                process_segment(None, base64_video, base64_frames, base64_audio, clip_id, sample)
                print(f"DEBUG: Finished segment {clip_id}")
        except Exception as e:
            print(f"ERROR on clip {clip_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/data.jsonl")
    args = parser.parse_args()

    with open(args.data_file, "r") as f:
        for line in f:
            streaming_process_video(json.loads(line))