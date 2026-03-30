import os
import json
import logging
import argparse
import glob
import pickle

from mmagent.videograph import VideoGraph
from mmagent.utils.video_processing import process_video_clip
from mmagent.face_processing import process_faces
from mmagent.audio_processing import process_voices
from mmagent.memory_processing import process_memories, generate_full_memories

logger = logging.getLogger(__name__)
processing_config = json.load(open("configs/processing_config.json"))
memory_config = json.load(open("configs/memory_config.json"))

preprocessing = []

def process_segment(
    video_graph,
    base64_video,
    base64_frames,
    base64_audio,
    clip_id,
    sample,
    clip_path
):
    save_path = sample["intermediate_path"]
    
    print(f"--- STARTING VOICE INFERENCE FOR CLIP {clip_id} ---")
    id2voices = process_voices(
        video_graph,
        base64_audio,
        base64_video,
        save_path=os.path.join(save_path, f"clip_{clip_id}_voices.json"),
        preprocessing=[],
    )

    print(f"--- STARTING FACE INFERENCE FOR CLIP {clip_id} ---")
    id2faces = process_faces(
        video_graph,
        base64_frames,
        save_path=os.path.join(save_path, f"clip_{clip_id}_faces.json"),
        preprocessing=[],
    )

    print(f"--- STARTING QWEN3 INFERENCE FOR CLIP {clip_id} ---")
    episodic_memories, semantic_memories = generate_full_memories(
        base64_video,
        base64_frames,
        id2faces,
        id2voices,
    )

    process_memories(video_graph, episodic_memories, clip_id, type="episodic")
    process_memories(video_graph, semantic_memories, clip_id, type="semantic")

def streaming_process_video(video_graph, sample):
    clips = sorted(glob.glob(sample["clip_path"] + "/*")) # Sorted to keep timeline order
    save_path = sample["intermediate_path"]
    
    for clip_path in clips:
        try:
            # 1. Extract clip_id
            clip_id = int(clip_path.replace("\\", "/").split("/")[-1].split(".")[0].split("_")[-1])
            
            voices_file = os.path.join(save_path, f"clip_{clip_id}_voices.json")
            faces_file = os.path.join(save_path, f"clip_{clip_id}_faces.json")
            
            # 2. Check existence & corruption
            if not os.path.exists(voices_file) or not os.path.exists(faces_file):
                print(f"--- SKIPPING CLIP {clip_id}: Required feature files not found ---")
                continue

            with open(voices_file, 'r') as f: json.load(f)
            with open(faces_file, 'r') as f: json.load(f)

            # 3. Process the clip
            print(f"DEBUG: Processing clip {clip_id} from {clip_path}")
            base64_video, base64_frames, base64_audio = process_video_clip(clip_path)

            if base64_frames:
                # If Qwen3 fails here, the 'except' block below catches it
                process_segment(
                    video_graph,
                    base64_video,
                    base64_frames,
                    base64_audio,
                    clip_id,
                    sample,
                    clip_path
                )
                print(f"Successfully integrated clip {clip_id}")

        except Exception as e:
            # This is the "magic" part: it catches the failure but stays in the loop
            print(f"WARNING: Clip {clip_id} failed. Skipping. Error: {e}")
            continue 
    
    # 4. Finalize the graph even if some clips were skipped
    print(f"--- Finalizing Graph for {sample['video_id']} ---")
    video_graph.refresh_equivalences()
    
    with open(sample["mem_path"], "wb") as f:
        pickle.dump(video_graph, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/data.jsonl")
    args = parser.parse_args()
    video_inputs = []
    
    with open(args.data_file, "r") as f:
        for line in f:
            sample = json.loads(line)
            if not os.path.exists(sample["mem_path"]):
                video_graph = VideoGraph(**memory_config)
                streaming_process_video(video_graph, sample)