import os
import subprocess
import math

def get_video_duration(input_video):
    """Uses ffprobe to get the exact duration of the video."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", input_video
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return float(result.stdout.strip())

def split_video(video_name, input_path, output_dir, segment_length=30):
    # Paths (matching your bash script structure)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get duration and calculate segments
    duration = get_video_duration(input_path)
    segments = math.ceil(duration / segment_length)
    
    print(f"Splitting {video_name} ({duration:.2f}s) into {segments} segments...")

    for i in range(segments):
        start_time = i * segment_length
        output_file = os.path.join(output_dir, f"{i}.mp4")
        
        # ffmpeg command
        # Note: we put -ss BEFORE -i for faster seeking
        cmd = [
            "ffmpeg", "-y", "-ss", str(start_time), "-i", input_path,
            "-t", str(segment_length), "-c", "copy", output_file
        ]
        
        print(f"Generating segment {i} (starting at {start_time}s)...")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print("Done!")

if __name__ == "__main__":
    # Default 30sec
    split_video(video_name="test", input_path="data/test/raw/test.mp4", output_dir="data/test/processed/clips")