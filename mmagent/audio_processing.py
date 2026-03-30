import base64
import struct
import json
import os
import logging
import torch
from io import BytesIO
from speakerlab.process.processor import FBank
from speakerlab.utils.builder import dynamic_import

import soundfile as sf
import librosa

from pydub import AudioSegment
from mmagent.utils.prompts import prompt_audio_segmentation
from mmagent.utils.chat_api import generate_messages, get_response
from mmagent.utils.general import validate_and_fix_json, normalize_embedding
from mmagent.utils.video_processing import process_video_clip
import io

processing_config = json.load(open("configs/processing_config.json"))

MAX_RETRIES = processing_config["max_retries"]

pretrained_state = torch.load("models/pretrained_eres2netv2.ckpt", map_location='cuda' if torch.cuda.is_available() else 'cpu')
model = {
    'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}
embedding_model = dynamic_import(model['obj'])(**model['args'])
embedding_model.load_state_dict(pretrained_state)
embedding_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
embedding_model.eval()
feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

def get_embedding(wav_file_obj):
    """
    Alternative implementation using SoundFile and Librosa to bypass 
    the broken TorchAudio FFmpeg/TorchCodec backend.
    """
    def load_wav_alternative(wav_io, obj_fs=16000):
        # 1. Load using soundfile (bypasses torchaudio.load)
        # wav_io can be a file path or a BytesIO object
        data, fs = sf.read(wav_io, dtype='float32')
        
        # 2. Resample using librosa (bypasses torchaudio.sox_effects)
        if fs != obj_fs:
            data = librosa.resample(data, orig_sr=fs, target_sr=obj_fs)
        
        # 3. Ensure format is [Channels, Time] for the model
        if data.ndim == 1:
            # Mono: [Time] -> [1, Time]
            wav = torch.from_numpy(data).unsqueeze(0)
        else:
            # Stereo: [Time, Channels] -> [1, Time] (take first channel)
            wav = torch.from_numpy(data[:, 0]).unsqueeze(0)
            
        return wav

    def compute_embedding(wav_io):
        wav = load_wav_alternative(wav_io)
        # Ensure the feature extractor receives the tensor correctly
        feat = feature_extractor(wav).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        with torch.no_grad():
            # Generate the speaker embedding
            embedding = embedding_model(feat).detach().squeeze(0).cpu().numpy()
        return embedding

    return compute_embedding(wav_file_obj)

@torch.no_grad()
def generate(wav):
    wav = base64.b64decode(wav)
    wav_file = BytesIO(wav)
    emb = get_embedding(wav_file)
    return emb

@torch.no_grad()
def get_audio_embeddings(audio_segments):
    res = []
    for wav in audio_segments:
        completion = generate(wav.decode("utf-8"))
        bytes_data = struct.pack('f' * len(completion), *completion)
        res.append(bytes_data)

    return res


# Configure logging
logger = logging.getLogger(__name__)


def process_voices(video_graph, base64_audio, base64_video, save_path, preprocessing=[]):
    def get_audio_segments(base64_audio, dialogs):
        # Decode base64 audio into bytes
        audio_data = base64.b64decode(base64_audio)
        
        # Create BytesIO object to hold audio data
        audio_io = io.BytesIO(audio_data)
        audio = AudioSegment.from_wav(audio_io)
        
        audio_segments = []
        for start_time, end_time in dialogs: 
            try:
                start_min, start_sec = map(int, start_time.split(':'))
                end_min, end_sec = map(int, end_time.split(':'))
            except ValueError:
                audio_segments.append(None)
                continue

            if (start_min < 0 or start_sec < 0 or start_sec >= 60) or (end_min < 0 or end_sec < 0 or end_sec >= 60):
                audio_segments.append(None)
                continue

            start_time_msec = (start_min * 60 + start_sec) * 1000
            end_time_msec = (end_min * 60 + end_sec) * 1000

            if start_time_msec >= end_time_msec:
                audio_segments.append(None)
                continue

            # Extract segment
            if end_time_msec > len(audio):  # AudioSegment uses milliseconds
                audio_segments.append(None)
                continue
            
            segment = audio[start_time_msec:end_time_msec]
        
            # Export segment to bytes buffer
            with io.BytesIO() as segment_buffer:
                segment.export(segment_buffer, format='wav')
                segment_buffer.seek(0)
                audio_segments.append(base64.b64encode(segment_buffer.getvalue()))
        
        return audio_segments

    def diarize_audio(base64_audio, filter=None):
        input = [
            {
                "type": "audio/wav",
                "content": base64_audio,
            },
            {
                "type": "text",
                "content": prompt_audio_segmentation,
            },
        ]
        messages = generate_messages(input)
        model = "gemini-2.5-flash-lite"
        asrs = None
        for i in range(1):
            # response, _ = get_response_with_retry(model, messages, timeout=30)
            response, _ = get_response(model, messages, timeout=30)
            asrs = validate_and_fix_json(response)
            if asrs is not None:
                break
        if asrs is None:
            raise Exception("Failed to diarize audio")

        for asr in asrs:
            start_min, start_sec = map(int, asr["start_time"].split(':'))
            end_min, end_sec = map(int, asr["end_time"].split(':'))
            asr["duration"] = (end_min * 60 + end_sec) - (start_min * 60 + start_sec)
            
        asrs = [asr for asr in asrs if filter(asr)]

        return asrs

    def get_normed_audio_embeddings(audios):
        """
        Get normalized audio embeddings for a list of base64 audio strings
        
        Args:
            base64_audios (list): List of base64 encoded audio strings
            
        Returns:
            list: List of normalized audio embeddings
        """
        audio_segments = [audio["audio_segment"] for audio in audios]
        embeddings = get_audio_embeddings(audio_segments)
        normed_embeddings = [normalize_embedding(embedding) for embedding in embeddings]
        for audio, embedding in zip(audios, normed_embeddings):
            audio["embedding"] = embedding
        return audios

    def create_audio_segments(base64_audio, asrs):
        dialogs = [(asr["start_time"], asr["end_time"]) for asr in asrs]
        audio_segments = get_audio_segments(base64_audio, dialogs)
        for asr, audio_segment in zip(asrs, audio_segments):
            asr["audio_segment"] = audio_segment

        return asrs

    def filter_duration_based(audio):
        min_duration = processing_config["min_duration_for_audio"]
        return audio["duration"] >= min_duration

    def update_videograph(video_graph, audios):
        id2audios = {}
        
        for audio in audios:
            # 1. Safe Access: Check if both ASR and Embedding exist
            asr_text = audio.get("asr")
            embedding = audio.get("embedding")
            
            # 2. Skip logic: If there's no speech or no vector, don't add to graph
            if asr_text is None or embedding is None or not str(asr_text).strip():
                continue

            audio_info = {
                "embeddings": [embedding],
                "contents": [asr_text]
            }
            
            # 3. Standard Matching Logic
            matched_nodes = video_graph.search_voice_nodes(audio_info)
            
            if len(matched_nodes) > 0:
                matched_node = matched_nodes[0][0]
                video_graph.update_node(matched_node, audio_info)
                audio["matched_node"] = matched_node
            else:
                matched_node = video_graph.add_voice_node(audio_info)
                audio["matched_node"] = matched_node
                
            # 4. Update the tracking dictionary
            if matched_node not in id2audios:
                id2audios[matched_node] = []
            id2audios[matched_node].append(audio)

        return id2audios

    if not base64_audio:
        return {}

    # Check if intermediate results exist
    try:
        with open(save_path, "r") as f:
            audios = json.load(f)
        for audio in audios:
            audio["audio_segment"] = audio["audio_segment"].encode("utf-8")
    except Exception as e:
        asrs = diarize_audio(base64_audio, filter=filter_duration_based)
        audios = create_audio_segments(base64_audio, asrs)
        audios = [audio for audio in audios if audio["audio_segment"] is not None]

        if len(audios) > 0:
            audios = get_normed_audio_embeddings(audios)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            for audio in audios:
                audio["audio_segment"] = audio["audio_segment"].decode("utf-8")
            json.dump(audios, f)
            for audio in audios:
                audio["audio_segment"] = audio["audio_segment"].encode("utf-8")
        
        logger.info(f"Write voice detection results to {save_path}")
    
    if "voice" in preprocessing or video_graph is None:
        return
    
    if len(audios) == 0:
        return {}

    id2audios = update_videograph(video_graph, audios)

    return id2audios

if __name__ == "__main__":
    video_path = "data/test/clips/0.mp4"
    save_path = "data/test/intermediate_outputs/voice_results.json"
    
    # 1. Extract raw data
    print("Extracting video/audio data...")
    video_b64, _, audio_b64 = process_video_clip(video_path)
    
    # 2. Run the full voice pipeline
    print("Running Voice Processing (Gemini + ERes2Net)...")
    # Note: Pass a dummy video_graph if you just want to test the file output
    results = process_voices(
        video_graph=None, 
        base64_audio=audio_b64, 
        base64_video=video_b64, 
        save_path=save_path,
        preprocessing=["voice"] # Skips graph update for testing
    )
    
    if results is None:
        print("Test Result: Voices processed and saved to JSON.")