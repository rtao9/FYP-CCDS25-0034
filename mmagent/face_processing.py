import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import json
import os
from insightface.app import FaceAnalysis

# Local project imports
from mmagent.utils.face_extraction import extract_faces
from mmagent.utils.face_clustering import cluster_faces
from mmagent.utils.video_processing import process_video_clip

processing_config = json.load(open("configs/processing_config.json"))
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)
cluster_size = processing_config["cluster_size"]

class Face:
    def __init__(self, frame_id, bbox, embedding, cluster_id, metadata):
        self.frame_id = frame_id
        self.bbox = bbox
        self.embedding = embedding
        self.cluster_id = cluster_id
        self.metadata = metadata
    
def get_face(frames):
    """Bridge between raw extraction and the Face class."""
    extracted_faces = extract_faces(face_app, frames)
    # Synchronizing keys: ensures face_emb is pulled correctly
    faces = [Face(
        frame_id=f['frame_id'], 
        bbox=f['bbox'], 
        embedding=f['embedding'], 
        cluster_id=f.get('cluster_id', -1), 
        metadata=f['metadata']
    ) for f in extracted_faces]
    return faces

def cluster_face(faces):
    faces_json = [{
        'frame_id': f.frame_id, 
        'bbox': f.bbox, 
        'embedding': f.embedding, 
        'cluster_id': f.cluster_id, 
        'metadata': f.metadata
    } for f in faces]

    # Run the HDBSCAN logic
    clustered_faces = cluster_faces(faces_json, min_cluster_size=2)
    
    faces = [Face(
        frame_id=f['frame_id'], 
        bbox=f['bbox'], 
        embedding=f['embedding'], 
        cluster_id=f['cluster_id'], 
        metadata=f['metadata']
    ) for f in clustered_faces]
    return faces

def process_faces(video_graph, base64_frames, save_path, preprocessing=[]):
    # Determine batching based on config
    batch_size = max(len(base64_frames) // cluster_size, 4)
    
    def _process_batch(params):
        frames, offset = params
        faces = get_face(frames)
        for face in faces:
            face.frame_id += offset
        return faces

    def get_embeddings(base64_frames, batch_size):
        num_batches = (len(base64_frames) + batch_size - 1) // batch_size
        batched_frames = [
            (base64_frames[i * batch_size : (i + 1) * batch_size], i * batch_size)
            for i in range(num_batches)
        ]

        faces = []
        # Parallel processing: using 4 workers to avoid GPU bottleneck
        with ThreadPoolExecutor(max_workers=min(num_batches, 4)) as executor:
            for batch_faces in tqdm(executor.map(_process_batch, batched_frames), 
                                total=num_batches, desc="Extracting Faces"):
                faces.extend(batch_faces)

        faces = cluster_face(faces)
        return faces

    def establish_mapping(faces, key="cluster_id", filter_func=None):
        mapping = {}
        for face in faces:
            if key not in face:
                continue
            if filter_func and not filter_func(face):
                continue
            
            cid = face[key]
            if cid == -1: continue # Skip noise
            
            if cid not in mapping:
                mapping[cid] = []
            mapping[cid].append(face)
        
        # Sort and prune based on quality
        max_faces = processing_config.get("max_faces_per_character", 5)
        for cid in mapping:
            mapping[cid] = sorted(
                mapping[cid],
                key=lambda x: (
                    float(x["metadata"]["det_score"]),
                    float(x["metadata"]["quality_score"]),
                ),
                reverse=True,
            )[:max_faces]
        return mapping

    def filter_score_based(face):
        dthresh = processing_config["face_detection_score_threshold"]
        qthresh = processing_config["face_quality_score_threshold"]
        return float(face["metadata"]["det_score"]) > dthresh and \
            float(face["metadata"]["quality_score"]) > qthresh

    # 1. Check for Cached Results
    if os.path.exists(save_path):
        try:
            with open(save_path, "r") as f:
                faces_json = json.load(f)
            print(f"Loaded cached face data from {save_path}")
        except:
            faces_json = []
    else:
        # 2. Run Extraction and Clustering
        faces_objs = get_embeddings(base64_frames, batch_size)
        faces_json = [
            {
                "frame_id": f.frame_id,
                "bbox": f.bbox,
                "embedding": f.embedding,
                "cluster_id": int(f.cluster_id),
                "metadata": f.metadata,
            }
            for f in faces_objs
        ]
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(faces_json, f)

    def update_videograph(video_graph, tempid2faces):
        id2faces = {}
        for tempid, faces in tempid2faces.items():
            if tempid == -1:
                continue
            if len(faces) == 0:
                continue
            face_info = {
                "embeddings": [face["embedding"] for face in faces],
                "contents": [face["metadata"]["face_thumb"] for face in faces],
            }
            matched_nodes = video_graph.search_img_nodes(face_info)
            if len(matched_nodes) > 0:
                matched_node = matched_nodes[0][0]
                video_graph.update_node(matched_node, face_info)
                for face in faces:
                    face["matched_node"] = matched_node
            else:
                matched_node = video_graph.add_img_node(face_info)
                for face in faces:
                    face["matched_node"] = matched_node
            if matched_node not in id2faces:
                id2faces[matched_node] = []
            id2faces[matched_node].extend(faces)
        
        max_faces = processing_config["max_faces_per_character"]
        for id, faces in id2faces.items():
            id2faces[id] = sorted(
                faces,
                key=lambda x: (
                    float(x["metadata"]["det_score"]),
                    float(x["metadata"]["quality_score"]),
                ),
                reverse=True
            )[:max_faces]

        return id2faces

    # Check if intermediate results exist
    try:
        with open(save_path, "r") as f:
            faces_json = json.load(f)
    except Exception as e:
        faces = get_embeddings(base64_frames, batch_size)

        faces_json = [
            {
                "frame_id": face.frame_id,
                "bounding_box": face.bounding_box,
                "face_emb": face.face_emb,
                "cluster_id": int(face.cluster_id),
                "extra_data": face.extra_data,
            }
            for face in faces
        ]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
        with open(save_path, "w") as f:
            json.dump(faces_json, f)
            
    if "face" in preprocessing:
        return

    if len(faces_json) == 0:
        return {}

    tempid2faces = establish_mapping(faces_json, key="cluster_id")
    if len(tempid2faces) == 0:
        return {}

    id2faces = update_videograph(video_graph, tempid2faces)

    return id2faces

# --- EXECUTION ---
def main():
    # Use your existing video processing utility to get base64 frames
    video_path = "data/test/clips/0.mp4"
    save_path="data/test/intermediate_outputs/face_detection_results.json"

    print(f"Processing video: {video_path}")
    _, frames, _ = process_video_clip(video_path)
    
    # Run the face pipeline
    results = process_faces(
        video_graph=None, 
        base64_frames=frames, 
        save_path=save_path, 
        preprocessing=["face"]
    )

    # 1. Gather stats
    entity_counts = {}
    noise_count = 0
    
    for face in results:
        # Handle both dictionary and object formats
        cid = face['cluster_id'] if isinstance(face, dict) else face.cluster_id
        
        if cid == -1:
            noise_count += 1
        else:
            entity_counts[cid] = entity_counts.get(cid, 0) + 1

    # 2. Print Summary Table
    print("\n" + "="*50)
    print(f"{'ENTITY SUMMARY':^50}")
    print("="*50)
    print(f"{'Entity ID':<15} | {'Instances (Frames)':<20}")
    print("-" * 50)
    
    # Sort entities by ID
    for eid in sorted(entity_counts.keys()):
        print(f"Person {eid:<10} | {entity_counts[eid]:<20}")
    
    print("-" * 50)
    print(f"Total Unique Entities: {len(entity_counts)}")
    print(f"Total Noise (Detections ignored): {noise_count}")
    print(f"Total Instances Detected: {len(results)}")
    print("="*50)

if __name__ == "__main__":
    main()