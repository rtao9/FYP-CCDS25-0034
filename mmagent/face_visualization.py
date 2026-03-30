import json
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def visualize_face_clusters(results_path="data/test/intermediate_outputs/clip_5_faces.json"):
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run face_processing.py first.")
        return

    with open(results_path, "r") as f:
        faces_data = json.load(f)

    # 1. Group faces by their Cluster ID
    clusters = {}
    for face in faces_data:
        cid = face.get("cluster_id", -1)
        if cid == -1: continue  # Skip noise
        
        # We take the face with the highest quality score as the "Profile Pic"
        if cid not in clusters:
            clusters[cid] = face
        else:
            current_q = float(clusters[cid]["metadata"].get("quality_score", 0))
            new_q = float(face["metadata"].get("quality_score", 0))
            if new_q > current_q:
                clusters[cid] = face

    if not clusters:
        print("No valid clusters found to visualize.")
        return

    # 2. Setup Plotting Grid
    num_clusters = len(clusters)
    cols = 4
    rows = math.ceil(num_clusters / cols)
    
    plt.figure(figsize=(15, 4 * rows))
    plt.suptitle("M3-Agent: Identity Memory Gallery", fontsize=16, fontweight='bold')

    # 3. Decode and Plot
    for i, cid in enumerate(sorted(clusters.keys())):
        face = clusters[cid]
        # Decode the Base64 thumbnail
        img_data = base64.b64decode(face["metadata"]["face_thumb"])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_rgb)
        plt.title(f"Entity ID: {cid}\nDetected in Frame: {face['frame_id']}")
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    import math # Needed for ceil
    visualize_face_clusters()