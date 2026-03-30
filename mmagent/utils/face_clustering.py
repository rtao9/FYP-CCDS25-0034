import numpy as np
import hdbscan
from sklearn.preprocessing import normalize

def cluster_faces(faces, min_cluster_size=2):
    # 1. Extraction with Safety
    embeddings = []
    metadata = []
    
    if not faces or len(faces) == 0:
        return []

    for i, face in enumerate(faces):
        embeddings.append(face["embedding"])
        metadata.append({
            "idx": i,
            "dscore": float(face["metadata"]["det_score"]),
            "qscore": float(face["metadata"]["quality_score"])
        })

    if not embeddings:
        return []
    
    embeddings = np.array(embeddings)
    # CRITICAL: Normalize embeddings so Dot Product == Cosine Similarity
    embeddings = normalize(embeddings, axis=1)

    # 2. Advanced Masking
    # M3-Agent prefers quality over quantity for the 'anchor' clusters
    good_mask = np.array([(m["dscore"] >= 0.8 and m["qscore"] >= 20) for m in metadata])
    
    all_labels = np.full(len(faces), -1)

    # 3. Clustering Logic
    if np.sum(good_mask) >= min_cluster_size:
        good_embs = embeddings[good_mask]
        
        # Calculate Distances: Max(0, ...) prevents tiny negative numbers due to float error
        similarity = np.dot(good_embs, good_embs.T)
        distances = np.clip(1 - similarity, 0, 2).astype(np.float64)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, 
            min_samples=1,                 # Makes the algorithm less conservative
            cluster_selection_epsilon=0.7, # CRITICAL: Merges clusters that are close
            metric='precomputed',
            cluster_selection_method='eom' # 'eom' (Excess of Mass) is better than 'leaf' for merging
        )
        good_labels = clusterer.fit_predict(distances)

        # Map labels back to the original list
        all_labels[good_mask] = good_labels

    # 4. Final Data Assembly
    result_faces = []
    for i, face in enumerate(faces):
        f_copy = face.copy()
        f_copy["cluster_id"] = int(all_labels[i]) # Ensure it's a standard Python int for JSON
        result_faces.append(f_copy)

    return result_faces