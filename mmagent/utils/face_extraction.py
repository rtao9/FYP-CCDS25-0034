import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import base64

def extract_faces(face_app, image_list, num_workers=4):
    faces = []

    def process_image(args):
        frame_idx, img_base64 = args
        try:
            # 1. Efficient Decoding
            img_bytes = base64.b64decode(img_base64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                return []

            # 2. Perception
            detected_faces = face_app.get(img)
            frame_faces = []

            for face in detected_faces:
                # Filter out low-confidence detections immediately
                if face.det_score < 0.6:
                    continue

                bbox = face.bbox.astype(int).tolist()
                
                # InsightFace already provides normed_embedding
                # This is the 'numerical fingerprint' for clustering
                embedding = face.normed_embedding.tolist()

                # Calculate Quality Score (L2 Norm of raw embedding)
                qscore = np.linalg.norm(face.embedding)

                # 3. Determine Face Orientation (Helper for M3 reasoning)
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                aspect_ratio = h / w if w > 0 else 0
                face_type = "ortho" if 1.0 < aspect_ratio < 1.5 else "side"

                # 4. Encode Face Crop (Thumbnail for Gemini/Gallery)
                # Ensure coordinates are within image bounds
                img_h, img_w = img.shape[:2]
                face_img = img[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                
                face_base64 = ""
                if face_img.size > 0:
                    _, buffer = cv2.imencode('.jpg', face_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    face_base64 = base64.b64encode(buffer).decode('utf-8')

                # 5. M3-Agent Structured Output
                frame_faces.append({
                    "frame_id": frame_idx,
                    "bbox": bbox,
                    "embedding": embedding,
                    "cluster_id": -1,  # To be filled by the clustering script
                    "metadata": {
                        "face_type": face_type,
                        "face_thumb": face_base64,
                        "det_score": round(float(face.det_score), 4),
                        "quality_score": round(float(qscore), 2),
                        "age": int(getattr(face, 'age', 0)),
                        "gender": "M" if getattr(face, 'gender', 0) == 1 else "F"
                    }
                })
                
            return frame_faces

        except Exception as e:
            print(f"Error on frame {frame_idx}: {str(e)}")
            return []

    # Process in parallel
    indexed_inputs = list(enumerate(image_list))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # map preserves the order of the input list
        results = list(tqdm(executor.map(process_image, indexed_inputs), 
                           total=len(image_list), 
                           desc="Extracting Faces"))
        
    # Flatten the list of lists
    for result in results:
        faces.extend(result)

    return faces