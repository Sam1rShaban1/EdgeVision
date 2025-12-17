import cv2
import numpy as np
import pickle
import os
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN

# =========================================================================
#  CONFIGURATION
# =========================================================================
RAW_DIR = r"C:\Users\Administrator\Desktop\dataset\dataset" # Your photos path

# CHANGE THIS to "buffalo_l" or "buffalo_s"
MODEL_NAME = "buffalo_s" 

# AUTO-GENERATED FILENAME
DB_OUTPUT = f"embeddings_{MODEL_NAME}.pkl"

# ADVANCED TUNING
MIN_DET_SCORE = 0.60      # Reject blurry/bad faces
CLUSTER_EPS = 0.50        # Similarity threshold for grouping "looks"

# =========================================================================
#  GENERATION LOGIC
# =========================================================================
print(f"🚀 Initializing {MODEL_NAME} for Ultimate Database Generation...")
app = FaceAnalysis(name=MODEL_NAME, root='.')
app.prepare(ctx_id=0, det_size=(640, 640)) 

embeddings_db = {}
total_sub_centers = 0

print(f"📂 Scanning Data: {RAW_DIR}...")
print(f"💾 Target File: {DB_OUTPUT}")

if not os.path.exists(RAW_DIR):
    print("❌ Error: Dataset path not found.")
    exit()

for person_name in os.listdir(RAW_DIR):
    person_dir = os.path.join(RAW_DIR, person_name)
    if not os.path.isdir(person_dir): continue
    
    print(f"👤 Analyzing Identity: {person_name}")
    
    valid_vectors = []
    valid_scores = []
    
    # 1. HARVEST DATA
    for file in os.listdir(person_dir):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        path = os.path.join(person_dir, file)
        img = cv2.imread(path)
        if img is None: continue
        
        faces = app.get(img)
        if not faces: continue
        
        # Pick largest face
        face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))[-1]
        
        # QUALITY GATE
        if face.det_score < MIN_DET_SCORE:
            continue
            
        # Normalize
        emb = face.embedding / np.linalg.norm(face.embedding)
        valid_vectors.append(emb)
        valid_scores.append(face.det_score)

    if len(valid_vectors) == 0:
        print(f"   ❌ No valid images for {person_name}")
        continue

    # 2. CLUSTERING (Sub-Centers)
    X = np.array(valid_vectors)
    
    if len(valid_vectors) < 5:
        # Not enough data to cluster, use single weighted average
        final_centers = [np.average(X, axis=0, weights=valid_scores)]
    else:
        # DBSCAN Clustering
        clustering = DBSCAN(eps=CLUSTER_EPS, min_samples=2, metric="euclidean").fit(X)
        labels = clustering.labels_
        
        final_centers = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1: continue # Skip noise/outliers
            
            # Weighted Average of this cluster
            mask = (labels == label)
            cluster_vectors = X[mask]
            cluster_scores = np.array(valid_scores)[mask]
            
            centroid = np.average(cluster_vectors, axis=0, weights=cluster_scores)
            centroid = centroid / np.linalg.norm(centroid)
            final_centers.append(centroid)
        
        # Fallback
        if not final_centers:
            final_centers = [np.mean(X, axis=0)]

    # 3. STORE
    # Normalize result one last time to be safe
    final_centers = [c / np.linalg.norm(c) for c in final_centers]
    embeddings_db[person_name] = final_centers
    total_sub_centers += len(final_centers)
    print(f"   ✅ Saved {len(final_centers)} sub-centers for {person_name}.")

# 4. SAVE TO FILE
with open(DB_OUTPUT, "wb") as f:
    pickle.dump(embeddings_db, f)

print(f"\n✅ DATABASE GENERATED: {DB_OUTPUT}")
print(f"   Model Used: {MODEL_NAME}")
print(f"   Total Identities: {len(embeddings_db)}")
print(f"   Total Reference Vectors: {total_sub_centers}")