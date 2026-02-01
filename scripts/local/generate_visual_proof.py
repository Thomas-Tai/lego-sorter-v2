import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
from scipy.spatial.distance import cdist
import time

def load_embeddings(db_path):
    print(f"Loading database from {db_path}...")
    with open(db_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} embeddings.")
    return data

def parse_part_id(filename):
    # Filename format: PartID_ColorID_Rotation.jpg or similar
    # e.g. 3001_15.jpg -> 3001
    # e.g. 3001_15_0.jpg -> 3001
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    if len(parts) >= 1:
        return parts[0]
    return "unknown"

def generate_tsne_plot(embeddings_dict, sample_size=2000, output_file="tsne_plot.png"):
    print(f"Generating T-SNE plot (Sample: {sample_size})...")
    
    # Stratified Sampling (try to get different parts)
    all_keys = list(embeddings_dict.keys())
    if len(all_keys) > sample_size:
        indices = np.random.choice(len(all_keys), sample_size, replace=False)
        keys = [all_keys[i] for i in indices]
    else:
        keys = all_keys
        
    vectors = np.array([embeddings_dict[k] for k in keys])
    labels = [parse_part_id(k) for k in keys]
    
    # T-SNE
    # Rely on defaults to avoid version issues
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    vis_data = tsne.fit_transform(vectors)
    
    # Plot
    plt.figure(figsize=(16, 12))
    
    # Dynamic coloring based on Part ID
    unique_labels = list(set(labels))
    # Map labels to integers for coloring
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    c_values = [label_to_int[l] for l in labels]
    
    scatter = plt.scatter(vis_data[:, 0], vis_data[:, 1], c=c_values, cmap='tab20', alpha=0.6, s=10)
    plt.title(f"T-SNE Visualization of LEGO Part Embeddings (N={len(keys)})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(scatter, label="Part ID Cluster")
    
    # Annotate a few points to show variety
    for i in range(0, len(keys), max(1, len(keys)//20)):
        plt.annotate(labels[i], (vis_data[i, 0], vis_data[i, 1]), fontsize=8, alpha=0.8)
        
    plt.savefig(output_file, dpi=150)
    print(f"Saved T-SNE plot to {output_file}")

def verify_retrieval_performance(embeddings_dict, test_size=1000):
    print(f"\nVerifying Retrieval Performance (Sample: {test_size})...")
    all_keys = list(embeddings_dict.keys())
    total_items = len(all_keys)
    
    if total_items > test_size:
        indices = np.random.choice(total_items, test_size, replace=False)
        test_keys = [all_keys[i] for i in indices]
    else:
        test_keys = all_keys
        
    # Prepare matrix for bulk calculation
    # Only if memory allows. 82k * 82k floats is huge (~26GB). Can't do full matrix.
    # We will do batch processing against the full DB.
    
    # For speed, verify "Self-Recall" (distance should be 0)
    # And check if Top-5 contains same Part ID.
    
    vector_dim = len(embeddings_dict[test_keys[0]])
    all_vectors = np.array(list(embeddings_dict.values()))
    all_filenames = list(embeddings_dict.keys())
    
    # To search efficiently, we might need a KDTree or just brute force for 1000 queries.
    # 1000 queries x 80000 db = 80M dot products. Doable.
    
    top1_correct = 0
    top5_same_part_rate = 0
    
    test_vectors = np.array([embeddings_dict[k] for k in test_keys])
    
    # Cosine Similarity = Dot Product (since normalized)
    # We want Max Similarity.
    
    print("Computing similarities...")
    start_time = time.time()
    
    # Process in chunks to avoid blowing memory
    chunk_size = 100
    for i in range(0, len(test_keys), chunk_size):
        chunk_keys = test_keys[i : i+chunk_size]
        chunk_vecs = test_vectors[i : i+chunk_size]
        
        # Shape: (Chunk, DB_Size)
        sim_matrix = np.matmul(chunk_vecs, all_vectors.T)
        
        for j, key in enumerate(chunk_keys):
            scores = sim_matrix[j]
            # Get Top 5 indices
            top_indices = np.argpartition(scores, -5)[-5:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            
            top_files = [all_filenames[idx] for idx in top_indices]
            
            # Check Top 1 (Self)
            if top_files[0] == key:
                top1_correct += 1
            
            # Check Top 5 for Same Part ID
            query_part = parse_part_id(key)
            matches = 0
            for res_file in top_files:
                if parse_part_id(res_file) == query_part:
                    matches += 1
            
            top5_same_part_rate += (matches / 5.0)
            
    elapsed = time.time() - start_time
    print(f"Verification completed in {elapsed:.2f}s")
    
    acc_top1 = (top1_correct / len(test_keys)) * 100
    avg_relevance = (top5_same_part_rate / len(test_keys)) * 100
    
    print(f"Top-1 Accuracy (Identity): {acc_top1:.2f}%")
    print(f"Top-5 Mean Relevance (Same Part ID): {avg_relevance:.2f}%")
    
    return acc_top1, avg_relevance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default=r"data/embeddings/legacy_embeddings.pkl")
    parser.add_argument("--output", type=str, default="models/proof_tsne.png")
    args = parser.parse_args()
    
    # Resolve Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    db_path = args.db
    if not os.path.isabs(db_path):
        db_path = os.path.join(project_root, db_path)
        
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)
        
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return

    data = load_embeddings(db_path)
    
    if len(data) == 0:
        print("Empty DB.")
        return

    # 1. Visualization
    try:
        generate_tsne_plot(data, sample_size=2000, output_file=output_path)
    except Exception as e:
        print(f"TSNE failed: {e}")

    # 2. Statistics
    verify_retrieval_performance(data, test_size=500)

if __name__ == "__main__":
    main()
