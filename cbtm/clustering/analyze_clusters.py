import argparse
import gzip
import itertools
import json
import numpy as np
import pandas as pd
import pickle
import torch
import tqdm
import random
import os
from itertools import chain
from pathlib import Path
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from typing import Dict
from kmeans_pytorch import KMeans as BalancedKMeans
from datasets import load_dataset

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from umap import UMAP
import seaborn as sns


def set_random_seeds(seed=1997):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Call before training
set_random_seeds(1997)

def load_model(path_to_model: Path):
    with open(path_to_model, 'rb') as f:
        out = pickle.load(f)
    return out


def load_tulu_dataset(dataset_name="allenai/tulu-3-sft-mixture", sample_size=20000):
    """Load Tulu dataset and extract user prompts"""
    dataset = load_dataset(dataset_name, split="train")
    
    # Extract user prompts
    user_prompts = []
    for i, example in enumerate(tqdm(dataset, desc="Extracting user prompts")):
        # Find the first user message
        for message in example['messages']:
            if message.get('role') == 'user':
                user_prompts.append({
                    "id": i,
                    "text": message.get('content', ''),
                    "source": example.get('source', ''),
                    "original_id": example.get('id', '')
                })
                break
    
    # Convert to DataFrame and sample
    df = pd.DataFrame(user_prompts)
    
    # Remove empty texts
    df = df[df['text'].str.strip() != '']
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Loaded {len(df)} user prompts from Tulu dataset")
    return df


def vectorize_tulu_dataset(model, dataset_name, sample_size=10000):
    """Vectorize Tulu dataset for evaluation"""
    texts_df = load_tulu_dataset(dataset_name, sample_size=sample_size)
    vecs = model.transform(tqdm(texts_df.text))
    return vecs, texts_df


def get_top_terms(vectorizer, kmeans):
    # this will only work if you use TFIDF vectorizer (which maintains vocab)
    original_space_centroids = vectorizer['svd'].inverse_transform(kmeans.cluster_centers.cpu())
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    vocab = vectorizer['tfidf'].get_feature_names_out()
    top_terms = []
    for i in range(kmeans.n_clusters):
        terms = {}
        for j in range(10):
            terms[f'term_{j}'] = vocab[order_centroids[i, j]]
        top_terms.append(terms)
    return pd.DataFrame(top_terms)

def number_normalizer(tokens):
    """Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    # this vectorizer replaces numbers with #NUMBER token
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))



def analyze_top_terms(vectorizer, kmeans, dataset_name):
    top_terms = get_top_terms(vectorizer, kmeans)
    
    # Print formatted top terms for each cluster
    print("=" * 80)
    print("CLUSTER ANALYSIS - TOP TERMS")
    print("=" * 80)
    
    for i in range(len(top_terms)):
        print(f"\nCluster {i} ({kmeans.cluster_centers.shape[0]} total clusters):")
        print("-" * 40)
        terms = [top_terms.iloc[i][f'term_{j}'] for j in range(10)]
        print(f"Top terms: {', '.join(terms)}")
        
        # Count samples in this cluster
        if hasattr(kmeans, 'labels_'):
            cluster_size = (kmeans.labels_ == i).sum()
            print(f"Size: {cluster_size} samples")
    
    return top_terms

def visualize_clusters_2d(vecs, cluster_labels, n_clusters, output_dir, method='tsne'):
    """Visualize clusters in 2D using t-SNE or UMAP"""
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vecs)//4))
        # For t-SNE, we need to include cluster centers in the original fit
        if hasattr(kmeans, 'cluster_centers_'):
            # Combine data points and cluster centers
            centers = kmeans.cluster_centers_.cpu().numpy()
            combined_data = np.vstack([vecs, centers])
            combined_2d = reducer.fit_transform(combined_data)
            
            # Split back into data points and centers
            vecs_2d = combined_2d[:-n_clusters]
            centers_2d = combined_2d[-n_clusters:]
        else:
            vecs_2d = reducer.fit_transform(vecs)
            centers_2d = None
    else:  # umap
        reducer = UMAP(n_components=2, random_state=42)
        vecs_2d = reducer.fit_transform(vecs)
        
        # UMAP does have transform method
        if hasattr(kmeans, 'cluster_centers_'):
            centers_2d = reducer.transform(kmeans.cluster_centers_.cpu().numpy())
        else:
            centers_2d = None
    
    # Create plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(vecs_2d[:, 0], vecs_2d[:, 1], 
                         c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'Cluster Visualization ({method.upper()})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    # Add cluster centers if available
    if centers_2d is not None:
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')
        plt.legend()
    
    plt.tight_layout()
    path_to_figure = output_dir / f'{method}_cluster_visualization_{str(n_clusters)}_clusters.png'
    plt.savefig(path_to_figure, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_labeled_clusters_2d(vecs, cluster_labels, n_clusters, output_dir, method='tsne', 
                         cluster_names=None, cluster_colors=None):
    """
    Visualize clusters in 2D using t-SNE or UMAP with custom cluster mappings
    
    Args:
        vecs: Feature vectors
        cluster_labels: Cluster assignments for each point
        n_clusters: Number of clusters
        output_dir: Directory to save visualization
        method: 'tsne' or 'umap'
        cluster_names: Dict mapping cluster_id -> name (e.g., {0: "Knowledge", 1: "Math"})
        cluster_colors: Dict mapping cluster_id -> color (e.g., {0: "#FF6B6B", 1: "#3498DB"})
    """
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vecs)//4))
        # For t-SNE, we need to include cluster centers in the original fit
        if hasattr(kmeans, 'cluster_centers_'):
            # Combine data points and cluster centers
            centers = kmeans.cluster_centers_.cpu().numpy()
            combined_data = np.vstack([vecs, centers])
            combined_2d = reducer.fit_transform(combined_data)
            
            # Split back into data points and centers
            vecs_2d = combined_2d[:-n_clusters]
            centers_2d = combined_2d[-n_clusters:]
        else:
            vecs_2d = reducer.fit_transform(vecs)
            centers_2d = None
    else:  # umap
        reducer = UMAP(n_components=2, random_state=42)
        vecs_2d = reducer.fit_transform(vecs)
        
        # UMAP does have transform method
        if hasattr(kmeans, 'cluster_centers_'):
            centers_2d = reducer.transform(kmeans.cluster_centers_.cpu().numpy())
        else:
            centers_2d = None
    
    # Create plot with larger figure for better readability
    plt.figure(figsize=(14, 10))
    
    # Plot each cluster with custom colors and labels
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_points = vecs_2d[mask]
        
        # Get name and color for this cluster
        name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        color = cluster_colors.get(cluster_id, f'C{cluster_id}')  # Default matplotlib colors
        
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=color, 
                   label=f"{name} (C{cluster_id})",
                   alpha=0.7, s=30)
    
    # Add cluster centers if available
    if centers_2d is not None:
        for i, (cx, cy) in enumerate(centers_2d):
            if i < n_clusters:
                # Plot center
                plt.scatter(cx, cy, c='black', marker='x', s=200, linewidths=3)
                
                # Add name annotation
                name = cluster_names.get(i, f"Cluster {i}")
                plt.annotate(name, (cx, cy), 
                           xytext=(10, 10), textcoords='offset points',
                           fontweight='bold', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.5', 
                                   facecolor='white', 
                                   edgecolor='black',
                                   alpha=0.9))
    
    # Styling
    plt.title(f'Dataset Cluster Analysis ({method.upper()})', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    
    # Enhanced legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              fontsize=11, title='Clusters', title_fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Save with descriptive filename
    plt.tight_layout()
    path_to_figure = output_dir / f'labeled_cluster_{method}_visualization_{n_clusters}_clusters.png'
    plt.savefig(path_to_figure, dpi=300, bbox_inches='tight')
    
    # Print summary
    print(f"\n📊 Cluster Visualization Saved: {path_to_figure}")
    print("🎯 Cluster Distribution:")
    for cid in range(n_clusters):
        name = cluster_names.get(cid, f"Cluster {cid}")
        count = (cluster_labels == cid).sum()
        percentage = count / len(cluster_labels) * 100
        print(f"   C{cid}: {name:<25} ({count:,} samples, {percentage:.1f}%)")
    
    plt.show()


def analyze_cluster_samples(metadata, n_samples_per_cluster=5):
    """Show sample prompts from each cluster"""
    
    print("\n" + "=" * 80)
    print("CLUSTER SAMPLES ANALYSIS")
    print("=" * 80)
    
    for cluster_id in sorted(metadata['cluster'].unique()):
        cluster_data = metadata[metadata['cluster'] == cluster_id]
        
        print(f"\n{'='*20} CLUSTER {cluster_id} {'='*20}")
        print(f"Size: {len(cluster_data)} samples")
        print(f"Percentage: {len(cluster_data)/len(metadata)*100:.1f}%")
        
        # Show sample prompts
        samples = cluster_data['text'].head(n_samples_per_cluster)
        for i, sample in enumerate(samples):
            print(f"\nSample {i+1}:")
            print(f"  {sample[:200]}..." if len(sample) > 200 else f"  {sample}")

def cluster_statistics(metadata):
    """Compute and display cluster statistics"""
    
    print("\n" + "=" * 80)
    print("CLUSTER STATISTICS")
    print("=" * 80)
    
    cluster_stats = []
    
    for cluster_id in sorted(metadata['cluster'].unique()):
        cluster_data = metadata[metadata['cluster'] == cluster_id]
        
        # Text length statistics
        text_lengths = cluster_data['text'].str.len()
        
        stats = {
            'cluster': cluster_id,
            'count': len(cluster_data),
            'percentage': len(cluster_data)/len(metadata)*100,
            'avg_length': text_lengths.mean(),
            'median_length': text_lengths.median(),
            'source_diversity': cluster_data['source'].nunique() if 'source' in cluster_data else 'N/A'
        }
        cluster_stats.append(stats)
    
    stats_df = pd.DataFrame(cluster_stats)
    print(stats_df.to_string(index=False))
    
    return stats_df

def analyze_silhouette_tulu(
    base_path: str,
    dataset_name: str,
    sample_size: int = 10000,
    ks: list[int] = [2, 4, 6, 10, 14, 19],
    plot_filename: str = 'tulu_silhouette_scores.png',
    save_plot: bool = True
) -> tuple[list[int], list[float]]:
    """
    Loads TF-IDF and KMeans models saved per k, vectorizes data, calculates silhouette scores,
    and plots performance across different k values.
    """
    ks_done, scores = [], []

    for k in ks:
        k_dir = os.path.join(base_path, str(k))
        tfidf_pkl = os.path.join(k_dir, 'tfidf.pkl')
        kmeans_pkl = os.path.join(k_dir, 'kmeans.pkl')
        if not os.path.exists(tfidf_pkl) or not os.path.exists(kmeans_pkl):
            print(f"⚠️ Skipping k={k}: missing pickle(s)")
            continue

        print(f"\n🔍 Processing k={k}")
        with open(tfidf_pkl, 'rb') as f:
            tfidf = pickle.load(f)
        with open(kmeans_pkl, 'rb') as f:
            kmeans = pickle.load(f)

        X, _ = vectorize_tulu_dataset(tfidf, dataset_name, sample_size=sample_size)
        X = X.toarray() if hasattr(X, 'toarray') else X
        X = X.astype(np.float64)

        if hasattr(kmeans, 'labels_') and len(kmeans.labels_) == X.shape[0]:
            labels = kmeans.labels_
        else:
            # Convert NumPy array to PyTorch tensor
            X_tensor = torch.from_numpy(X).float()
            labels = kmeans.predict(X_tensor)

        if len(np.unique(labels)) < 2:
            print(f"⚠️ Only {len(np.unique(labels))} cluster(s); skipping")
            continue

        score = silhouette_score(X, labels, metric='euclidean')  # 💡 Mean silhouette score :contentReference[oaicite:6]{index=6}
        print(f"✅ k={k} → silhouette score = {score:.4f}")

        ks_done.append(k)
        scores.append(score)

    if not scores:
        print("❌ No valid silhouette scores computed.")
        return ks_done, scores

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ks_done, scores, 'o-', label='Silhouette Score', color='tab:blue')
    best = ks_done[int(np.argmax(scores))]
    best_score = max(scores)
    plt.plot(best, best_score, 'r*', markersize=15, label=f'Best: k={best}')

    plt.xticks(ks_done)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Mean Silhouette Score')
    plt.title('Silhouette Score vs k for Tulu Dataset')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_filename, dpi=300)
        print(f"📈 Plot saved to {plot_filename}")
    plt.show()

    return ks_done, scores


def analyze_davies_bouldin_tulu(
    base_path: str,
    dataset_name: str,
    sample_size: int = 10000,
    ks: list[int] = [2, 4, 6, 10, 14, 19],
    plot_filename: str = 'tulu_davies_bouldin_scores.png',
    save_plot: bool = True
) -> tuple[list[int], list[float]]:
    """
    Loads TF-IDF and KMeans models saved per k, vectorizes data, calculates Davies-Bouldin scores,
    and plots performance across different k values.
    
    Note: Lower Davies-Bouldin scores indicate better clustering.
    """
    ks_done, scores = [], []

    for k in ks:
        k_dir = os.path.join(base_path, str(k))
        tfidf_pkl = os.path.join(k_dir, 'tfidf.pkl')
        kmeans_pkl = os.path.join(k_dir, 'kmeans.pkl')
        if not os.path.exists(tfidf_pkl) or not os.path.exists(kmeans_pkl):
            print(f"⚠️ Skipping k={k}: missing pickle(s)")
            continue

        print(f"\n🔍 Processing k={k}")
        with open(tfidf_pkl, 'rb') as f:
            tfidf = pickle.load(f)
        with open(kmeans_pkl, 'rb') as f:
            kmeans = pickle.load(f)

        X, _ = vectorize_tulu_dataset(tfidf, dataset_name, sample_size=sample_size)
        X = X.toarray() if hasattr(X, 'toarray') else X
        X = X.astype(np.float64)

        if hasattr(kmeans, 'labels_') and len(kmeans.labels_) == X.shape[0]:
            labels = kmeans.labels_
        else:
            # Convert NumPy array to PyTorch tensor
            X_tensor = torch.from_numpy(X).float()
            labels = kmeans.predict(X_tensor)

        if len(np.unique(labels)) < 2:
            print(f"⚠️ Only {len(np.unique(labels))} cluster(s); skipping")
            continue

        score = davies_bouldin_score(X, labels)  # 💡 Lower is better for Davies-Bouldin
        print(f"✅ k={k} → Davies-Bouldin score = {score:.4f}")

        ks_done.append(k)
        scores.append(score)

    if not scores:
        print("❌ No valid Davies-Bouldin scores computed.")
        return ks_done, scores

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ks_done, scores, 'o-', label='Davies-Bouldin Score', color='tab:orange')
    best = ks_done[int(np.argmin(scores))]  # 💡 argmin because lower is better
    best_score = min(scores)  # 💡 min because lower is better
    plt.plot(best, best_score, 'r*', markersize=15, label=f'Best: k={best}')

    plt.xticks(ks_done)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Score vs k for Tulu Dataset (Lower = Better)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_filename, dpi=300)
        print(f"📈 Plot saved to {plot_filename}")
    plt.show()

    return ks_done, scores

def analyze_elbow_tulu(
    base_path: str,
    dataset_name: str,
    sample_size: int = 10000,
    ks: list[int] = [2, 4, 6, 10, 14, 19],
    plot_filename: str = 'tulu_elbow_scores.png',
    save_plot: bool = True
) -> tuple[list[int], list[float]]:
    """
    Loads TF-IDF and KMeans models saved per k, vectorizes data, calculates WCSS (inertia),
    and plots elbow curve across different k values.
    
    Args:
        base_path: Directory containing k subdirectories with pickle files
        dataset_name: Name of dataset to load (e.g., 'allenai/tulu-3-sft-mixture')
        sample_size: Number of samples to use for analysis
        ks: List of k values to analyze
        plot_filename: Output filename for the plot
        save_plot: Whether to save the plot to file
    
    Returns:
        Tuple of (k_values_processed, wcss_scores)
    """
    ks_done, wcss_scores = [], []

    for k in ks:
        k_dir = os.path.join(base_path, str(k))
        tfidf_pkl = os.path.join(k_dir, 'tfidf.pkl')
        kmeans_pkl = os.path.join(k_dir, 'kmeans.pkl')
        
        if not os.path.exists(tfidf_pkl) or not os.path.exists(kmeans_pkl):
            print(f"⚠️ Skipping k={k}: missing pickle(s)")
            continue

        print(f"\n🔍 Processing k={k}")
        
        # Load models
        with open(tfidf_pkl, 'rb') as f:
            tfidf = pickle.load(f)
        with open(kmeans_pkl, 'rb') as f:
            kmeans = pickle.load(f)

        # Vectorize data using the k-specific TF-IDF vectorizer
        X, _ = vectorize_tulu_dataset(tfidf, dataset_name, sample_size=sample_size)
        X = X.toarray() if hasattr(X, 'toarray') else X
        X = X.astype(np.float64)

        # Get cluster labels
        if hasattr(kmeans, 'labels_') and len(kmeans.labels_) == X.shape[0]:
            labels = kmeans.labels_
        else:
            # Convert NumPy array to PyTorch tensor if needed
            if hasattr(kmeans, 'predict') and 'torch' in str(type(kmeans)):
                X_tensor = torch.from_numpy(X).float()
                labels = kmeans.predict(X_tensor)
            else:
                labels = kmeans.predict(X)

        # Check for valid clustering
        if len(np.unique(labels)) < 2:
            print(f"⚠️ Only {len(np.unique(labels))} cluster(s); skipping")
            continue

        # Calculate WCSS (Within-Cluster Sum of Squares)
        # This is the inertia - sum of squared distances to centroids
        if hasattr(kmeans, 'inertia_'):
            wcss = kmeans.inertia_
        else:
            # Calculate manually if inertia not available
            wcss = 0
            unique_labels = np.unique(labels)
            for label in unique_labels:
                cluster_points = X[labels == label]
                if len(cluster_points) > 0:
                    # Get centroid for this cluster
                    if hasattr(kmeans, 'cluster_centers_'):
                        centroid = kmeans.cluster_centers_[label]
                    else:
                        centroid = np.mean(cluster_points, axis=0)
                    
                    # Sum of squared distances to centroid
                    distances = np.sum((cluster_points - centroid) ** 2, axis=1)
                    wcss += np.sum(distances)

        print(f"✅ k={k} → WCSS = {wcss:.2f}")

        ks_done.append(k)
        wcss_scores.append(wcss)

    if not wcss_scores:
        print("❌ No valid WCSS scores computed.")
        return ks_done, wcss_scores

    # Calculate elbow score (optional - rate of change in WCSS)
    elbow_scores = []
    if len(wcss_scores) > 2:
        for i in range(1, len(wcss_scores) - 1):
            # Calculate second derivative approximation
            # Get the actual k values and step sizes
            k_prev, k_curr, k_next = ks[i-1], ks[i], ks[i+1]
            step1 = k_curr - k_prev
            step2 = k_next - k_curr
            
            # Calculate normalized second derivative
            slope1 = (wcss_scores[i] - wcss_scores[i-1]) / step1
            slope2 = (wcss_scores[i+1] - wcss_scores[i]) / step2
            
            # Second derivative normalized by average step size
            avg_step = (step1 + step2) / 2
            elbow_score = (slope2 - slope1) / avg_step
            # elbow_score = wcss_scores[i-1] - 2*wcss_scores[i] + wcss_scores[i+1]
            elbow_scores.append(elbow_score)
        
        # Find the elbow point (maximum second derivative)
        if elbow_scores:
            elbow_idx = np.argmax(elbow_scores) + 1  # +1 because we start from index 1
            elbow_k = ks_done[elbow_idx]
        else:
            elbow_k = ks_done[0]
    else:
        elbow_k = ks_done[0] if ks_done else None

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ks_done, wcss_scores, 'o-', label='WCSS', color='tab:red', linewidth=2, markersize=8)
    
    # Mark the elbow point if found
    if elbow_k is not None and len(wcss_scores) > 2:
        elbow_wcss = wcss_scores[ks_done.index(elbow_k)]
        plt.plot(elbow_k, elbow_wcss, 'g*', markersize=15, label=f'Elbow: k={elbow_k}')

    plt.xticks(ks_done)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method: WCSS vs k for Tulu Dataset')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Add annotation about elbow interpretation
    if len(ks_done) > 1:
        plt.text(0.02, 0.98, 
                'Look for the "elbow" - point where\nWCSS reduction starts to level off', 
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📈 Plot saved to {plot_filename}")
    plt.show()

    # Print summary
    print(f"\n📊 ELBOW METHOD SUMMARY:")
    print(f"k values tested: {ks_done}")
    print(f"WCSS scores: {[f'{score:.1f}' for score in wcss_scores]}")
    if elbow_k is not None and len(wcss_scores) > 2:
        print(f"Suggested elbow point: k={elbow_k}")
    else:
        print("Elbow point detection requires more data points")
    
    # Calculate percentage reduction between consecutive k values
    if len(wcss_scores) > 1:
        print(f"\nWCSS Reduction per k increase:")
        for i in range(1, len(wcss_scores)):
            reduction = ((wcss_scores[i-1] - wcss_scores[i]) / wcss_scores[i-1]) * 100
            print(f"k={ks_done[i-1]} → k={ks_done[i]}: {reduction:.1f}% reduction")

    return ks_done, wcss_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset-name', default="allenai/tulu-3-sft-mixture", type=str,
                       help='Hugging Face dataset name')
    parser.add_argument('--num-clusters', required=True, type=int)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--sample-size', type=int, default=10000, 
                       help='Number of samples to use for training')

    args = parser.parse_args()

    path_to_vectorizer = args.output_dir / "tfidf.pkl"
    path_to_kmeans = args.output_dir / "kmeans.pkl"
    vectorizer = load_model(path_to_vectorizer)
    kmeans = load_model(path_to_kmeans)

    # Get top terms
    top_terms = get_top_terms(vectorizer, kmeans)
    print(top_terms)

    # Load a sample for evaluation
    vecs, metadata = vectorize_tulu_dataset(vectorizer, args.dataset_name, sample_size=args.sample_size)

    print("vecs.shape", vecs.shape)


    # Predict clusters
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    metadata['cluster'] = kmeans.predict(torch.from_numpy(vecs).to(device)).cpu().numpy()

    # Analyze top terms in each cluster
    analyze_top_terms(vectorizer, kmeans, args.dataset_name)

    # Compute and display cluster statistics
    cluster_statistics(metadata)

    # Analyze cluster samples
    analyze_cluster_samples(metadata, 5)

    # Visualize the clusters
    visualize_clusters_2d(vecs, metadata["cluster"], args.num_clusters, args.output_dir, method='tsne')

    # # Custom mapping
    # cluster_names = {
    #     0: "Task Definition & Language Processing",     # Task instructions, language identification
    #     1: "Reasoning & Stream of Consciousness",       # Logic, reasoning, step-by-step explanations
    #     2: "Short Story & Creative Writing",            # Creative narratives, short stories
    #     3: "Translation Tasks & Language Pairs",        # Translation between specific languages
    #     4: "Casual Conversation & Mixed Topics",        # General chat, varied discussions
    #     5: "Alex Character Math & Scenarios",           # Math problems featuring Alex
    #     6: "Malagasy Language Content",                 # Specific to Malagasy language
    #     7: "Social Media & Digital Engagement",         # Social media posts, digital content
    #     8: "Language Identification Tasks",             # Detecting languages, NLP tasks
    #     9: "Blog Posts & Article Writing",              # Blog writing, article creation
    #     10: "Business Revenue & Tax Analysis",          # Company finances, tax calculations
    #     11: "Matrix & Vector Mathematics",              # Linear algebra, mathematical modeling
    #     12: "Performance Scoring & Analytics",          # Score analysis, performance metrics
    #     13: "Sinhala Language Content",                 # Sinhala script and language
    #     14: "Mathematical Sequences & Series",          # Fibonacci, arithmetic sequences
    #     15: "General Information & Assistance",         # Broad help requests
    #     16: "Solar Energy & Scientific Applications",   # Energy calculations, scientific problems
    #     17: "String Processing Functions",              # String manipulation in Python
    #     18: "Array Operations & Algorithms",            # Array processing, numerical algorithms
    #     19: "Italian & Romance Language Mix",           # Italian and related languages
    #     20: "Differential Equations & Greek Letters",   # Advanced math with Greek symbols
    #     21: "Indonesian & Malay Languages",             # Indonesian, Malaysian content
    #     22: "Emily Character Simple Math",              # Basic math with Emily character
    #     23: "Machine Learning & AI Development",        # ML models, AI systems
    #     24: "Geometry & Area Calculations",             # Geometric shapes, measurements
    #     25: "Etsy Titles & Art Descriptions",           # Product titles, art descriptions
    #     26: "Python Dictionary & List Operations",      # Python data structures
    #     27: "Budget Optimization & Problem Solving",    # Financial optimization, complex problems
    #     28: "Historical Events & Questions",            # History-related queries
    #     29: "Advanced Mathematical Functions",          # Complex trigonometry, advanced math
    #     30: "Premise-Hypothesis Logic Tests",           # Logical reasoning, hypothesis testing
    #     31: "Step-by-Step Reasoning Processes",         # Detailed logical explanations
    #     32: "Content Generation & Miscellaneous",       # General content creation
    #     33: "Paragraph Writing & Formatting",           # Structured writing, formatting
    #     34: "Non-English Script Mixed Content",         # Various non-Latin scripts
    #     35: "Programming Class Design",                 # Object-oriented programming
    #     36: "Python Function & List Processing",        # Python function development
    #     37: "Comedic & Vivid Storytelling",            # Comedy, detailed descriptions
    #     38: "Security & Information Protection",        # Cybersecurity, privacy
    #     39: "Jamie Character & Entertainment",          # Problems with Jamie character
    #     40: "Word Analysis & English Linguistics",      # Word processing, English language
    #     41: "Educational & School Mathematics",         # Student problems, classroom scenarios
    #     42: "Graph Theory & Network Analysis",          # Network graphs, connections
    #     43: "Mathematical Optimization Problems",       # Min/max problems, optimization
    #     44: "Logical Premise Evaluation",              # Premise-conclusion testing
    #     45: "Calendar & Daily Scheduling",             # Time management, scheduling
    #     46: "Growth Models & Differential Equations",   # Population growth, differential math
    #     47: "Number Theory & Digit Problems",          # Number properties, combinatorics
    #     48: "Alex Time & Duration Problems",           # Time calculations with Alex
    #     49: "Group Events & Social Organization",       # Event planning, group management
    #     50: "Set Theory & Mathematical Sets",          # Set operations, mathematical sets
    #     51: "Emily Daily Life Problems",               # Simple problems with Emily
    #     52: "Cyrillic Mixed Content",                  # Russian/Ukrainian mixed content
    #     53: "Area & Measurement Calculations",         # Geometric measurements
    #     54: "Age & Time-related Problems",             # Age calculations, temporal problems
    #     55: "Python Code Debugging & Analysis",        # Code review, error fixing
    #     56: "Question-Answer General Knowledge",        # Broad Q&A, information queries
    #     57: "Python Function Development",             # Custom Python function creation
    #     58: "Cookies & Simple Item Counting",          # Basic counting problems
    #     59: "Probability & Statistical Analysis",       # Statistics, probability calculations
    #     60: "Geometric Points & Lines",                # Point geometry, coordinate systems
    #     61: "Spanish Mixed Content",                   # Spanish language with mixed elements
    #     62: "AI Character & World Building",           # AI discussions, character development
    #     63: "Cyrillic Technical Content",              # Russian/Ukrainian technical material
    #     64: "Python Error Correction",                 # Debugging, error handling
    #     65: "Python Dictionary Operations",            # Dictionary processing, key-value pairs
    #     66: "World Building & Content Generation",      # Creative world creation
    #     67: "Books & Reading Content",                 # Book-related problems, reading
    #     68: "String Character Manipulation",           # Character-level string processing
    #     69: "City Travel & Distance Problems",         # Travel calculations, city problems
    #     70: "General Help & Technology",               # Technical assistance, general help
    #     71: "Detailed Instructions & Health",          # Comprehensive explanations, health
    #     72: "Translation & Language Change",           # Translation tasks, language conversion
    #     73: "List Processing & Task Instructions",     # List operations, task completion
    #     74: "Task Classification & Q&A",              # Task-based questions, classification
    #     75: "Cost & Budget Calculations",             # Financial calculations, pricing
    #     76: "Sentence Logic & Common Sense",          # Sentence evaluation, logical reasoning
    #     77: "Casual Chat & Varied Topics",            # Informal conversation, mixed subjects
    #     78: "Travel & Speed Calculations",            # Transportation problems, speed/distance
    #     79: "Tamil Language Content",                 # Tamil script and language
    #     80: "Business Cost & Revenue Analysis",       # Business mathematics, cost analysis
    #     81: "Detailed Response Formatting",           # Structured responses, bullet points
    #     82: "Spanish Language Content",               # Spanish text and questions
    #     83: "Mathematical Growth Functions",           # Growth models, rate calculations
    #     84: "AI & Programming Ethics",                # AI development, programming ethics
    #     85: "Function & Data Handling",               # Function processing, data management
    #     86: "Program Design & Development",           # Software development, program creation
    #     87: "Tax & File Processing",                  # Document processing, legal text
    #     88: "Film & Company Classification",          # Content classification, categories
    #     89: "Integer & Digit Operations",             # Number operations, integer math
    #     90: "File & Script Processing",               # File handling, script development
    #     91: "Mathematical Calculations & Arrays",     # Mathematical operations, array math
    #     92: "Time Duration & Scheduling",             # Time calculations, scheduling
    #     93: "Response Creation & Formatting",         # Response structure, content creation
    #     94: "Music & Entertainment Content",          # Songs, music industry
    #     95: "Word & Calculation Problems",            # Word problems, calculation tasks
    #     96: "Spanish & Chinese Mixed Content",        # Spanish with Chinese elements
    #     97: "Casual Greetings & Simple Responses",    # Basic interactions, greetings
    #     98: "Task Processing & PersonX",              # Specific task formats, PersonX references
    #     99: "Algorithm & Performance Analysis",       # Algorithm complexity, performance
    #     100: "Polynomial & Equation Solving",         # Mathematical equations, polynomials
    #     101: "SQL Database Operations",               # Database queries, SQL commands
    #     102: "Medical Research & JSON Processing",    # Medical abstracts, structured data
    #     103: "Prime Numbers & Algorithms",            # Prime number algorithms, mathematical proofs
    #     104: "Health & Mental Wellness",              # Health information, mental health
    #     105: "Weekly Time & Training Schedules",      # Training programs, weekly planning
    #     106: "Quantum Physics & Advanced Concepts",   # Quantum computing, theoretical physics
    #     107: "Books & Library Collections",           # Book management, library systems
    #     108: "Yes/No Questions & Claims",             # Binary questions, claim verification
    #     109: "Sports Physics & Measurements",         # Sports calculations, physics problems
    #     110: "Team Sports & Gaming",                  # Sports teams, game statistics
    #     111: "Data Structure & Processing",           # Data analysis, structure processing
    #     112: "Community & Group Management",          # People management, community organization
    #     113: "Japanese & Miscellaneous Content",      # Japanese language with mixed content
    #     114: "Character Development & Narratives",    # Story characters, narrative development
    #     115: "Cyrillic & Advanced Mathematics",       # Russian/Ukrainian with complex math
    #     116: "Romance Languages (European)",          # French, Italian, Spanish mix
    #     117: "Counting & Numerical Problems",         # Number counting, numerical reasoning
    #     118: "Advanced Calculus & Functions",         # Complex mathematical functions
    #     119: "SVG Graphics & Web Development",        # SVG code, web graphics
    #     120: "Creative Content & Game Development",   # Creative projects, game design
    #     121: "Population & Growth Models",            # Population studies, growth analysis
    #     122: "Annual & Time-based Calculations",      # Yearly calculations, time periods
    #     123: "Database & Security Systems",           # Database management, security
    #     124: "Product Reviews & Sentiment",           # Review analysis, product feedback
    #     125: "Hours & Time Management",               # Time tracking, hour calculations
    #     126: "Trigonometry & Advanced Math",          # Trigonometric functions, complex math
    #     127: "Geometric Shapes & Drawing"             # Geometry, shape calculations
    # }

    # custom_colors = {
    #     0: '#FF5733', 1: '#33FF57', 2: '#F1FF33', 3: '#FF33F1', 4: '#9333FF',
    #     5: '#33F1FF', 6: '#FF0000', 7: '#0066FF', 8: '#FF6600', 9: '#800080',
    #     10: '#008000', 11: '#FFD700', 12: '#4B0082', 13: '#8B4513', 14: '#2E8B57',
    #     15: '#FF69B4', 16: '#DC143C', 17: '#00CED1', 18: '#228B22', 19: '#DA70D6',
    #     20: '#4169E1', 21: '#CD853F', 22: '#FF1493', 23: '#32CD32', 24: '#FF8C00',
    #     25: '#6A5ACD', 26: '#20B2AA', 27: '#B22222', 28: '#3CB371', 29: '#8A2BE2',
    #     30: '#00FF7F', 31: '#DC143C', 32: '#FF4500', 33: '#9370DB', 34: '#00FA9A',
    #     35: '#FF6347', 36: '#40E0D0', 37: '#EE82EE', 38: '#90EE90', 39: '#F0E68C',
    #     40: '#DDA0DD', 41: '#98FB98', 42: '#F5DEB3', 43: '#FFB6C1', 44: '#87CEEB',
    #     45: '#D2B48C', 46: '#778899', 47: '#B0C4DE', 48: '#FFFFE0', 49: '#00FFFF',
    #     50: '#FAFAD2', 51: '#FFE4E1', 52: '#DCDCDC', 53: '#FDF5E6', 54: '#F0F8FF',
    #     55: '#F5F5F5', 56: '#FFF8DC', 57: '#FFFACD', 58: '#FDF5E6', 59: '#F0FFFF',
    #     60: '#F5FFFA', 61: '#FFF5EE', 62: '#F0F0F0', 63: '#FFFAFA', 64: '#F8F8FF',
    #     65: '#F5F5DC', 66: '#FDF5E6', 67: '#FFFAF0', 68: '#F0FFF0', 69: '#F5FFFA',
    #     70: '#F0F8FF', 71: '#E6E6FA', 72: '#FFF0F5', 73: '#FFE4E1', 74: '#FFEBCD',
    #     75: '#FFE4B5', 76: '#FFDEAD', 77: '#F5DEB3', 78: '#DDA0DD', 79: '#DA70D6',
    #     80: '#FF69B4', 81: '#FF1493', 82: '#DC143C', 83: '#B22222', 84: '#A0522D',
    #     85: '#8B4513', 86: '#D2691E', 87: '#CD853F', 88: '#F4A460', 89: '#DEB887',
    #     90: '#D2B48C', 91: '#BC8F8F', 92: '#F0E68C', 93: '#EEE8AA', 94: '#BDB76B',
    #     95: '#F5DEB3', 96: '#FFE4B5', 97: '#FFDEAD', 98: '#F5DEB3', 99: '#DDA0DD',
    #     100: '#D8BFD8', 101: '#DDA0DD', 102: '#EE82EE', 103: '#DA70D6', 104: '#FF69B4',
    #     105: '#FF1493', 106: '#DC143C', 107: '#B22222', 108: '#A0522D', 109: '#8B4513',
    #     110: '#D2691E', 111: '#CD853F', 112: '#F4A460', 113: '#DEB887', 114: '#D2B48C',
    #     115: '#BC8F8F', 116: '#F0E68C', 117: '#EEE8AA', 118: '#BDB76B', 119: '#9ACD32',
    #     120: '#ADFF2F', 121: '#7FFF00', 122: '#7CFC00', 123: '#00FF00', 124: '#32CD32',
    #     125: '#98FB98', 126: '#90EE90', 127: '#00FA9A'
    # }

    # visualize_labeled_clusters_2d(vecs, metadata["cluster"], args.num_clusters, args.output_dir, 
    #                     method='tsne',
    #                     cluster_names=cluster_names, 
    #                     cluster_colors=custom_colors)


    analyze_elbow_tulu(
        base_path='/home/ehghaghi/scratch/ehghaghi/clusters/allenai/tulu-3-sft-mixture',
         dataset_name='allenai/tulu-3-sft-mixture',
         sample_size=10000,
         ks=[2, 4, 8, 16, 32, 64, 128],
         plot_filename=args.output_dir / 'tulu_elbow_scores.png',
         save_plot=True)

    analyze_silhouette_tulu(
        base_path='/home/ehghaghi/scratch/ehghaghi/clusters/allenai/tulu-3-sft-mixture',
        dataset_name='allenai/tulu-3-sft-mixture',
        sample_size=10000,
        ks=[2, 4, 8, 16, 32, 64, 128],
        plot_filename=args.output_dir / 'tulu_silhouette_scores.png',
        save_plot=True)


    analyze_davies_bouldin_tulu(
        base_path='/home/ehghaghi/scratch/ehghaghi/clusters/allenai/tulu-3-sft-mixture',
         dataset_name='allenai/tulu-3-sft-mixture',
         sample_size=10000,
         ks=[2, 4, 8, 16, 32, 64, 128],
         plot_filename=args.output_dir / 'tulu_davies_bouldin_scores.png',
         save_plot=True)



    
