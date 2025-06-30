"""
Neural Representation Evaluation Module
======================================
Comprehensive evaluation metrics for neural representation learning embeddings.
Includes clustering quality, RSA, linear probing, and visualization tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    calinski_harabasz_score, classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage
import umap
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings
from collections import defaultdict
import itertools

warnings.filterwarnings('ignore')

# ========================
# CONFIGURATION
# ========================

@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    # Clustering parameters
    n_clusters_range: Tuple[int, int] = (2, 20)
    clustering_methods: List[str] = None
    
    # Linear probing parameters
    cv_folds: int = 5
    test_size: float = 0.2
    classifiers: List[str] = None
    
    # Visualization parameters
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 300
    save_plots: bool = False
    plot_dir: str = "plots"
    
    # RSA parameters
    distance_metrics: List[str] = None
    
    # Device for computations
    device: str = 'cpu'
    
    def __post_init__(self):
        if self.clustering_methods is None:
            self.clustering_methods = ['kmeans']
        if self.classifiers is None:
            self.classifiers = ['logistic', 'svm']
        if self.distance_metrics is None:
            self.distance_metrics = ['euclidean', 'cosine']

# ========================
# CLUSTERING EVALUATION
# ========================

class ClusteringEvaluator:
    """Evaluate clustering quality of embeddings."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate_clustering(self, 
                          embeddings: np.ndarray, 
                          true_labels: List[str],
                          label_hierarchy: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Comprehensive clustering evaluation.
        
        Args:
            embeddings: Embedding vectors (n_samples, embedding_dim)
            true_labels: Ground truth labels
            label_hierarchy: Optional hierarchical label structure
            
        Returns:
            Dictionary of clustering metrics
        """
        print("🔍 Evaluating clustering quality...")
        
        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(true_labels)
        n_true_clusters = len(np.unique(encoded_labels))
        
        results = {
            'n_samples': len(embeddings),
            'n_true_clusters': n_true_clusters,
            'label_encoder': le,
            'clustering_metrics': {}
        }
        
        # Test different numbers of clusters
        k_range = range(
            max(2, self.config.n_clusters_range[0]),
            min(n_true_clusters * 2, self.config.n_clusters_range[1] + 1)
        )
        
        best_silhouette = -1
        best_k = None
        
        for k in k_range:
            print(f"   Testing k={k}...")
            
            # K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Compute metrics
            metrics = self._compute_clustering_metrics(
                embeddings, encoded_labels, cluster_labels, k
            )
            
            results['clustering_metrics'][k] = metrics
            
            # Track best silhouette score
            if metrics['silhouette_score'] > best_silhouette:
                best_silhouette = metrics['silhouette_score']
                best_k = k
        
        results['best_k'] = best_k
        results['best_silhouette'] = best_silhouette
        
        # Evaluate clustering at true number of clusters
        if n_true_clusters in results['clustering_metrics']:
            results['true_k_metrics'] = results['clustering_metrics'][n_true_clusters]
        
        # Hierarchical clustering evaluation if hierarchy provided
        if label_hierarchy:
            results['hierarchical_metrics'] = self._evaluate_hierarchical_clustering(
                embeddings, true_labels, label_hierarchy
            )
        
        print(f"   Best k: {best_k} (silhouette: {best_silhouette:.3f})")
        
        return results
    
    def _compute_clustering_metrics(self, 
                                  embeddings: np.ndarray,
                                  true_labels: np.ndarray,
                                  cluster_labels: np.ndarray,
                                  k: int) -> Dict[str, float]:
        """Compute clustering quality metrics."""
        metrics = {}
        
        # Silhouette score
        if k > 1 and len(np.unique(cluster_labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(embeddings, cluster_labels)
        else:
            metrics['silhouette_score'] = -1.0
        
        # Calinski-Harabasz score
        if k > 1 and len(np.unique(cluster_labels)) > 1:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, cluster_labels)
        else:
            metrics['calinski_harabasz_score'] = 0.0
        
        # Adjusted Rand Index (comparing to true labels)
        metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, cluster_labels)
        
        # Normalized Mutual Information
        metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, cluster_labels)
        
        # Purity score
        metrics['purity'] = self._compute_purity(true_labels, cluster_labels)
        
        return metrics
    
    def _compute_purity(self, true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Compute purity score."""
        n_correct = 0
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_true_labels = true_labels[cluster_mask]
            if len(cluster_true_labels) > 0:
                most_common = np.bincount(cluster_true_labels).argmax()
                n_correct += np.sum(cluster_true_labels == most_common)
        
        return n_correct / len(true_labels)
    
    def _evaluate_hierarchical_clustering(self, 
                                        embeddings: np.ndarray,
                                        labels: List[str],
                                        hierarchy: Dict[str, List[str]]) -> Dict[str, Any]:
        """Evaluate hierarchical clustering structure."""
        results = {}
        
        # For each level in hierarchy, evaluate clustering
        for level_name, level_labels in hierarchy.items():
            if len(set(level_labels)) > 1:
                le = LabelEncoder()
                encoded = le.fit_transform(level_labels)
                n_clusters = len(np.unique(encoded))
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                pred_labels = kmeans.fit_predict(embeddings)
                
                results[level_name] = {
                    'n_clusters': n_clusters,
                    'adjusted_rand_score': adjusted_rand_score(encoded, pred_labels),
                    'normalized_mutual_info': normalized_mutual_info_score(encoded, pred_labels),
                    'silhouette_score': silhouette_score(embeddings, pred_labels) if n_clusters > 1 else -1
                }
        
        return results
    
    def plot_clustering_metrics(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot clustering evaluation metrics."""
        metrics_data = results['clustering_metrics']
        
        if not metrics_data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize)
        fig.suptitle('Clustering Quality Metrics', fontsize=16)
        
        k_values = sorted(metrics_data.keys())
        
        # Silhouette score
        silhouette_scores = [metrics_data[k]['silhouette_score'] for k in k_values]
        axes[0, 0].plot(k_values, silhouette_scores, 'bo-')
        axes[0, 0].axvline(x=results['best_k'], color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Silhouette Score')
        axes[0, 0].set_xlabel('Number of Clusters (k)')
        axes[0, 0].set_ylabel('Silhouette Score')
        
        # Calinski-Harabasz score
        ch_scores = [metrics_data[k]['calinski_harabasz_score'] for k in k_values]
        axes[0, 1].plot(k_values, ch_scores, 'go-')
        axes[0, 1].set_title('Calinski-Harabasz Score')
        axes[0, 1].set_xlabel('Number of Clusters (k)')
        axes[0, 1].set_ylabel('CH Score')
        
        # Adjusted Rand Index
        ari_scores = [metrics_data[k]['adjusted_rand_score'] for k in k_values]
        axes[1, 0].plot(k_values, ari_scores, 'ro-')
        axes[1, 0].axvline(x=results['n_true_clusters'], color='g', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Adjusted Rand Index')
        axes[1, 0].set_xlabel('Number of Clusters (k)')
        axes[1, 0].set_ylabel('ARI')
        
        # Normalized Mutual Information
        nmi_scores = [metrics_data[k]['normalized_mutual_info'] for k in k_values]
        axes[1, 1].plot(k_values, nmi_scores, 'mo-')
        axes[1, 1].axvline(x=results['n_true_clusters'], color='g', linestyle='--', alpha=0.7)
        axes[1, 1].set_title('Normalized Mutual Information')
        axes[1, 1].set_xlabel('Number of Clusters (k)')
        axes[1, 1].set_ylabel('NMI')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()

# ========================
# LINEAR PROBING EVALUATION
# ========================

class LinearProbingEvaluator:
    """Evaluate embeddings using linear probing."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate_linear_probing(self, 
                              embeddings: np.ndarray,
                              labels: List[str],
                              metadata: List[Dict],
                              label_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive linear probing evaluation.
        
        Args:
            embeddings: Embedding vectors
            labels: Primary labels for classification
            metadata: Metadata for each sample
            label_types: Additional label types to evaluate
            
        Returns:
            Dictionary of linear probing results
        """
        print("🔍 Evaluating linear probing performance...")
        
        results = {
            'primary_labels': self._evaluate_classification_task(embeddings, labels, 'primary')
        }
        
        # Evaluate additional label types if provided
        if label_types and metadata:
            for label_type in label_types:
                if all(label_type in meta for meta in metadata):
                    task_labels = [meta[label_type] for meta in metadata]
                    if len(set(task_labels)) > 1:  # Only if there's variation
                        results[label_type] = self._evaluate_classification_task(
                            embeddings, task_labels, label_type
                        )
        
        # Hierarchical evaluation if metadata contains hierarchy info
        hierarchy_results = self._evaluate_hierarchical_classification(embeddings, metadata)
        if hierarchy_results:
            results['hierarchical'] = hierarchy_results
        
        return results
    
    def _evaluate_classification_task(self, 
                                    embeddings: np.ndarray,
                                    labels: List[str],
                                    task_name: str) -> Dict[str, Any]:
        """Evaluate a single classification task."""
        print(f"   Evaluating {task_name} classification...")
        
        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        n_classes = len(np.unique(encoded_labels))
        
        if n_classes < 2:
            return {'error': 'Not enough classes for classification'}
        
        # Standardize embeddings
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        
        results = {
            'n_classes': n_classes,
            'n_samples': len(embeddings),
            'label_encoder': le,
            'scaler': scaler,
            'classifiers': {}
        }
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        
        # Test different classifiers
        classifiers = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42),
        }
        
        for clf_name in self.config.classifiers:
            if clf_name in classifiers:
                clf = classifiers[clf_name]
                
                # Cross-validation scores
                cv_scores = cross_val_score(
                    clf, scaled_embeddings, encoded_labels, 
                    cv=cv, scoring='accuracy'
                )
                
                # Fit on full data for confusion matrix
                clf.fit(scaled_embeddings, encoded_labels)
                predictions = clf.predict(scaled_embeddings)
                
                # Compute metrics
                accuracy = accuracy_score(encoded_labels, predictions)
                f1 = f1_score(encoded_labels, predictions, average='weighted')
                
                # Confusion matrix
                cm = confusion_matrix(encoded_labels, predictions)
                
                results['classifiers'][clf_name] = {
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'confusion_matrix': cm,
                    'classification_report': classification_report(
                        encoded_labels, predictions, 
                        target_names=le.classes_, 
                        output_dict=True
                    )
                }
        
        # Find best classifier
        best_clf = max(
            results['classifiers'].keys(),
            key=lambda x: results['classifiers'][x]['cv_mean']
        )
        results['best_classifier'] = best_clf
        
        print(f"      Best classifier: {best_clf} "
              f"(CV accuracy: {results['classifiers'][best_clf]['cv_mean']:.3f} ± "
              f"{results['classifiers'][best_clf]['cv_std']:.3f})")
        
        return results
    
    def _evaluate_hierarchical_classification(self, 
                                            embeddings: np.ndarray,
                                            metadata: List[Dict]) -> Optional[Dict[str, Any]]:
        """Evaluate hierarchical classification tasks."""
        # Look for hierarchical structure in metadata
        hierarchy_keys = ['image_type', 'category', 'family']
        available_keys = [key for key in hierarchy_keys if all(key in meta for meta in metadata)]
        
        if len(available_keys) < 2:
            return None
        
        results = {}
        
        for key in available_keys:
            task_labels = [meta[key] for meta in metadata]
            if len(set(task_labels)) > 1:
                results[key] = self._evaluate_classification_task(embeddings, task_labels, key)
        
        return results
    
    def plot_confusion_matrix(self, 
                            results: Dict[str, Any],
                            task_name: str,
                            classifier_name: str,
                            save_path: Optional[str] = None):
        """Plot confusion matrix for a classification task."""
        if task_name not in results or classifier_name not in results[task_name]['classifiers']:
            print(f"Results not found for task '{task_name}' and classifier '{classifier_name}'")
            return
        
        task_results = results[task_name]
        clf_results = task_results['classifiers'][classifier_name]
        
        cm = clf_results['confusion_matrix']
        labels = task_results['label_encoder'].classes_
        
        plt.figure(figsize=self.config.figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix: {task_name} ({classifier_name})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_classification_performance(self, 
                                      results: Dict[str, Any],
                                      save_path: Optional[str] = None):
        """Plot classification performance across tasks."""
        tasks = [key for key in results.keys() if key not in ['hierarchical']]
        
        if not tasks:
            return
        
        # Collect performance data
        performance_data = []
        
        for task in tasks:
            if 'classifiers' in results[task]:
                for clf_name, clf_results in results[task]['classifiers'].items():
                    performance_data.append({
                        'Task': task,
                        'Classifier': clf_name,
                        'CV_Accuracy': clf_results['cv_mean'],
                        'CV_Std': clf_results['cv_std'],
                        'Accuracy': clf_results['accuracy'],
                        'F1_Score': clf_results['f1_score']
                    })
        
        if not performance_data:
            return
        
        df = pd.DataFrame(performance_data)
        
        fig, axes = plt.subplots(1, 2, figsize=self.config.figsize)
        
        # CV Accuracy
        sns.barplot(data=df, x='Task', y='CV_Accuracy', hue='Classifier', ax=axes[0])
        axes[0].set_title('Cross-Validation Accuracy')
        axes[0].set_ylabel('CV Accuracy')
        
        # F1 Score
        sns.barplot(data=df, x='Task', y='F1_Score', hue='Classifier', ax=axes[1])
        axes[1].set_title('F1 Score')
        axes[1].set_ylabel('F1 Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()

# ========================
# RSA EVALUATION
# ========================

class RSAEvaluator:
    """Representational Similarity Analysis."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def compute_rsa(self, 
                   embeddings: np.ndarray,
                   labels: List[str],
                   metadata: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Compute representational similarity analysis.
        
        Args:
            embeddings: Embedding vectors
            labels: Labels for each sample
            metadata: Optional metadata for additional analysis
            
        Returns:
            RSA results dictionary
        """
        print("🔍 Computing Representational Similarity Analysis...")
        
        results = {
            'distance_matrices': {},
            'similarity_matrices': {},
            'label_similarities': {},
            'mds_coordinates': {}
        }
        
        # Compute distance matrices for different metrics
        for metric in self.config.distance_metrics:
            print(f"   Computing {metric} distances...")
            
            if metric == 'cosine':
                # Cosine distance
                normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
                distance_matrix = 1 - similarity_matrix
            else:
                # Euclidean and other metrics
                distances = pdist(embeddings, metric=metric)
                distance_matrix = squareform(distances)
                similarity_matrix = 1 / (1 + distance_matrix)  # Convert to similarity
            
            results['distance_matrices'][metric] = distance_matrix
            results['similarity_matrices'][metric] = similarity_matrix
            
            # MDS coordinates for visualization
            if distance_matrix.shape[0] <= 1000:  # Only for reasonable sizes
                mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
                mds_coords = mds.fit_transform(distance_matrix)
                results['mds_coordinates'][metric] = mds_coords
        
        # Compute label-based similarities
        results['label_similarities'] = self._compute_label_similarities(labels, metadata)
        
        # Compare embedding similarities with label similarities
        if results['label_similarities']:
            results['similarity_correlations'] = self._compute_similarity_correlations(
                results['similarity_matrices'], results['label_similarities']
            )
        
        return results
    
    def _compute_label_similarities(self, 
                                  labels: List[str],
                                  metadata: Optional[List[Dict]] = None) -> Dict[str, np.ndarray]:
        """Compute similarity matrices based on labels."""
        n_samples = len(labels)
        similarities = {}
        
        # Binary similarity for primary labels
        label_sim = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                label_sim[i, j] = 1.0 if labels[i] == labels[j] else 0.0
        
        similarities['primary_labels'] = label_sim
        
        # Additional similarities from metadata
        if metadata:
            for key in ['image_type', 'category', 'family']:
                if all(key in meta for meta in metadata):
                    values = [meta[key] for meta in metadata]
                    sim_matrix = np.zeros((n_samples, n_samples))
                    
                    for i in range(n_samples):
                        for j in range(n_samples):
                            sim_matrix[i, j] = 1.0 if values[i] == values[j] else 0.0
                    
                    similarities[key] = sim_matrix
        
        return similarities
    
    def _compute_similarity_correlations(self, 
                                       embedding_similarities: Dict[str, np.ndarray],
                                       label_similarities: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Compute correlations between embedding and label similarities."""
        correlations = {}
        
        for emb_metric, emb_sim in embedding_similarities.items():
            correlations[emb_metric] = {}
            
            # Get upper triangle (exclude diagonal)
            triu_indices = np.triu_indices_from(emb_sim, k=1)
            emb_sim_vec = emb_sim[triu_indices]
            
            for label_type, label_sim in label_similarities.items():
                label_sim_vec = label_sim[triu_indices]
                
                # Compute correlation
                corr, p_value = spearmanr(emb_sim_vec, label_sim_vec)
                correlations[emb_metric][label_type] = {
                    'correlation': corr,
                    'p_value': p_value
                }
        
        return correlations
    
    def plot_similarity_matrices(self, 
                                results: Dict[str, Any],
                                save_path: Optional[str] = None):
        """Plot similarity matrices."""
        similarity_matrices = results['similarity_matrices']
        
        n_metrics = len(similarity_matrices)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, (metric, sim_matrix) in enumerate(similarity_matrices.items()):
            im = axes[idx].imshow(sim_matrix, cmap='RdYlBu_r', aspect='auto')
            axes[idx].set_title(f'{metric.capitalize()} Similarity')
            plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_mds_projection(self, 
                          results: Dict[str, Any],
                          labels: List[str],
                          save_path: Optional[str] = None):
        """Plot MDS projection of embeddings."""
        mds_coords = results['mds_coordinates']
        
        if not mds_coords:
            print("No MDS coordinates available for plotting")
            return
        
        n_metrics = len(mds_coords)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        # Create color map for labels
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        label_to_color = dict(zip(unique_labels, colors))
        
        for idx, (metric, coords) in enumerate(mds_coords.items()):
            for label in unique_labels:
                mask = np.array(labels) == label
                if np.any(mask):
                    axes[idx].scatter(coords[mask, 0], coords[mask, 1], 
                                    c=[label_to_color[label]], label=label, alpha=0.7)
            
            axes[idx].set_title(f'MDS Projection ({metric})')
            axes[idx].set_xlabel('MDS Component 1')
            axes[idx].set_ylabel('MDS Component 2')
            if len(unique_labels) <= 20:  # Only show legend if not too many labels
                axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()

# ========================
# VISUALIZATION TOOLS
# ========================

class EmbeddingVisualizer:
    """Visualization tools for embeddings."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def plot_embedding_space(self, 
                           embeddings: np.ndarray,
                           labels: List[str],
                           method: str = 'tsne',
                           save_path: Optional[str] = None):
        """Plot 2D projection of embedding space."""
        print(f"🎨 Creating {method.upper()} visualization...")
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        coords_2d = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=self.config.figsize)
        
        # Create color map
        unique_labels = list(set(labels))
        if len(unique_labels) <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_labels), 20)))
        
        label_to_color = dict(zip(unique_labels, colors))
        
        # Plot points
        for label in unique_labels:
            mask = np.array(labels) == label
            if np.any(mask):
                plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                          c=[label_to_color[label]], label=label, alpha=0.7, s=50)
        
        plt.title(f'{method.upper()} Projection of Embedding Space')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        
        if len(unique_labels) <= 15:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
        
        return coords_2d
    
    def plot_hierarchical_embedding(self, 
                                   embeddings: np.ndarray,
                                   metadata: List[Dict],
                                   hierarchy_keys: List[str] = None,
                                   save_path: Optional[str] = None):
        """Plot embeddings with hierarchical coloring."""
        if hierarchy_keys is None:
            hierarchy_keys = ['image_type', 'category']
        
        # Filter available keys
        available_keys = [key for key in hierarchy_keys if all(key in meta for meta in metadata)]
        
        if not available_keys:
            print("No hierarchical keys found in metadata")
            return
        
        # Use t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
        coords_2d = tsne.fit_transform(embeddings)
        
        n_levels = len(available_keys)
        fig, axes = plt.subplots(1, n_levels, figsize=(6 * n_levels, 5))
        
        if n_levels == 1:
            axes = [axes]
        
        for idx, key in enumerate(available_keys):
            values = [meta[key] for meta in metadata]
            unique_values = list(set(values))
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))
            value_to_color = dict(zip(unique_values, colors))
            
            for value in unique_values:
                mask = np.array(values) == value
                if np.any(mask):
                    axes[idx].scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                                    c=[value_to_color[value]], label=value, alpha=0.7)
            
            axes[idx].set_title(f'Embeddings by {key}')
            axes[idx].set_xlabel('t-SNE Component 1')
            axes[idx].set_ylabel('t-SNE Component 2')
            
            if len(unique_values) <= 10:
                axes[idx].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()

# ========================
# EMBEDDING NORMALIZATION
# ========================

class EmbeddingNormalizer:
    """Comprehensive embedding normalization for cross-region comparison."""
    
    def __init__(self):
        self.normalization_params = {}
    
    def normalize_embeddings(self, 
                           embeddings: np.ndarray,
                           method: str = 'unit_sphere',
                           fit_params: bool = True,
                           region_name: str = 'default') -> np.ndarray:
        """
        Normalize embeddings using specified method.
        
        Args:
            embeddings: Input embeddings (n_samples, embedding_dim)
            method: Normalization method ('unit_sphere', 'standardize', 'center', 'robust')
            fit_params: Whether to fit normalization parameters (True) or use existing ones (False)
            region_name: Name of brain region for parameter storage
            
        Returns:
            Normalized embeddings
        """
        if method == 'unit_sphere':
            return self._unit_sphere_normalize(embeddings)
        
        elif method == 'standardize':
            return self._standardize_normalize(embeddings, fit_params, region_name)
        
        elif method == 'center':
            return self._center_normalize(embeddings, fit_params, region_name)
        
        elif method == 'robust':
            return self._robust_normalize(embeddings, fit_params, region_name)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def _unit_sphere_normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize each embedding vector to unit sphere."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    def _standardize_normalize(self, 
                             embeddings: np.ndarray, 
                             fit_params: bool, 
                             region_name: str) -> np.ndarray:
        """Z-score normalize each embedding dimension."""
        if fit_params:
            mean = np.mean(embeddings, axis=0)
            std = np.std(embeddings, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            
            self.normalization_params[region_name] = {
                'method': 'standardize',
                'mean': mean,
                'std': std
            }
        else:
            if region_name not in self.normalization_params:
                raise ValueError(f"No normalization parameters found for region {region_name}")
            
            params = self.normalization_params[region_name]
            mean = params['mean']
            std = params['std']
        
        return (embeddings - mean) / std
    
    def _center_normalize(self, 
                        embeddings: np.ndarray, 
                        fit_params: bool, 
                        region_name: str) -> np.ndarray:
        """Center embeddings around origin."""
        if fit_params:
            mean = np.mean(embeddings, axis=0)
            self.normalization_params[region_name] = {
                'method': 'center',
                'mean': mean
            }
        else:
            if region_name not in self.normalization_params:
                raise ValueError(f"No normalization parameters found for region {region_name}")
            
            mean = self.normalization_params[region_name]['mean']
        
        return embeddings - mean
    
    def _robust_normalize(self, 
                        embeddings: np.ndarray, 
                        fit_params: bool, 
                        region_name: str) -> np.ndarray:
        """Robust normalization using median and MAD."""
        if fit_params:
            median = np.median(embeddings, axis=0)
            mad = np.median(np.abs(embeddings - median), axis=0)
            mad[mad == 0] = 1  # Avoid division by zero
            
            self.normalization_params[region_name] = {
                'method': 'robust',
                'median': median,
                'mad': mad
            }
        else:
            if region_name not in self.normalization_params:
                raise ValueError(f"No normalization parameters found for region {region_name}")
            
            params = self.normalization_params[region_name]
            median = params['median']
            mad = params['mad']
        
        return (embeddings - median) / mad
    
    def procrustes_align(self, 
                        embeddings_source: np.ndarray,
                        embeddings_target: np.ndarray,
                        scaling: bool = True,
                        reflection: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Align source embeddings to target embeddings using Procrustes analysis.
        
        Args:
            embeddings_source: Source embeddings to be aligned
            embeddings_target: Target embeddings (reference)
            scaling: Whether to allow scaling
            reflection: Whether to allow reflection
            
        Returns:
            aligned_source: Aligned source embeddings
            aligned_target: Aligned target embeddings  
            alignment_info: Dictionary with alignment parameters
        """
        from scipy.spatial.distance import procrustes
        
        # Perform Procrustes analysis
        aligned_target, aligned_source, disparity = procrustes(
            embeddings_target, embeddings_source
        )
        
        alignment_info = {
            'disparity': disparity,
            'source_shape': embeddings_source.shape,
            'target_shape': embeddings_target.shape,
            'scaling_used': scaling,
            'reflection_used': reflection
        }
        
        return aligned_source, aligned_target, alignment_info
    
    def compare_embedding_spaces(self, 
                               embeddings_dict: Dict[str, np.ndarray],
                               normalization_method: str = 'unit_sphere',
                               align_to_reference: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare multiple embedding spaces with proper normalization.
        
        Args:
            embeddings_dict: Dictionary mapping region names to embeddings
            normalization_method: Method for normalization
            align_to_reference: Region name to use as reference for alignment (optional)
            
        Returns:
            Dictionary with normalized embeddings and comparison metrics
        """
        print(f"🔄 Comparing embedding spaces with {normalization_method} normalization...")
        
        results = {
            'normalized_embeddings': {},
            'alignment_info': {},
            'comparison_metrics': {}
        }
        
        # Step 1: Normalize all embeddings
        for region_name, embeddings in embeddings_dict.items():
            print(f"   Normalizing {region_name} embeddings...")
            normalized = self.normalize_embeddings(
                embeddings, 
                method=normalization_method,
                fit_params=True,
                region_name=region_name
            )
            results['normalized_embeddings'][region_name] = normalized
        
        # Step 2: Optional Procrustes alignment
        if align_to_reference and align_to_reference in embeddings_dict:
            print(f"   Aligning all embeddings to {align_to_reference}...")
            reference_embeddings = results['normalized_embeddings'][align_to_reference]
            
            for region_name, normalized_embeddings in results['normalized_embeddings'].items():
                if region_name != align_to_reference:
                    aligned_source, aligned_target, alignment_info = self.procrustes_align(
                        normalized_embeddings, reference_embeddings
                    )
                    
                    results['normalized_embeddings'][region_name] = aligned_source
                    results['alignment_info'][region_name] = alignment_info
        
        # Step 3: Compute comparison metrics
        region_names = list(embeddings_dict.keys())
        for i, region1 in enumerate(region_names):
            for j, region2 in enumerate(region_names):
                if i < j:  # Only compute upper triangle
                    emb1 = results['normalized_embeddings'][region1]
                    emb2 = results['normalized_embeddings'][region2]
                    
                    # Compute similarity metrics
                    metrics = self._compute_embedding_similarity(emb1, emb2)
                    results['comparison_metrics'][f"{region1}_vs_{region2}"] = metrics
        
        return results
    
    def _compute_embedding_similarity(self, 
                                    embeddings1: np.ndarray,
                                    embeddings2: np.ndarray) -> Dict[str, float]:
        """Compute similarity metrics between two embedding spaces."""
        # Ensure same number of samples
        min_samples = min(len(embeddings1), len(embeddings2))
        emb1 = embeddings1[:min_samples]
        emb2 = embeddings2[:min_samples]
        
        metrics = {}
        
        # Canonical Correlation Analysis (CCA)
        try:
            from sklearn.cross_decomposition import CCA
            cca = CCA(n_components=min(emb1.shape[1], emb2.shape[1], 10))
            cca.fit(emb1, emb2)
            
            # Transform both sets
            emb1_cca, emb2_cca = cca.transform(emb1, emb2)
            
            # Compute canonical correlations
            correlations = []
            for i in range(emb1_cca.shape[1]):
                corr = np.corrcoef(emb1_cca[:, i], emb2_cca[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            metrics['mean_canonical_correlation'] = np.mean(correlations) if correlations else 0.0
            metrics['max_canonical_correlation'] = np.max(correlations) if correlations else 0.0
            
        except Exception as e:
            metrics['mean_canonical_correlation'] = 0.0
            metrics['max_canonical_correlation'] = 0.0
        
        # Centered Kernel Alignment (CKA)
        metrics['linear_cka'] = self._compute_linear_cka(emb1, emb2)
        
        # Frobenius norm distance
        if emb1.shape == emb2.shape:
            metrics['frobenius_distance'] = np.linalg.norm(emb1 - emb2, 'fro')
        else:
            metrics['frobenius_distance'] = np.inf
        
        # Cosine similarity of means
        mean1 = np.mean(emb1, axis=0)
        mean2 = np.mean(emb2, axis=0)
        metrics['mean_cosine_similarity'] = np.dot(mean1, mean2) / (np.linalg.norm(mean1) * np.linalg.norm(mean2))
        
        return metrics
    
    def _compute_linear_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Linear Centered Kernel Alignment (CKA)."""
        def centering(K):
            n = K.shape[0]
            unit = np.ones([n, n])
            I = np.eye(n)
            H = I - unit / n
            return np.dot(np.dot(H, K), H)
        
        def linear_HSIC(X, Y):
            L_X = np.dot(X, X.T)
            L_Y = np.dot(Y, Y.T)
            return np.trace(np.dot(centering(L_X), centering(L_Y)))
        
        def linear_CKA(X, Y):
            hsic = linear_HSIC(X, Y)
            var1 = np.sqrt(linear_HSIC(X, X))
            var2 = np.sqrt(linear_HSIC(Y, Y))
            
            return hsic / (var1 * var2) if var1 * var2 > 0 else 0.0
        
        return linear_CKA(X, Y)
    
    def plot_embedding_comparison(self, 
                                comparison_results: Dict[str, Any],
                                save_path: Optional[str] = None):
        """Plot embedding space comparison results."""
        normalized_embeddings = comparison_results['normalized_embeddings']
        comparison_metrics = comparison_results['comparison_metrics']
        
        n_regions = len(normalized_embeddings)
        region_names = list(normalized_embeddings.keys())
        
        # Create subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Embedding spaces (t-SNE)
        for i, (region_name, embeddings) in enumerate(normalized_embeddings.items()):
            plt.subplot(2, n_regions, i + 1)
            
            # Use t-SNE for visualization (sample if too large)
            n_samples = min(1000, len(embeddings))
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            sample_embeddings = embeddings[indices]
            
            if sample_embeddings.shape[1] > 2:
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples//4))
                coords_2d = tsne.fit_transform(sample_embeddings)
            else:
                coords_2d = sample_embeddings
            
            plt.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.6, s=20)
            plt.title(f'{region_name} Embeddings')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
        
        # Plot 2: Comparison metrics heatmap
        plt.subplot(2, 1, 2)
        
        if comparison_metrics:
            # Create similarity matrix
            similarity_matrix = np.zeros((n_regions, n_regions))
            similarity_matrix.fill(np.nan)
            
            # Fill diagonal with 1s
            np.fill_diagonal(similarity_matrix, 1.0)
            
            # Fill upper triangle with comparison metrics
            for comparison, metrics in comparison_metrics.items():
                region1, region2 = comparison.split('_vs_')
                i = region_names.index(region1)
                j = region_names.index(region2)
                
                # Use mean canonical correlation as the similarity metric
                similarity = metrics.get('mean_canonical_correlation', 0.0)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Make symmetric
            
            # Plot heatmap
            im = plt.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
            plt.colorbar(im, label='Mean Canonical Correlation')
            plt.xticks(range(n_regions), region_names, rotation=45)
            plt.yticks(range(n_regions), region_names)
            plt.title('Cross-Region Embedding Similarity')
            
            # Add text annotations
            for i in range(n_regions):
                for j in range(n_regions):
                    if not np.isnan(similarity_matrix[i, j]):
                        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                                      ha="center", va="center", color="black")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_normalization_summary(self) -> pd.DataFrame:
        """Get summary of normalization parameters across regions."""
        summary_data = []
        
        for region_name, params in self.normalization_params.items():
            summary_data.append({
                'Region': region_name,
                'Method': params['method'],
                'Parameters': list(params.keys())
            })
        
        return pd.DataFrame(summary_data)

# ========================
# COMPREHENSIVE EVALUATOR (UPDATED)
# ========================

class NeuralEmbeddingEvaluator:
    """Comprehensive evaluator for neural embeddings with cross-region comparison."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        
        self.clustering_evaluator = ClusteringEvaluator(self.config)
        self.linear_probing_evaluator = LinearProbingEvaluator(self.config)
        self.rsa_evaluator = RSAEvaluator(self.config)
        self.visualizer = EmbeddingVisualizer(self.config)
        self.normalizer = EmbeddingNormalizer()  # NEW: Add normalizer
    
    def evaluate_all(self, 
                    embeddings: np.ndarray,
                    categories: List[str],
                    metadata: List[Dict],
                    save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of embeddings.
        
        Args:
            embeddings: Embedding vectors (n_samples, embedding_dim)
            categories: Primary category labels
            metadata: Metadata for each sample
            save_dir: Directory to save plots (optional)
            
        Returns:
            Comprehensive evaluation results
        """
        print("🚀 Running comprehensive embedding evaluation...")
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Unique categories: {len(set(categories))}")
        print(f"   Total samples: {len(embeddings)}")
        
        results = {
            'summary': {
                'n_samples': len(embeddings),
                'embedding_dim': embeddings.shape[1],
                'n_categories': len(set(categories)),
                'categories': list(set(categories))
            }
        }
        
        # 1. Clustering evaluation
        print("\n" + "="*50)
        results['clustering'] = self.clustering_evaluator.evaluate_clustering(
            embeddings, categories
        )
        
        # 2. Linear probing evaluation
        print("\n" + "="*50)
        results['linear_probing'] = self.linear_probing_evaluator.evaluate_linear_probing(
            embeddings, categories, metadata, 
            label_types=['image_type', 'category'] if metadata else None
        )
        
        # 3. RSA evaluation
        print("\n" + "="*50)
        results['rsa'] = self.rsa_evaluator.compute_rsa(embeddings, categories, metadata)
        
        # 4. Generate visualizations
        print("\n" + "="*50)
        print("🎨 Generating visualizations...")
        
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            # Clustering plots
            self.clustering_evaluator.plot_clustering_metrics(
                results['clustering'], 
                save_path=os.path.join(save_dir, 'clustering_metrics.png')
            )
            
            # Linear probing plots
            self.linear_probing_evaluator.plot_classification_performance(
                results['linear_probing'],
                save_path=os.path.join(save_dir, 'classification_performance.png')
            )
            
            # RSA plots
            self.rsa_evaluator.plot_similarity_matrices(
                results['rsa'],
                save_path=os.path.join(save_dir, 'similarity_matrices.png')
            )
            
            self.rsa_evaluator.plot_mds_projection(
                results['rsa'], categories,
                save_path=os.path.join(save_dir, 'mds_projection.png')
            )
            
            # Embedding visualizations
            self.visualizer.plot_embedding_space(
                embeddings, categories, method='tsne',
                save_path=os.path.join(save_dir, 'tsne_embedding.png')
            )
            
            if metadata:
                self.visualizer.plot_hierarchical_embedding(
                    embeddings, metadata,
                    save_path=os.path.join(save_dir, 'hierarchical_embedding.png')
                )
        
        # 5. Summary report
        results['evaluation_summary'] = self._generate_summary_report(results)
        
        print("\n" + "="*50)
        print("✅ Comprehensive evaluation completed!")
        
        return results
    
    def evaluate_cross_region(self, 
                            embeddings_dict: Dict[str, np.ndarray],
                            categories_dict: Dict[str, List[str]],
                            metadata_dict: Dict[str, List[Dict]],
                            normalization_method: str = 'unit_sphere',
                            align_to_reference: Optional[str] = None,
                            save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive cross-region comparison of embeddings.
        
        Args:
            embeddings_dict: Dictionary mapping region names to embeddings
            categories_dict: Dictionary mapping region names to categories
            metadata_dict: Dictionary mapping region names to metadata
            normalization_method: Method for embedding normalization
            align_to_reference: Region to use as reference for alignment
            save_dir: Directory to save plots
            
        Returns:
            Cross-region comparison results
        """
        print("🔬 Running cross-region embedding comparison...")
        print(f"   Regions: {list(embeddings_dict.keys())}")
        print(f"   Normalization method: {normalization_method}")
        
        # Validate inputs
        regions = list(embeddings_dict.keys())
        if not all(region in categories_dict for region in regions):
            raise ValueError("Missing categories for some regions")
        if not all(region in metadata_dict for region in regions):
            raise ValueError("Missing metadata for some regions")
        
        results = {
            'regions': regions,
            'normalization_method': normalization_method,
            'region_evaluations': {},
            'cross_region_comparison': {},
            'summary_comparison': {}
        }
        
        # Step 1: Normalize and align embeddings
        print("\n🔄 Normalizing and aligning embeddings...")
        comparison_results = self.normalizer.compare_embedding_spaces(
            embeddings_dict, 
            normalization_method=normalization_method,
            align_to_reference=align_to_reference
        )
        
        results['cross_region_comparison'] = comparison_results
        
        # Step 2: Evaluate each region individually (using normalized embeddings)
        normalized_embeddings = comparison_results['normalized_embeddings']
        
        for region in regions:
            print(f"\n📊 Evaluating {region}...")
            
            region_results = self.evaluate_all(
                normalized_embeddings[region],
                categories_dict[region],
                metadata_dict[region],
                save_dir=os.path.join(save_dir, region) if save_dir else None
            )
            
            results['region_evaluations'][region] = region_results
        
        # Step 3: Generate cross-region comparison metrics
        print("\n📊 Computing cross-region comparison metrics...")
        results['summary_comparison'] = self._generate_cross_region_summary(results)
        
        # Step 4: Generate cross-region visualizations
        if save_dir:
            print("\n🎨 Generating cross-region visualizations...")
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            # Plot embedding comparison
            self.normalizer.plot_embedding_comparison(
                comparison_results,
                save_path=os.path.join(save_dir, 'cross_region_comparison.png')
            )
            
            # Plot cross-region metrics comparison
            self._plot_cross_region_metrics(
                results,
                save_path=os.path.join(save_dir, 'cross_region_metrics.png')
            )
        
        print("\n" + "="*60)
        print("✅ Cross-region comparison completed!")
        
        return results
    
    def _generate_cross_region_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of cross-region comparison."""
        regions = results['regions']
        region_evaluations = results['region_evaluations']
        comparison_metrics = results['cross_region_comparison']['comparison_metrics']
        
        summary = {
            'clustering_comparison': {},
            'linear_probing_comparison': {},
            'embedding_similarity': {},
            'ranking': {}
        }
        
        # Clustering comparison
        for region in regions:
            if 'clustering' in region_evaluations[region]:
                clustering = region_evaluations[region]['clustering']
                summary['clustering_comparison'][region] = {
                    'best_silhouette': clustering.get('best_silhouette', 0),
                    'best_k': clustering.get('best_k', 0),
                    'n_true_clusters': clustering.get('n_true_clusters', 0)
                }
                
                if 'true_k_metrics' in clustering:
                    true_k = clustering['true_k_metrics']
                    summary['clustering_comparison'][region].update({
                        'ari_true_k': true_k.get('adjusted_rand_score', 0),
                        'nmi_true_k': true_k.get('normalized_mutual_info', 0)
                    })
        
        # Linear probing comparison
        for region in regions:
            if 'linear_probing' in region_evaluations[region]:
                lp = region_evaluations[region]['linear_probing']
                summary['linear_probing_comparison'][region] = {}
                
                for task, task_results in lp.items():
                    if 'best_classifier' in task_results:
                        best_clf = task_results['best_classifier']
                        best_results = task_results['classifiers'][best_clf]
                        summary['linear_probing_comparison'][region][task] = {
                            'cv_accuracy': best_results['cv_mean'],
                            'f1_score': best_results['f1_score']
                        }
        
        # Embedding similarity
        summary['embedding_similarity'] = comparison_metrics
        
        # Generate rankings
        summary['ranking'] = self._generate_region_rankings(summary)
        
        return summary
    
    def _generate_region_rankings(self, summary: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate rankings of regions by different metrics."""
        regions = list(summary['clustering_comparison'].keys())
        rankings = {}
        
        # Rank by clustering performance (silhouette score)
        silhouette_scores = {
            region: summary['clustering_comparison'][region].get('best_silhouette', 0)
            for region in regions
        }
        rankings['best_clustering'] = sorted(regions, key=lambda x: silhouette_scores[x], reverse=True)
        
        # Rank by linear probing performance (primary task accuracy)
        if 'primary_labels' in summary['linear_probing_comparison'].get(regions[0], {}):
            accuracy_scores = {
                region: summary['linear_probing_comparison'][region]['primary_labels'].get('cv_accuracy', 0)
                for region in regions
                if 'primary_labels' in summary['linear_probing_comparison'].get(region, {})
            }
            if accuracy_scores:
                rankings['best_linear_probing'] = sorted(regions, key=lambda x: accuracy_scores.get(x, 0), reverse=True)
        
        return rankings
    
    def _plot_cross_region_metrics(self, 
                                 results: Dict[str, Any],
                                 save_path: Optional[str] = None):
        """Plot cross-region metrics comparison."""
        regions = results['regions']
        summary = results['summary_comparison']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cross-Region Performance Comparison', fontsize=16)
        
        # Plot 1: Clustering performance
        clustering_data = summary['clustering_comparison']
        if clustering_data:
            silhouette_scores = [clustering_data[region].get('best_silhouette', 0) for region in regions]
            
            axes[0, 0].bar(regions, silhouette_scores, alpha=0.7)
            axes[0, 0].set_title('Clustering Quality (Silhouette Score)')
            axes[0, 0].set_ylabel('Silhouette Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Linear probing performance
        lp_data = summary['linear_probing_comparison']
        if lp_data and 'primary_labels' in lp_data.get(regions[0], {}):
            accuracy_scores = [
                lp_data[region]['primary_labels'].get('cv_accuracy', 0) 
                for region in regions
                if 'primary_labels' in lp_data.get(region, {})
            ]
            
            axes[0, 1].bar(regions[:len(accuracy_scores)], accuracy_scores, alpha=0.7, color='orange')
            axes[0, 1].set_title('Linear Probing Accuracy')
            axes[0, 1].set_ylabel('CV Accuracy')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Embedding similarity heatmap
        similarity_data = summary['embedding_similarity']
        if similarity_data:
            n_regions = len(regions)
            similarity_matrix = np.zeros((n_regions, n_regions))
            np.fill_diagonal(similarity_matrix, 1.0)
            
            for comparison, metrics in similarity_data.items():
                if '_vs_' in comparison:
                    region1, region2 = comparison.split('_vs_')
                    if region1 in regions and region2 in regions:
                        i = regions.index(region1)
                        j = regions.index(region2)
                        similarity = metrics.get('mean_canonical_correlation', 0.0)
                        similarity_matrix[i, j] = similarity
                        similarity_matrix[j, i] = similarity
            
            im = axes[1, 0].imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
            axes[1, 0].set_xticks(range(n_regions))
            axes[1, 0].set_xticklabels(regions, rotation=45)
            axes[1, 0].set_yticks(range(n_regions))
            axes[1, 0].set_yticklabels(regions)
            axes[1, 0].set_title('Cross-Region Similarity')
            plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 4: Rankings
        if 'ranking' in summary:
            rankings = summary['ranking']
            y_pos = np.arange(len(regions))
            
            # Use clustering ranking as example
            if 'best_clustering' in rankings:
                ranked_regions = rankings['best_clustering']
                rank_positions = [ranked_regions.index(region) + 1 for region in regions]
                
                axes[1, 1].barh(y_pos, rank_positions, alpha=0.7, color='green')
                axes[1, 1].set_yticks(y_pos)
                axes[1, 1].set_yticklabels(regions)
                axes[1, 1].set_xlabel('Rank (1 = Best)')
                axes[1, 1].set_title('Clustering Performance Ranking')
                axes[1, 1].invert_xaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_cross_region_summary(self, results: Dict[str, Any]):
        """Print formatted cross-region comparison summary."""
        print("\n" + "="*70)
        print("🧠 CROSS-REGION COMPARISON SUMMARY")
        print("="*70)
        
        regions = results['regions']
        summary = results['summary_comparison']
        
        print(f"📋 Regions Compared: {', '.join(regions)}")
        print(f"🔄 Normalization Method: {results['normalization_method']}")
        
        # Clustering comparison
        if 'clustering_comparison' in summary:
            print(f"\n🎯 Clustering Performance:")
            clustering = summary['clustering_comparison']
            
            for region in regions:
                if region in clustering:
                    data = clustering[region]
                    print(f"   {region}:")
                    print(f"     Silhouette Score: {data.get('best_silhouette', 0):.3f}")
                    print(f"     Best k: {data.get('best_k', 'N/A')}")
                    if 'ari_true_k' in data:
                        print(f"     ARI (true k): {data['ari_true_k']:.3f}")
                        print(f"     NMI (true k): {data['nmi_true_k']:.3f}")
        
        # Linear probing comparison
        if 'linear_probing_comparison' in summary:
            print(f"\n🎯 Linear Probing Performance:")
            lp = summary['linear_probing_comparison']
            
            for region in regions:
                if region in lp and 'primary_labels' in lp[region]:
                    data = lp[region]['primary_labels']
                    print(f"   {region}:")
                    print(f"     CV Accuracy: {data.get('cv_accuracy', 0):.3f}")
                    print(f"     F1 Score: {data.get('f1_score', 0):.3f}")
        
        # Embedding similarity
        if 'embedding_similarity' in summary:
            print(f"\n🎯 Cross-Region Similarity:")
            similarity = summary['embedding_similarity']
            
            for comparison, metrics in similarity.items():
                if '_vs_' in comparison:
                    print(f"   {comparison.replace('_vs_', ' ↔ ')}:")
                    print(f"     Canonical Correlation: {metrics.get('mean_canonical_correlation', 0):.3f}")
                    print(f"     Linear CKA: {metrics.get('linear_cka', 0):.3f}")
        
        # Rankings
        if 'ranking' in summary:
            rankings = summary['ranking']
            print(f"\n🏆 Performance Rankings:")
            
            if 'best_clustering' in rankings:
                print(f"   Clustering: {' > '.join(rankings['best_clustering'])}")
            
            if 'best_linear_probing' in rankings:
                print(f"   Linear Probing: {' > '.join(rankings['best_linear_probing'])}")
        
        print("\n" + "="*70)
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary report of evaluation results."""
        summary = {}
        
        # Clustering summary
        if 'clustering' in results:
            clustering = results['clustering']
            summary['clustering'] = {
                'best_silhouette_score': clustering.get('best_silhouette', 0),
                'best_k': clustering.get('best_k', 0),
                'n_true_clusters': clustering.get('n_true_clusters', 0)
            }
            
            if 'true_k_metrics' in clustering:
                true_k = clustering['true_k_metrics']
                summary['clustering']['true_k_performance'] = {
                    'adjusted_rand_score': true_k.get('adjusted_rand_score', 0),
                    'normalized_mutual_info': true_k.get('normalized_mutual_info', 0)
                }
        
        # Linear probing summary
        if 'linear_probing' in results:
            lp = results['linear_probing']
            summary['linear_probing'] = {}
            
            for task, task_results in lp.items():
                if 'best_classifier' in task_results:
                    best_clf = task_results['best_classifier']
                    best_results = task_results['classifiers'][best_clf]
                    summary['linear_probing'][task] = {
                        'best_classifier': best_clf,
                        'cv_accuracy': best_results['cv_mean'],
                        'cv_std': best_results['cv_std'],
                        'f1_score': best_results['f1_score']
                    }
        
        # RSA summary
        if 'rsa' in results:
            rsa = results['rsa']
            if 'similarity_correlations' in rsa:
                summary['rsa'] = {}
                for metric, correlations in rsa['similarity_correlations'].items():
                    summary['rsa'][metric] = {
                        label_type: corr_data['correlation'] 
                        for label_type, corr_data in correlations.items()
                    }
        
        return summary
    
    def print_summary_report(self, results: Dict[str, Any]):
        """Print a formatted summary report."""
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY REPORT")
        print("="*60)
        
        summary = results.get('evaluation_summary', {})
        
        # Basic info
        basic = results.get('summary', {})
        print(f"📋 Dataset Info:")
        print(f"   Samples: {basic.get('n_samples', 'N/A')}")
        print(f"   Embedding Dimension: {basic.get('embedding_dim', 'N/A')}")
        print(f"   Categories: {basic.get('n_categories', 'N/A')}")
        
        # Clustering results
        if 'clustering' in summary:
            clustering = summary['clustering']
            print(f"\n🎯 Clustering Performance:")
            print(f"   Best Silhouette Score: {clustering.get('best_silhouette_score', 0):.3f}")
            print(f"   Optimal k: {clustering.get('best_k', 'N/A')}")
            print(f"   True k: {clustering.get('n_true_clusters', 'N/A')}")
            
            if 'true_k_performance' in clustering:
                true_k = clustering['true_k_performance']
                print(f"   ARI (true k): {true_k.get('adjusted_rand_score', 0):.3f}")
                print(f"   NMI (true k): {true_k.get('normalized_mutual_info', 0):.3f}")
        
        # Linear probing results
        if 'linear_probing' in summary:
            lp = summary['linear_probing']
            print(f"\n🎯 Linear Probing Performance:")
            for task, task_summary in lp.items():
                print(f"   {task.capitalize()}:")
                print(f"     Best Classifier: {task_summary.get('best_classifier', 'N/A')}")
                print(f"     CV Accuracy: {task_summary.get('cv_accuracy', 0):.3f} ± {task_summary.get('cv_std', 0):.3f}")
                print(f"     F1 Score: {task_summary.get('f1_score', 0):.3f}")
        
        # RSA results
        if 'rsa' in summary:
            rsa = summary['rsa']
            print(f"\n🎯 Representational Similarity:")
            for metric, correlations in rsa.items():
                print(f"   {metric.capitalize()} Distance:")
                for label_type, correlation in correlations.items():
                    print(f"     {label_type}: r = {correlation:.3f}")
        
        print("\n" + "="*60)

# ========================
# EXAMPLE USAGE
# ========================

def create_example_evaluation():
    """Example usage of the evaluation module."""
    print("🧪 Creating example evaluation...")
    
    # Create synthetic embedding data
    np.random.seed(42)
    n_samples = 200
    embedding_dim = 64
    
    # Create structured embeddings (3 clusters)
    cluster_centers = np.array([[0, 0], [5, 5], [-5, 5]])
    n_per_cluster = n_samples // 3
    
    embeddings_2d = []
    categories = []
    metadata = []
    
    for i, center in enumerate(cluster_centers):
        # Generate points around cluster center
        cluster_points = np.random.multivariate_normal(
            center, np.eye(2), n_per_cluster
        )
        embeddings_2d.extend(cluster_points)
        
        # Create labels
        categories.extend([f'category_{i}'] * n_per_cluster)
        
        # Create metadata
        for j in range(n_per_cluster):
            metadata.append({
                'image_type': f'type_{i}',
                'category': f'category_{i}',
                'family': f'family_{i}_{j // 10}',
                'image_id': len(metadata)
            })
    
    # Expand to full embedding dimension
    embeddings = np.zeros((len(embeddings_2d), embedding_dim))
    embeddings[:, :2] = embeddings_2d
    embeddings[:, 2:] = np.random.randn(len(embeddings_2d), embedding_dim - 2) * 0.1
    
    # Configure evaluation
    config = EvaluationConfig(
        n_clusters_range=(2, 6),
        cv_folds=3,
        figsize=(10, 6)
    )
    
    # Run evaluation
    evaluator = NeuralEmbeddingEvaluator(config)
    results = evaluator.evaluate_all(embeddings, categories, metadata)
    
    # Print summary
    evaluator.print_summary_report(results)
    
    return results

def create_example_cross_region_evaluation():
    """Example cross-region comparison."""
    print("🧪 Creating example cross-region evaluation...")
    
    np.random.seed(42)
    n_samples = 150
    embedding_dim = 32
    
    # Simulate 3 brain regions with different clustering properties
    regions = ['V1', 'V2', 'V4']
    embeddings_dict = {}
    categories_dict = {}
    metadata_dict = {}
    
    # Create shared category structure
    base_categories = ['natural_animals', 'natural_objects', 'texture_rough', 'texture_smooth', 'noise_white']
    
    for region_idx, region in enumerate(regions):
        print(f"   Creating {region} embeddings...")
        
        # Each region has different clustering strength
        cluster_strength = [1.0, 2.0, 3.0][region_idx]  # V1 < V2 < V4 clustering
        
        region_embeddings = []
        region_categories = []
        region_metadata = []
        
        n_per_category = n_samples // len(base_categories)
        
        for cat_idx, category in enumerate(base_categories):
            # Create cluster center
            center = np.random.randn(embedding_dim) * 5
            
            # Generate points with region-specific clustering strength
            for i in range(n_per_category):
                if i < len(region_embeddings):
                    continue
                    
                # Add noise based on clustering strength (higher = tighter clusters)
                noise_scale = 1.0 / cluster_strength
                point = center + np.random.randn(embedding_dim) * noise_scale
                
                region_embeddings.append(point)
                region_categories.append(category)
                
                # Create metadata
                image_type, category_name = category.split('_', 1)
                region_metadata.append({
                    'image_type': image_type,
                    'category': category_name,
                    'full_category': category,
                    'family': f'family_{cat_idx}_{i // 5}',
                    'image_id': len(region_metadata),
                    'region': region
                })
        
        embeddings_dict[region] = np.array(region_embeddings)
        categories_dict[region] = region_categories
        metadata_dict[region] = region_metadata
    
    # Configure evaluation
    config = EvaluationConfig(
        n_clusters_range=(2, 10),
        cv_folds=3,
        figsize=(10, 6)
    )
    
    # Run cross-region evaluation
    evaluator = NeuralEmbeddingEvaluator(config)
    
    cross_region_results = evaluator.evaluate_cross_region(
        embeddings_dict,
        categories_dict, 
        metadata_dict,
        normalization_method='unit_sphere',
        align_to_reference='V1'  # Align all to V1
    )
    
    # Print summary
    evaluator.print_cross_region_summary(cross_region_results)
    
    return cross_region_results

def demonstrate_normalization_methods():
    """Demonstrate different normalization methods."""
    print("🧪 Demonstrating normalization methods...")
    
    # Create example embeddings with different scales
    np.random.seed(42)
    embeddings_raw = np.random.randn(100, 10)
    
    # Scale and shift to simulate different regions
    v1_embeddings = embeddings_raw * 1.0 + 0.0  # Original scale
    v2_embeddings = embeddings_raw * 3.0 + 2.0  # Larger scale, shifted
    v4_embeddings = embeddings_raw * 0.5 - 1.0  # Smaller scale, shifted
    
    normalizer = EmbeddingNormalizer()
    
    print("\nOriginal embeddings statistics:")
    print(f"V1: mean={np.mean(v1_embeddings):.3f}, std={np.std(v1_embeddings):.3f}")
    print(f"V2: mean={np.mean(v2_embeddings):.3f}, std={np.std(v2_embeddings):.3f}")
    print(f"V4: mean={np.mean(v4_embeddings):.3f}, std={np.std(v4_embeddings):.3f}")
    
    # Test different normalization methods
    methods = ['unit_sphere', 'standardize', 'center', 'robust']
    
    for method in methods:
        print(f"\nAfter {method} normalization:")
        
        v1_norm = normalizer.normalize_embeddings(v1_embeddings, method, True, 'V1')
        v2_norm = normalizer.normalize_embeddings(v2_embeddings, method, True, 'V2') 
        v4_norm = normalizer.normalize_embeddings(v4_embeddings, method, True, 'V4')
        
        print(f"V1: mean={np.mean(v1_norm):.3f}, std={np.std(v1_norm):.3f}")
        print(f"V2: mean={np.mean(v2_norm):.3f}, std={np.std(v2_norm):.3f}")
        print(f"V4: mean={np.mean(v4_norm):.3f}, std={np.std(v4_norm):.3f}")

if __name__ == "__main__":
    print("🚀 Neural Representation Evaluation Examples")
    print("=" * 60)
    
    # Run single region evaluation
    print("\n1. Single Region Evaluation:")
    example_results = create_example_evaluation()
    
    # Run cross-region evaluation
    print("\n2. Cross-Region Evaluation:")
    cross_region_results = create_example_cross_region_evaluation()
    
    # Demonstrate normalization
    print("\n3. Normalization Methods:")
    demonstrate_normalization_methods()
    
    print("\n🎉 All examples completed!")
    
    # Show usage summary
    print("\n" + "="*60)
    print("📖 USAGE SUMMARY FOR YOUR RESEARCH")
    print("="*60)
    print("""
SINGLE REGION ANALYSIS:
# Extract embeddings from trained model
embeddings, categories, metadata = learner.extract_embeddings(dataloader)

# Evaluate
evaluator = NeuralEmbeddingEvaluator()
results = evaluator.evaluate_all(embeddings, categories, metadata, save_dir="results/V1")

CROSS-REGION COMPARISON:
# Extract embeddings from each region
v1_embeddings, v1_categories, v1_metadata = learner_v1.extract_embeddings(dataloader)
v2_embeddings, v2_categories, v2_metadata = learner_v2.extract_embeddings(dataloader)

# Compare across regions with normalization
embeddings_dict = {'V1': v1_embeddings, 'V2': v2_embeddings}
categories_dict = {'V1': v1_categories, 'V2': v2_categories}
metadata_dict = {'V1': v1_metadata, 'V2': v2_metadata}

cross_results = evaluator.evaluate_cross_region(
    embeddings_dict, categories_dict, metadata_dict,
    normalization_method='unit_sphere',  # or 'standardize', 'center'
    align_to_reference='V1',  # Optional Procrustes alignment
    save_dir="results/cross_region"
)

RESEARCH QUESTIONS ANSWERED:
✅ Do higher visual areas show better clustering within categories?
   → Compare clustering metrics across regions
✅ Are similar categories closer in higher areas?
   → Analyze RSA correlations across hierarchy
✅ How do representation geometries differ across regions?
   → Use embedding similarity metrics and visualizations
""")
    print("="*60)
