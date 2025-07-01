#!/usr/bin/env python3
"""
Advanced Neural Representation Evaluation Script

This script demonstrates advanced evaluation techniques for neural representation learning,
including detailed RSA analysis, embedding manifold analysis, and interpretability methods.

Advanced Topics Covered:
1. Detailed Representational Similarity Analysis (RSA)
2. Embedding manifold analysis and topology
3. Interpretability and feature analysis
4. Temporal dynamics analysis (for temporal data)
5. Robustness and generalization testing
6. Advanced visualization techniques

Expected runtime: 20-40 minutes
Prerequisites: Run basic usage or cross-region comparison first to have trained models.
"""

# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
import pickle
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Import our packages (you may need to adjust these imports based on your setup)
try:
    from neural_repr_package import NeuralRepresentationLearner, Config
    from neural_evaluation import NeuralEmbeddingEvaluator, EvaluationConfig
except ImportError:
    print("Warning: neural_repr_package and neural_evaluation not found.")
    print("You may need to install these packages or adjust the import paths.")

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("📦 Advanced analysis packages imported successfully!")
print(f"🔧 CUDA available: {torch.cuda.is_available()}")


def load_or_create_analysis_data():
    """
    Load existing results or create new data for advanced analysis.
    """
    results_dir = Path("results")
    
    # Try to load existing results
    embeddings_path = results_dir / "cross_region_embeddings.npz"
    
    if embeddings_path.exists():
        print("📂 Loading existing cross-region results...")
        
        # Load embeddings
        data = np.load(embeddings_path, allow_pickle=True)
        regions = data['regions']
        
        embeddings_dict = {}
        categories_dict = {}
        
        for region in regions:
            embeddings_dict[region] = data[f'{region}_embeddings']
            categories_dict[region] = data[f'{region}_categories']
        
        print(f"   Loaded {len(regions)} regions: {list(regions)}")
        return embeddings_dict, categories_dict, regions
    
    else:
        print("📊 Creating new data for advanced analysis...")
        
        # Create synthetic data for demonstration
        try:
            from neural_repr_package import create_test_data
            neural_data, stim_table = create_test_data()
        except ImportError:
            # Create dummy data if package not available
            print("   Creating synthetic data for demo...")
            np.random.seed(42)
            n_samples = 200
            n_neurons = 100
            n_categories = 5
            
            neural_data = np.random.randn(n_neurons, n_samples)
            categories = [f"category_{i%n_categories}" for i in range(n_samples)]
            
            # Add some structure to the data
            for i in range(n_categories):
                mask = np.array([j%n_categories == i for j in range(n_samples)])
                neural_data[:, mask] += np.random.randn(n_neurons, 1) * 2
            
            # Create dummy embeddings
            embeddings = np.random.randn(n_samples, 16)
            
            # Add category structure to embeddings
            for i in range(n_categories):
                mask = np.array([j%n_categories == i for j in range(n_samples)])
                embeddings[mask] += np.random.randn(1, 16) * 3
            
            embeddings_dict = {'Demo_Region': embeddings}
            categories_dict = {'Demo_Region': categories}
            regions = ['Demo_Region']
            
            print(f"   Created demo region with {embeddings.shape[0]} samples")
            return embeddings_dict, categories_dict, regions
        
        # Configure and train a quick model (if packages available)
        config = Config(
            hidden_dims=[64, 32],
            embedding_dim=16,
            batch_size=16,
            cuda_id=0 if torch.cuda.is_available() else None
        )
        
        learner = NeuralRepresentationLearner(config)
        train_ds, test_ds, info = learner.create_datasets_from_stim_table(neural_data, stim_table)
        
        train_loader = learner.create_dataloader(train_ds)
        test_loader = learner.create_dataloader(test_ds, shuffle=False)
        
        learner.initialize_model(neural_data.shape[0])
        
        # Quick training
        for epoch in range(10):
            loss = learner.train_epoch(train_loader)
            if epoch % 3 == 0:
                print(f"   Epoch {epoch+1}/10: Loss = {loss:.4f}")
        
        # Extract embeddings
        embeddings, categories, metadata = learner.extract_embeddings(test_loader)
        
        # Create single-region result for demo
        embeddings_dict = {'Demo_Region': embeddings}
        categories_dict = {'Demo_Region': categories}
        regions = ['Demo_Region']
        
        print(f"   Created demo region with {embeddings.shape[0]} samples")
        return embeddings_dict, categories_dict, regions


def perform_detailed_rsa(embeddings, categories, metadata=None):
    """
    Perform detailed RSA analysis with multiple similarity measures.
    """
    print("🔍 Performing detailed RSA analysis...")
    
    n_samples = len(embeddings)
    
    # 1. Compute multiple distance matrices
    distance_metrics = ['euclidean', 'cosine', 'correlation', 'manhattan']
    distance_matrices = {}
    
    for metric in distance_metrics:
        if metric == 'correlation':
            # Correlation distance: 1 - correlation
            corr_matrix = np.corrcoef(embeddings)
            distance_matrices[metric] = 1 - corr_matrix
        else:
            distances = pairwise_distances(embeddings, metric=metric)
            distance_matrices[metric] = distances
    
    # 2. Create categorical similarity matrices
    unique_categories = list(set(categories))
    n_categories = len(unique_categories)
    
    # Binary category similarity
    category_similarity = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            category_similarity[i, j] = 1.0 if categories[i] == categories[j] else 0.0
    
    # 3. Hierarchical similarity (if metadata available)
    hierarchical_similarities = {}
    if metadata:
        # Try to extract hierarchical info
        hierarchy_keys = ['image_type', 'category']
        
        for key in hierarchy_keys:
            if all(key in meta for meta in metadata):
                values = [meta[key] for meta in metadata]
                similarity_matrix = np.zeros((n_samples, n_samples))
                
                for i in range(n_samples):
                    for j in range(n_samples):
                        similarity_matrix[i, j] = 1.0 if values[i] == values[j] else 0.0
                
                hierarchical_similarities[key] = similarity_matrix
    
    # 4. Compute RSA correlations
    rsa_results = {}
    
    for metric, dist_matrix in distance_matrices.items():
        # Convert distance to similarity
        similarity_matrix = 1 / (1 + dist_matrix)
        
        # Get upper triangle (exclude diagonal)
        triu_indices = np.triu_indices_from(similarity_matrix, k=1)
        emb_sim_vec = similarity_matrix[triu_indices]
        cat_sim_vec = category_similarity[triu_indices]
        
        # Spearman correlation
        rsa_corr, rsa_p = stats.spearmanr(emb_sim_vec, cat_sim_vec)
        
        rsa_results[metric] = {
            'correlation': rsa_corr,
            'p_value': rsa_p,
            'similarity_matrix': similarity_matrix
        }
        
        print(f"   {metric.capitalize()}: r = {rsa_corr:.3f}, p = {rsa_p:.3f}")
    
    # 5. Hierarchical RSA
    hierarchical_rsa = {}
    if hierarchical_similarities:
        print(f"\n   Hierarchical RSA:")
        
        for hier_key, hier_sim in hierarchical_similarities.items():
            hier_sim_vec = hier_sim[triu_indices]
            
            # Use best embedding metric
            best_metric = max(rsa_results.keys(), key=lambda x: abs(rsa_results[x]['correlation']))
            best_emb_sim = rsa_results[best_metric]['similarity_matrix'][triu_indices]
            
            hier_corr, hier_p = stats.spearmanr(best_emb_sim, hier_sim_vec)
            hierarchical_rsa[hier_key] = {
                'correlation': hier_corr,
                'p_value': hier_p
            }
            
            print(f"     {hier_key}: r = {hier_corr:.3f}, p = {hier_p:.3f}")
    
    return {
        'distance_matrices': distance_matrices,
        'rsa_results': rsa_results,
        'hierarchical_rsa': hierarchical_rsa,
        'category_similarity': category_similarity
    }


def plot_rsa_results(rsa_results_dict, regions):
    """
    Plot comprehensive RSA visualization.
    """
    n_regions = len(regions)
    fig, axes = plt.subplots(2, n_regions, figsize=(5 * n_regions, 10))
    
    if n_regions == 1:
        axes = axes.reshape(2, 1)
    
    for i, region in enumerate(regions):
        rsa_data = rsa_results_dict[region]
        
        # Plot 1: Best similarity matrix
        best_metric = max(rsa_data['rsa_results'].keys(), 
                         key=lambda x: abs(rsa_data['rsa_results'][x]['correlation']))
        
        similarity_matrix = rsa_data['rsa_results'][best_metric]['similarity_matrix']
        
        im1 = axes[0, i].imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto')
        axes[0, i].set_title(f'{region}\nEmbedding Similarity ({best_metric})')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Plot 2: Category similarity
        category_sim = rsa_data['category_similarity']
        im2 = axes[1, i].imshow(category_sim, cmap='RdYlBu_r', aspect='auto')
        axes[1, i].set_title(f'{region}\nCategory Similarity')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.tight_layout()
    plt.show()
    
    # Plot RSA correlation comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    metrics = ['euclidean', 'cosine', 'correlation', 'manhattan']
    x = np.arange(len(metrics))
    width = 0.8 / len(regions)
    
    for i, region in enumerate(regions):
        correlations = [rsa_results_dict[region]['rsa_results'][metric]['correlation'] 
                       for metric in metrics]
        
        ax.bar(x + i * width, correlations, width, label=region, alpha=0.8)
    
    ax.set_xlabel('Distance Metric')
    ax.set_ylabel('RSA Correlation')
    ax.set_title('RSA Correlations Across Distance Metrics')
    ax.set_xticks(x + width * (len(regions) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_embedding_manifold(embeddings, categories, region_name):
    """
    Comprehensive manifold analysis of embeddings.
    """
    print(f"🌐 Manifold analysis for {region_name}...")
    
    results = {}
    
    # 1. Intrinsic dimensionality estimation
    print(f"   Estimating intrinsic dimensionality...")
    
    # PCA explained variance
    pca = PCA()
    pca.fit(embeddings)
    explained_var = pca.explained_variance_ratio_
    cumsum_var = np.cumsum(explained_var)
    
    # Estimate intrinsic dimension (90% variance)
    intrinsic_dim = np.argmax(cumsum_var >= 0.9) + 1
    
    results['pca'] = {
        'explained_variance_ratio': explained_var,
        'cumulative_variance': cumsum_var,
        'intrinsic_dimension_90': intrinsic_dim,
        'n_components': len(explained_var)
    }
    
    print(f"     Intrinsic dimension (90% var): {intrinsic_dim}/{len(explained_var)}")
    
    # 2. Local neighborhood analysis
    print(f"   Analyzing local neighborhoods...")
    
    # k-NN analysis
    k_values = [5, 10, 20]
    neighbor_analysis = {}
    
    for k in k_values:
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)  # +1 to exclude self
        distances, indices = nbrs.kneighbors(embeddings)
        
        # Calculate same-category neighbors percentage
        same_category_ratios = []
        
        for i in range(len(embeddings)):
            neighbor_indices = indices[i, 1:]  # Exclude self (first index)
            neighbor_categories = [categories[j] for j in neighbor_indices]
            same_category_count = sum(1 for cat in neighbor_categories if cat == categories[i])
            same_category_ratios.append(same_category_count / k)
        
        neighbor_analysis[k] = {
            'mean_same_category_ratio': np.mean(same_category_ratios),
            'std_same_category_ratio': np.std(same_category_ratios),
            'mean_distance': np.mean(distances[:, 1:]),  # Exclude self-distance
            'std_distance': np.std(distances[:, 1:])
        }
        
        print(f"     k={k}: {np.mean(same_category_ratios):.2%} same-category neighbors")
    
    results['neighborhood'] = neighbor_analysis
    
    # 3. Clustering analysis with DBSCAN
    print(f"   DBSCAN clustering analysis...")
    
    # Try different eps values
    eps_values = np.percentile(pairwise_distances(embeddings).flatten(), [10, 25, 50])
    dbscan_results = {}
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        cluster_labels = dbscan.fit_predict(embeddings)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        dbscan_results[f'eps_{eps:.3f}'] = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'noise_ratio': n_noise / len(embeddings),
            'cluster_labels': cluster_labels
        }
    
    results['dbscan'] = dbscan_results
    
    # 4. Manifold learning embeddings
    print(f"   Computing manifold embeddings...")
    
    manifold_methods = {
        'PCA': PCA(n_components=2),
        't-SNE': TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4)),
        'Isomap': Isomap(n_components=2, n_neighbors=min(10, len(embeddings)//10))
    }
    
    manifold_embeddings = {}
    
    for method_name, method in manifold_methods.items():
        try:
            coords_2d = method.fit_transform(embeddings)
            manifold_embeddings[method_name] = coords_2d
            print(f"     {method_name}: Success")
        except Exception as e:
            print(f"     {method_name}: Failed ({str(e)[:50]}...)")
            manifold_embeddings[method_name] = None
    
    results['manifold_embeddings'] = manifold_embeddings
    
    return results


def plot_manifold_analysis(manifold_results_dict, regions, categories_dict):
    """
    Plot manifold analysis results.
    """
    n_regions = len(regions)
    
    # Plot 1: PCA explained variance
    fig, axes = plt.subplots(1, n_regions, figsize=(6 * n_regions, 5))
    if n_regions == 1:
        axes = [axes]
    
    for i, region in enumerate(regions):
        pca_data = manifold_results_dict[region]['pca']
        
        # Plot explained variance
        axes[i].plot(range(1, len(pca_data['explained_variance_ratio']) + 1), 
                    pca_data['cumulative_variance'], 'bo-', linewidth=2)
        axes[i].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% variance')
        axes[i].axvline(x=pca_data['intrinsic_dimension_90'], color='red', linestyle='--', alpha=0.7)
        
        axes[i].set_xlabel('Principal Component')
        axes[i].set_ylabel('Cumulative Explained Variance')
        axes[i].set_title(f'{region}\nIntrinsic Dimension Analysis')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
        # Add text annotation
        axes[i].text(0.7, 0.2, f'Intrinsic dim: {pca_data["intrinsic_dimension_90"]}', 
                    transform=axes[i].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Neighborhood analysis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    k_values = [5, 10, 20]
    x = np.arange(len(k_values))
    width = 0.8 / len(regions)
    
    for i, region in enumerate(regions):
        neighbor_data = manifold_results_dict[region]['neighborhood']
        ratios = [neighbor_data[k]['mean_same_category_ratio'] for k in k_values]
        stds = [neighbor_data[k]['std_same_category_ratio'] for k in k_values]
        
        ax.bar(x + i * width, ratios, width, yerr=stds, 
               label=region, alpha=0.8, capsize=5)
    
    ax.set_xlabel('k (Number of Neighbors)')
    ax.set_ylabel('Same-Category Neighbor Ratio')
    ax.set_title('Local Neighborhood Purity')
    ax.set_xticks(x + width * (len(regions) - 1) / 2)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Manifold embeddings
    methods = ['PCA', 't-SNE', 'Isomap']
    available_methods = [m for m in methods if any(
        manifold_results_dict[region]['manifold_embeddings'].get(m) is not None 
        for region in regions
    )]
    
    if available_methods:
        fig, axes = plt.subplots(len(regions), len(available_methods), 
                               figsize=(6 * len(available_methods), 5 * len(regions)))
        
        if len(regions) == 1:
            axes = axes.reshape(1, -1)
        if len(available_methods) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, region in enumerate(regions):
            categories = categories_dict[region]
            unique_categories = list(set(categories))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
            color_map = dict(zip(unique_categories, colors))
            
            for j, method in enumerate(available_methods):
                coords = manifold_results_dict[region]['manifold_embeddings'].get(method)
                
                if coords is not None:
                    for category in unique_categories:
                        mask = np.array(categories) == category
                        if np.any(mask):
                            axes[i, j].scatter(coords[mask, 0], coords[mask, 1], 
                                             c=[color_map[category]], label=category, alpha=0.7)
                    
                    axes[i, j].set_title(f'{region} - {method}')
                    axes[i, j].set_xlabel(f'{method} Component 1')
                    axes[i, j].set_ylabel(f'{method} Component 2')
                    
                    if len(unique_categories) <= 10:  # Only show legend if not too many
                        axes[i, j].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    axes[i, j].text(0.5, 0.5, f'{method}\nNot Available', 
                                   ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].set_title(f'{region} - {method}')
        
        plt.tight_layout()
        plt.show()


def analyze_embedding_interpretability(embeddings, categories, region_name):
    """
    Analyze the interpretability of embedding dimensions.
    """
    print(f"🔍 Interpretability analysis for {region_name}...")
    
    results = {}
    
    # 1. Dimension variance analysis
    dim_variances = np.var(embeddings, axis=0)
    dim_means = np.mean(embeddings, axis=0)
    
    # Sort dimensions by variance
    variance_order = np.argsort(dim_variances)[::-1]
    
    results['dimension_stats'] = {
        'variances': dim_variances,
        'means': dim_means,
        'variance_order': variance_order,
        'top_dims': variance_order[:5]  # Top 5 most variable dimensions
    }
    
    print(f"   Top 5 most variable dimensions: {variance_order[:5]}")
    print(f"   Variance range: {dim_variances.min():.3f} - {dim_variances.max():.3f}")
    
    # 2. Category separability per dimension
    unique_categories = list(set(categories))
    n_categories = len(unique_categories)
    
    if n_categories > 1:
        dim_separability = []
        
        for dim_idx in range(embeddings.shape[1]):
            dim_values = embeddings[:, dim_idx]
            
            # One-way ANOVA across categories
            category_groups = [dim_values[np.array(categories) == cat] for cat in unique_categories]
            category_groups = [group for group in category_groups if len(group) > 0]
            
            if len(category_groups) > 1:
                f_stat, p_value = stats.f_oneway(*category_groups)
                
                # Effect size (eta-squared)
                ss_between = sum(len(group) * (np.mean(group) - np.mean(dim_values))**2 
                               for group in category_groups)
                ss_total = np.sum((dim_values - np.mean(dim_values))**2)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                dim_separability.append({
                    'dimension': dim_idx,
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'eta_squared': eta_squared,
                    'significant': p_value < 0.05
                })
        
        # Sort by effect size
        dim_separability.sort(key=lambda x: x['eta_squared'], reverse=True)
        
        results['dimension_separability'] = dim_separability
        
        # Top separating dimensions
        top_separating = [d['dimension'] for d in dim_separability[:5]]
        print(f"   Top 5 category-separating dimensions: {top_separating}")
        
        n_significant = sum(1 for d in dim_separability if d['significant'])
        print(f"   Significantly separating dimensions: {n_significant}/{len(dim_separability)}")
    
    # 3. Correlation between high-variance and separating dimensions
    if 'dimension_separability' in results:
        top_variance_dims = set(variance_order[:10])
        top_separating_dims = set(d['dimension'] for d in dim_separability[:10])
        
        overlap = len(top_variance_dims.intersection(top_separating_dims))
        overlap_ratio = overlap / min(len(top_variance_dims), len(top_separating_dims))
        
        results['variance_separability_overlap'] = {
            'overlap_count': overlap,
            'overlap_ratio': overlap_ratio,
            'top_variance_dims': list(top_variance_dims),
            'top_separating_dims': list(top_separating_dims)
        }
        
        print(f"   Overlap between high-variance and separating dims: {overlap}/10 ({overlap_ratio:.1%})")
    
    # 4. Dimension clustering (which dimensions are correlated?)
    dim_correlations = np.corrcoef(embeddings.T)
    
    # Find highly correlated dimension pairs
    high_corr_pairs = []
    for i in range(len(dim_correlations)):
        for j in range(i+1, len(dim_correlations)):
            if abs(dim_correlations[i, j]) > 0.7:  # High correlation threshold
                high_corr_pairs.append({
                    'dim1': i,
                    'dim2': j,
                    'correlation': dim_correlations[i, j]
                })
    
    results['dimension_correlations'] = {
        'correlation_matrix': dim_correlations,
        'high_correlation_pairs': high_corr_pairs
    }
    
    print(f"   Highly correlated dimension pairs (|r| > 0.7): {len(high_corr_pairs)}")
    
    return results


def plot_interpretability_analysis(interpretability_results_dict, regions):
    """
    Plot interpretability analysis results.
    """
    n_regions = len(regions)
    
    # Plot 1: Dimension variance and separability
    fig, axes = plt.subplots(2, n_regions, figsize=(6 * n_regions, 10))
    if n_regions == 1:
        axes = axes.reshape(2, 1)
    
    for i, region in enumerate(regions):
        results = interpretability_results_dict[region]
        
        # Plot variance per dimension
        variances = results['dimension_stats']['variances']
        axes[0, i].bar(range(len(variances)), variances, alpha=0.7)
        axes[0, i].set_xlabel('Dimension')
        axes[0, i].set_ylabel('Variance')
        axes[0, i].set_title(f'{region}\nDimension Variances')
        axes[0, i].grid(True, alpha=0.3)
        
        # Plot separability (eta-squared) per dimension
        if 'dimension_separability' in results:
            sep_data = results['dimension_separability']
            dims = [d['dimension'] for d in sep_data]
            eta_squared = [d['eta_squared'] for d in sep_data]
            significant = [d['significant'] for d in sep_data]
            
            colors = ['red' if sig else 'blue' for sig in significant]
            
            axes[1, i].bar(dims, eta_squared, alpha=0.7, color=colors)
            axes[1, i].set_xlabel('Dimension')
            axes[1, i].set_ylabel('Effect Size (η²)')
            axes[1, i].set_title(f'{region}\nCategory Separability\n(Red = Significant)')
            axes[1, i].grid(True, alpha=0.3)
        else:
            axes[1, i].text(0.5, 0.5, 'No separability\nanalysis available', 
                           ha='center', va='center', transform=axes[1, i].transAxes)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Dimension correlation matrices
    fig, axes = plt.subplots(1, n_regions, figsize=(6 * n_regions, 5))
    if n_regions == 1:
        axes = [axes]
    
    for i, region in enumerate(regions):
        corr_matrix = interpretability_results_dict[region]['dimension_correlations']['correlation_matrix']
        
        im = axes[i].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        axes[i].set_xlabel('Dimension')
        axes[i].set_ylabel('Dimension')
        axes[i].set_title(f'{region}\nDimension Correlations')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Summary statistics
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    metrics = ['High Variance Dims', 'Separating Dims', 'Significant Dims', 'High Corr Pairs']
    
    x = np.arange(len(metrics))
    width = 0.8 / len(regions)
    
    for i, region in enumerate(regions):
        results = interpretability_results_dict[region]
        
        values = [
            len(results['dimension_stats']['top_dims']),  # Top 5 high variance
            len(results.get('dimension_separability', [])[:5]) if 'dimension_separability' in results else 0,  # Top 5 separating
            sum(1 for d in results.get('dimension_separability', []) if d.get('significant', False)),  # Significant
            len(results['dimension_correlations']['high_correlation_pairs'])  # High corr pairs
        ]
        
        ax.bar(x + i * width, values, width, label=region, alpha=0.8)
    
    ax.set_xlabel('Interpretability Metric')
    ax.set_ylabel('Count')
    ax.set_title('Embedding Interpretability Summary')
    ax.set_xticks(x + width * (len(regions) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_robustness(embeddings, categories, region_name):
    """
    Analyze robustness of embeddings to noise and perturbations.
    """
    print(f"🛡️ Robustness analysis for {region_name}...")
    
    results = {}
    
    # 1. Noise robustness
    print(f"   Testing noise robustness...")
    
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    noise_robustness = {}
    
    # Original clustering quality (using silhouette score)
    original_silhouette = silhouette_score(embeddings, categories) if len(set(categories)) > 1 else 0
    
    for noise_level in noise_levels:
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, embeddings.shape)
        noisy_embeddings = embeddings + noise
        
        # Compute silhouette score with noise
        if len(set(categories)) > 1:
            noisy_silhouette = silhouette_score(noisy_embeddings, categories)
            robustness_ratio = noisy_silhouette / original_silhouette if original_silhouette > 0 else 0
        else:
            noisy_silhouette = 0
            robustness_ratio = 0
        
        noise_robustness[noise_level] = {
            'silhouette_score': noisy_silhouette,
            'robustness_ratio': robustness_ratio
        }
    
    results['noise_robustness'] = {
        'original_silhouette': original_silhouette,
        'noise_results': noise_robustness
    }
    
    print(f"     Original silhouette: {original_silhouette:.3f}")
    print(f"     Robustness at 10% noise: {noise_robustness[0.1]['robustness_ratio']:.2%}")
    
    # 2. Subset stability
    print(f"   Testing subset stability...")
    
    subset_sizes = [0.5, 0.7, 0.9]
    n_subsets = 5
    subset_stability = {}
    
    for subset_size in subset_sizes:
        subset_silhouettes = []
        
        for _ in range(n_subsets):
            # Random subset
            n_subset = int(len(embeddings) * subset_size)
            subset_indices = np.random.choice(len(embeddings), n_subset, replace=False)
            
            subset_embeddings = embeddings[subset_indices]
            subset_categories = [categories[i] for i in subset_indices]
            
            if len(set(subset_categories)) > 1:
                subset_silhouette = silhouette_score(subset_embeddings, subset_categories)
                subset_silhouettes.append(subset_silhouette)
        
        if subset_silhouettes:
            subset_stability[subset_size] = {
                'mean_silhouette': np.mean(subset_silhouettes),
                'std_silhouette': np.std(subset_silhouettes),
                'stability_ratio': np.mean(subset_silhouettes) / original_silhouette if original_silhouette > 0 else 0
            }
    
    results['subset_stability'] = subset_stability
    
    # 3. Distance preservation under perturbations
    print(f"   Testing distance preservation...")
    
    # Compute original pairwise distances (sample for efficiency)
    n_sample = min(100, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), n_sample, replace=False)
    sample_embeddings = embeddings[sample_indices]
    
    original_distances = pairwise_distances(sample_embeddings)
    
    distance_preservation = {}
    
    for noise_level in [0.05, 0.1, 0.2]:
        noise = np.random.normal(0, noise_level, sample_embeddings.shape)
        noisy_sample = sample_embeddings + noise
        noisy_distances = pairwise_distances(noisy_sample)
        
        # Correlation between original and noisy distances
        dist_correlation = np.corrcoef(original_distances.flatten(), noisy_distances.flatten())[0, 1]
        
        distance_preservation[noise_level] = {
            'distance_correlation': dist_correlation
        }
    
    results['distance_preservation'] = distance_preservation
    
    return results


def plot_robustness_analysis(robustness_results_dict, regions):
    """
    Plot robustness analysis results.
    """
    # Plot 1: Noise robustness
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Noise robustness curves
    for region in regions:
        noise_data = robustness_results_dict[region]['noise_robustness']['noise_results']
        noise_levels = list(noise_data.keys())
        robustness_ratios = [noise_data[level]['robustness_ratio'] for level in noise_levels]
        
        axes[0].plot(noise_levels, robustness_ratios, 'o-', linewidth=2, label=region)
    
    axes[0].set_xlabel('Noise Level (σ)')
    axes[0].set_ylabel('Clustering Quality Ratio')
    axes[0].set_title('Noise Robustness')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Original')
    axes[0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
    
    # Distance preservation
    for region in regions:
        dist_data = robustness_results_dict[region]['distance_preservation']
        noise_levels = list(dist_data.keys())
        correlations = [dist_data[level]['distance_correlation'] for level in noise_levels]
        
        axes[1].plot(noise_levels, correlations, 's-', linewidth=2, label=region)
    
    axes[1].set_xlabel('Noise Level (σ)')
    axes[1].set_ylabel('Distance Correlation')
    axes[1].set_title('Distance Preservation Under Noise')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Subset stability
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    subset_sizes = [0.5, 0.7, 0.9]
    x = np.arange(len(subset_sizes))
    width = 0.8 / len(regions)
    
    for i, region in enumerate(regions):
        subset_data = robustness_results_dict[region]['subset_stability']
        stability_ratios = [subset_data[size]['stability_ratio'] for size in subset_sizes if size in subset_data]
        stability_stds = [subset_data[size]['std_silhouette'] / robustness_results_dict[region]['noise_robustness']['original_silhouette'] 
                         for size in subset_sizes if size in subset_data]
        
        if len(stability_ratios) == len(subset_sizes):
            ax.bar(x + i * width, stability_ratios, width, yerr=stability_stds, 
                   label=region, alpha=0.8, capsize=5)
    
    ax.set_xlabel('Subset Size')
    ax.set_ylabel('Stability Ratio')
    ax.set_title('Subset Stability Analysis')
    ax.set_xticks(x + width * (len(regions) - 1) / 2)
    ax.set_xticklabels([f'{int(s*100)}%' for s in subset_sizes])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def generate_comprehensive_report(regions, rsa_results_dict, manifold_results_dict, 
                                interpretability_results_dict, robustness_results_dict):
    """
    Generate a comprehensive summary report of all analyses.
    """
    print("📋 COMPREHENSIVE ADVANCED ANALYSIS REPORT")
    print("=" * 60)
    
    for region in regions:
        print(f"\n🧠 {region.upper()} ANALYSIS SUMMARY")
        print("-" * 40)
        
        # 1. RSA Summary
        if region in rsa_results_dict:
            rsa_data = rsa_results_dict[region]['rsa_results']
            best_rsa_metric = max(rsa_data.keys(), key=lambda x: abs(rsa_data[x]['correlation']))
            best_rsa_corr = rsa_data[best_rsa_metric]['correlation']
            
            print(f"📊 Representational Similarity Analysis:")
            print(f"   Best RSA correlation: {best_rsa_corr:.3f} ({best_rsa_metric})")
            
            significant_rsa = sum(1 for metric in rsa_data.values() if metric['p_value'] < 0.05)
            print(f"   Significant correlations: {significant_rsa}/{len(rsa_data)}")
        
        # 2. Manifold Summary
        if region in manifold_results_dict:
            manifold_data = manifold_results_dict[region]
            intrinsic_dim = manifold_data['pca']['intrinsic_dimension_90']
            total_dim = manifold_data['pca']['n_components']
            
            print(f"\n🌐 Manifold Analysis:")
            print(f"   Intrinsic dimension (90% var): {intrinsic_dim}/{total_dim}")
            print(f"   Dimension efficiency: {intrinsic_dim/total_dim:.1%}")
            
            # Neighborhood purity
            if 'neighborhood' in manifold_data:
                k10_purity = manifold_data['neighborhood'][10]['mean_same_category_ratio']
                print(f"   Local purity (k=10): {k10_purity:.1%}")
        
        # 3. Interpretability Summary
        if region in interpretability_results_dict:
            interp_data = interpretability_results_dict[region]
            
            print(f"\n🔍 Interpretability Analysis:")
            
            if 'dimension_separability' in interp_data:
                sep_data = interp_data['dimension_separability']
                n_significant = sum(1 for d in sep_data if d['significant'])
                best_eta2 = max(d['eta_squared'] for d in sep_data) if sep_data else 0
                
                print(f"   Category-separating dimensions: {n_significant}/{len(sep_data)}")
                print(f"   Best effect size (η²): {best_eta2:.3f}")
            
            if 'variance_separability_overlap' in interp_data:
                overlap_data = interp_data['variance_separability_overlap']
                print(f"   Variance-separability overlap: {overlap_data['overlap_ratio']:.1%}")
            
            # Dimension correlations
            n_high_corr = len(interp_data['dimension_correlations']['high_correlation_pairs'])
            print(f"   Highly correlated dim pairs: {n_high_corr}")
        
        # 4. Robustness Summary
        if region in robustness_results_dict:
            robust_data = robustness_results_dict[region]
            
            print(f"\n🛡️ Robustness Analysis:")
            
            original_silhouette = robust_data['noise_robustness']['original_silhouette']
            robustness_10 = robust_data['noise_robustness']['noise_results'][0.1]['robustness_ratio']
            
            print(f"   Original clustering quality: {original_silhouette:.3f}")
            print(f"   Robustness at 10% noise: {robustness_10:.1%}")
            
            # Distance preservation
            dist_preserv_10 = robust_data['distance_preservation'][0.1]['distance_correlation']
            print(f"   Distance preservation (10% noise): {dist_preserv_10:.3f}")
        
        print()


def main():
    """
    Main function to run the complete advanced analysis pipeline.
    """
    print("🚀 Starting Advanced Neural Representation Evaluation")
    print("=" * 60)
    
    # 1. Load or create analysis data
    print("\n1. Loading Data...")
    embeddings_dict, categories_dict, regions = load_or_create_analysis_data()
    
    print(f"\n✅ Data ready for advanced analysis:")
    for region in regions:
        emb_shape = embeddings_dict[region].shape
        n_categories = len(set(categories_dict[region]))
        print(f"   {region}: {emb_shape[0]} samples, {emb_shape[1]}D, {n_categories} categories")
    
    # 2. Perform RSA analysis
    print("\n2. Performing RSA Analysis...")
    rsa_results_dict = {}
    
    for region in regions:
        print(f"\n🧠 RSA Analysis for {region}:")
        
        # Get metadata if available (create dummy if not)
        categories = categories_dict[region]
        metadata = None
        
        # Create dummy metadata for demo
        if region == 'Demo_Region':
            metadata = []
            for i, cat in enumerate(categories):
                # Extract image_type and category from full category name
                parts = cat.split('_')
                if len(parts) >= 2:
                    image_type, category = parts[0], '_'.join(parts[1:])
                else:
                    image_type, category = 'unknown', cat
                
                metadata.append({
                    'image_type': image_type,
                    'category': category,
                    'image_id': i
                })
        
        rsa_results_dict[region] = perform_detailed_rsa(
            embeddings_dict[region], 
            categories, 
            metadata
        )
    
    print(f"\n✅ RSA analysis completed for all regions!")
    
    # Visualize RSA results
    plot_rsa_results(rsa_results_dict, regions)
    
    # 3. Perform manifold analysis
    print("\n3. Performing Manifold Analysis...")
    manifold_results_dict = {}
    
    for region in regions:
        print(f"\n🧠 {region} Manifold Analysis:")
        manifold_results_dict[region] = analyze_embedding_manifold(
            embeddings_dict[region], 
            categories_dict[region], 
            region
        )
    
    print(f"\n✅ Manifold analysis completed for all regions!")
    
    # Visualize manifold analysis
    plot_manifold_analysis(manifold_results_dict, regions, categories_dict)
    
    # 4. Perform interpretability analysis
    print("\n4. Performing Interpretability Analysis...")
    interpretability_results_dict = {}
    
    for region in regions:
        print(f"\n🧠 {region} Interpretability Analysis:")
        interpretability_results_dict[region] = analyze_embedding_interpretability(
            embeddings_dict[region], 
            categories_dict[region], 
            region
        )
    
    print(f"\n✅ Interpretability analysis completed for all regions!")
    
    # Visualize interpretability analysis
    plot_interpretability_analysis(interpretability_results_dict, regions)
    
    # 5. Perform robustness analysis
    print("\n5. Performing Robustness Analysis...")
    robustness_results_dict = {}
    
    for region in regions:
        print(f"\n🧠 {region} Robustness Analysis:")
        np.random.seed(42)  # For reproducible results
        robustness_results_dict[region] = analyze_robustness(
            embeddings_dict[region], 
            categories_dict[region], 
            region
        )
    
    print(f"\n✅ Robustness analysis completed for all regions!")
    
    # Visualize robustness analysis
    plot_robustness_analysis(robustness_results_dict, regions)
    
    # 6. Generate comprehensive report
    print("\n6. Generating Comprehensive Report...")
    generate_comprehensive_report(
        regions, 
        rsa_results_dict, 
        manifold_results_dict, 
        interpretability_results_dict, 
        robustness_results_dict
    )
    
    print(f"\n🎉 Advanced analysis pipeline completed successfully!")
    print("=" * 60)
    
    return {
        'embeddings': embeddings_dict,
        'categories': categories_dict,
        'regions': regions,
        'rsa_results': rsa_results_dict,
        'manifold_results': manifold_results_dict,
        'interpretability_results': interpretability_results_dict,
        'robustness_results': robustness_results_dict
    }


if __name__ == "__main__":
    # Run the complete analysis
    results = main()