# Neural Representation Learning Package

A comprehensive toolkit for learning and analyzing neural representations using contrastive learning, designed specifically for cross-region comparison in the mouse visual hierarchy.

## 🎯 Research Focus

This package enables testing the hypothesis that **higher visual cortical regions naturally show more clusters within categories and closeness between similar categories** based on neural responses.

## 📦 Package Components

### Core Training (`neural_repr_package.py`)
- **GPU-ready contrastive learning** with InfoNCE loss
- **Automatic data preprocessing** with temporal binning and smoothing
- **Balanced sampling** for imbalanced categories
- **Smart stimulus table conversion** with auto-detection
- **Cross-region model training** with consistent parameters

### Advanced Evaluation (`neural_evaluation.py`)
- **Clustering quality assessment** (silhouette, ARI, NMI)
- **Linear probing analysis** with cross-validation
- **Representational Similarity Analysis (RSA)**
- **Embedding normalization** for cross-region comparison
- **Comprehensive visualization tools**

### Interactive Notebooks
- `01_basic_usage.ipynb` - Complete training pipeline
- `02_cross_region_comparison.ipynb` - Multi-region analysis
- `03_advanced_evaluation.ipynb` - Advanced metrics and interpretability

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone/download the package
git clone <repository> # or download files

# Create conda environment (with GPU support)
conda env create -f environment.yml
conda activate neural-repr-learning

# OR CPU-only version
conda env create -f environment-cpu.yml
conda activate neural-repr-learning-cpu

# Verify installation
python setup.py
```

### 2. Basic Usage

```python
from neural_repr_package import NeuralRepresentationLearner, Config
from neural_evaluation import NeuralEmbeddingEvaluator

# Configure model
config = Config(
    hidden_dims=[128, 64],
    embedding_dim=32,
    batch_size=32,
    cuda_id=0  # Use GPU 0, or None for CPU
)

# Train model
learner = NeuralRepresentationLearner(config)
train_ds, test_ds, info = learner.create_datasets_from_stim_table(
    neural_data,  # (neurons, images, trials, [timebins])
    stim_table    # DataFrame with stimulus info
)

# Initialize and train
learner.initialize_model(neural_data.shape[0])
for epoch in range(num_epochs):
    loss = learner.train_epoch(train_loader)

# Extract embeddings for analysis
embeddings, categories, metadata = learner.extract_embeddings(test_loader)
```

### 3. Cross-Region Analysis

```python
# Train models for multiple regions
regions = ['V1', 'V2', 'V4']
embeddings_dict = {}

for region in regions:
    # Train region-specific model
    learner = NeuralRepresentationLearner(config)
    # ... training code ...
    embeddings_dict[region] = learner.extract_embeddings(test_loader)

# Compare across regions with proper normalization
evaluator = NeuralEmbeddingEvaluator()
cross_results = evaluator.evaluate_cross_region(
    embeddings_dict, categories_dict, metadata_dict,
    normalization_method='unit_sphere',  # Essential!
    save_dir="results/cross_region"
)

# Print comprehensive comparison
evaluator.print_cross_region_summary(cross_results)
```

## 📊 Data Format

### Neural Data
```python
# Temporal data (recommended)
neural_data.shape  # (n_neurons, n_images, n_trials, n_timebins)

# Or trial-averaged data
neural_data.shape  # (n_neurons, n_images, n_trials)
```

### Stimulus Table
```python
# Required columns (auto-detected)
stim_table = pd.DataFrame({
    'stim_type': ['natural_images', 'textures', 'noise'],     # Stimulus type
    'category': ['animals', 'objects', 'rough'],              # Category within type
    'unique_img': ['img_001.jpg', 'img_002.jpg'],            # Image identifier
    'block': [1, 1, 2],                                      # Optional: experimental block
    # ... additional columns preserved
})
```

## 🧠 Research Applications

### Cross-Hierarchy Comparison
```python
# Test core hypothesis: do higher areas show better clustering?
v1_results = evaluator.evaluate_all(v1_embeddings, v1_categories)
v2_results = evaluator.evaluate_all(v2_embeddings, v2_categories)
v4_results = evaluator.evaluate_all(v4_embeddings, v4_categories)

# Compare clustering quality
v1_silhouette = v1_results['clustering']['best_silhouette']
v2_silhouette = v2_results['clustering']['best_silhouette'] 
v4_silhouette = v4_results['clustering']['best_silhouette']

print(f"Clustering quality: V1={v1_silhouette:.3f}, V2={v2_silhouette:.3f}, V4={v4_silhouette:.3f}")
# Expect: V1 < V2 < V4 if hypothesis is correct
```

### Advanced Metrics
- **Silhouette Score**: Within-category clustering quality
- **Adjusted Rand Index**: Alignment with ground truth categories  
- **Linear Probing Accuracy**: Category separability
- **RSA Correlations**: Representational similarity structure
- **Canonical Correlation Analysis**: Cross-region similarity
- **Robustness Testing**: Stability under noise

## 📁 Directory Structure

```
neural-repr-learning/
├── environment.yml              # Conda environment (GPU)
├── environment-cpu.yml          # Conda environment (CPU)
├── neural_repr_package.py       # Core training package
├── neural_evaluation.py         # Evaluation and analysis tools
├── test_neural_package.py       # Unit tests
├── setup.py                     # Setup and validation script
│
├── notebooks/                   # Interactive analysis
│   ├── 01_basic_usage.ipynb
│   ├── 02_cross_region_comparison.ipynb
│   └── 03_advanced_evaluation.ipynb
│
├── data/                        # Your neural data
│   ├── raw/                     # Original data files
│   └── processed/               # Preprocessed data
│
├── results/                     # All outputs
│   ├── models/                  # Trained models
│   ├── embeddings/              # Extracted embeddings
│   ├── plots/                   # Visualizations
│   └── cross_region/            # Cross-region analysis
│
├── scripts/                     # Batch processing
│   └── batch_process.py         # Multi-region pipeline
│
└── configs/                     # Configuration files
    └── example_config.yaml
```

## 🔧 Key Features

### ✅ GPU Acceleration
- Automatic device detection and management
- CUDA memory optimization
- CPU fallback for compatibility

### ✅ Robust Preprocessing
- Temporal binning and Gaussian smoothing
- Z-score normalization across neurons
- Trial averaging with reliability weighting

### ✅ Fair Cross-Region Comparison
- **Essential embedding normalization** (unit sphere, standardization)
- Procrustes alignment for geometric comparison
- Canonical correlation analysis (CCA)
- Linear centered kernel alignment (CKA)

### ✅ Comprehensive Evaluation
- Multiple clustering metrics
- Cross-validated linear probing
- Representational similarity analysis (RSA)
- Manifold analysis and interpretability
- Noise robustness testing

### ✅ Publication-Ready Outputs
- Automated plot generation
- Statistical significance testing
- Comprehensive summary reports
- Reproducible analysis pipeline

## 🎯 Expected Results

For the mouse visual hierarchy hypothesis:

### Clustering Quality (Silhouette Score)
```
V1:  0.2-0.4  (weak clustering)
V2:  0.4-0.6  (moderate clustering)  
V4:  0.6-0.8  (strong clustering)
```

### Linear Probing Accuracy
```
V1:  60-75%   (basic separability)
V2:  75-85%   (good separability)
V4:  85-95%   (excellent separability)
```

### Cross-Region Similarity
```
V1 ↔ V2:  High similarity (adjacent regions)
V1 ↔ V4:  Low similarity (distant regions)
V2 ↔ V4:  Moderate similarity (adjacent regions)
```

## 🔬 Advanced Analysis

### Temporal Dynamics (for 4D data)
- Track representation emergence over response time
- Compare temporal evolution across regions
- Analyze when category information appears

### Interpretability Analysis  
- Identify which embedding dimensions encode categories
- Measure dimension separability and variance
- Analyze correlation structure between dimensions

### Robustness Testing
- Noise resilience (10-50% Gaussian noise)
- Subset stability (50-90% data subsets)
- Distance preservation under perturbations

## 📈 Performance Optimization

### Memory Management
```python
# For large datasets, reduce batch size
config = Config(batch_size=16)  # Instead of 64

# Use gradient accumulation for effective larger batches
# (implement in custom training loop if needed)
```

### Speed Optimization
```python
# GPU acceleration
config = Config(cuda_id=0)

# Efficient data loading
# Use fewer trials for faster preprocessing
neural_data = neural_data[:, :, :5, :]  # Use first 5 trials
```

## 🆘 Troubleshooting

### Common Issues

**Memory Errors**
```python
# Reduce batch size
config = Config(batch_size=16)

# Use CPU if GPU memory insufficient  
config = Config(cuda_id=None)
```

**Import Errors**
```bash
# Ensure packages are in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/neural-repr-learning"

# Or in Python
import sys
sys.path.append('/path/to/neural-repr-learning')
```

**Data Format Issues**
```python
# Check data shapes
print(f"Neural data: {neural_data.shape}")
print(f"Stimulus table: {stim_table.shape}")
print(f"Required: (neurons, images, trials[, time])")

# Verify stimulus table columns
print(f"Columns: {stim_table.columns.tolist()}")
print(f"Required: stim_type, category, unique_img")
```

**CUDA Errors**
```bash
# Check CUDA installation
nvidia-smi

# Install correct PyTorch version
# Visit: https://pytorch.org/get-started/locally/
```

## 📚 Citation

If you use this package in your research, please cite:

```bibtex
@software{neural_repr_learning,
  title={Neural Representation Learning Package},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

[Your License Here]

## 🧠 Research Context

This package was designed to test the hypothesis that higher visual cortical regions develop more structured categorical representations through experience. The contrastive learning approach mirrors how the brain might learn to group similar stimuli and separate different categories.

**Key Research Questions:**
1. Do higher visual areas show better within-category clustering?
2. Are similar categories closer together in higher areas?
3. How do representation geometries differ across the hierarchy?
4. What drives the emergence of categorical structure?

**Experimental Design:**
- Train identical models on neural responses from different brain regions
- Use the same stimulus set across all regions for fair comparison
- Apply proper normalization to compare embedding spaces
- Quantify clustering and separability at each level of the hierarchy

This approach provides a rigorous, quantitative framework for testing hypotheses about representation learning in the visual system.

---

**🎉 Ready to explore neural representations across the visual hierarchy!** 🧠📊🔬
