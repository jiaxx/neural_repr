# Neural Representation Learning Package - Setup Guide

## 🚀 Quick Setup

### Option 1: With GPU Support (Recommended)

```bash
# Clone or download the package files
# Make sure you have conda/mamba installed

# Create environment
conda env create -f environment.yml

# Activate environment
conda activate neural-repr-learning

# Verify PyTorch GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
```

### Option 2: CPU Only

```bash
# Create CPU-only environment
conda env create -f environment-cpu.yml

# Activate environment
conda activate neural-repr-learning-cpu
```

### Option 3: Using pip (if you prefer)

```bash
# Create virtual environment
python -m venv neural-repr-env
source neural-repr-env/bin/activate  # On Windows: neural-repr-env\Scripts\activate

# Install PyTorch (visit pytorch.org for your specific setup)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy scipy pandas scikit-learn matplotlib seaborn plotly umap-learn jupyter tqdm statsmodels
```

## 📁 Package Structure

Organize your files like this:

```
neural-repr-learning/
│
├── environment.yml
├── environment-cpu.yml
├── neural_repr_package.py          # Main training package
├── neural_evaluation.py            # Evaluation module
├── test_neural_package.py          # Unit tests
│
├── notebooks/
│   ├── 01_basic_usage.ipynb
│   ├── 02_cross_region_comparison.ipynb
│   └── 03_advanced_evaluation.ipynb
│
├── data/                           # Your neural data
│   ├── v1_neural_data.npy
│   ├── v2_neural_data.npy
│   └── stimulus_table.csv
│
└── results/                        # Output directory
    ├── models/
    ├── embeddings/
    └── plots/
```

## 🧪 Verify Installation

Run the test suite to make sure everything works:

```bash
# Activate your environment
conda activate neural-repr-learning

# Run unit tests
python test_neural_package.py

# Start Jupyter
jupyter notebook
```

## 🔧 Troubleshooting

### CUDA Issues
```bash
# Check CUDA version
nvidia-smi

# Install specific PyTorch version for your CUDA
# Visit: https://pytorch.org/get-started/locally/

# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Memory Issues
If you encounter memory issues:

```python
# In your code, adjust batch sizes
config = Config(
    batch_size=16,  # Reduce from default 64
    cuda_id=0
)

# Or use CPU
config = Config(
    batch_size=64,
    cuda_id=None  # Use CPU
)
```

### Import Issues
Make sure the package files are in your Python path:

```python
import sys
sys.path.append('/path/to/neural-repr-learning')

from neural_repr_package import NeuralRepresentationLearner, Config
from neural_evaluation import NeuralEmbeddingEvaluator, EvaluationConfig
```

## 📊 Expected Performance

### Training Time (approximate)
- **CPU**: 5-10 minutes per epoch (small dataset)
- **GPU**: 30-60 seconds per epoch (small dataset)
- **Large dataset**: Scale accordingly

### Memory Requirements
- **Minimum**: 8GB RAM, 4GB GPU memory
- **Recommended**: 16GB RAM, 8GB GPU memory
- **Large datasets**: 32GB RAM, 16GB GPU memory

## 🎯 Next Steps

1. **Start with basic usage**: Open `notebooks/01_basic_usage.ipynb`
2. **Load your data**: Adapt the data loading examples to your format
3. **Train models**: Follow the cross-region comparison notebook
4. **Analyze results**: Use the evaluation tools

## 💡 Tips for Your Research

### Data Organization
```python
# Organize your neural data consistently
neural_data_v1.shape  # (neurons, images, trials, timebins) or (neurons, images, trials)
neural_data_v2.shape  # Same format across regions
stim_table.columns    # ['stim_type', 'category', 'unique_img', 'block']
```

### Experiment Tracking
```python
# Optional: Use wandb for experiment tracking
import wandb

wandb.init(project="neural-repr-learning", 
          config={"region": "V1", "embedding_dim": 64})

# Log metrics during training
wandb.log({"epoch": epoch, "loss": loss})
```

### Batch Processing Multiple Regions
```python
# Process multiple regions efficiently
regions = ['V1', 'V2', 'V4']
results = {}

for region in regions:
    print(f"Processing {region}...")
    
    # Load region-specific data
    neural_data = load_neural_data(region)
    
    # Train model
    learner = NeuralRepresentationLearner(config)
    # ... training code ...
    
    # Save results
    results[region] = learner.extract_embeddings(dataloader)
    learner.save_model(f"models/{region}_model.pth")

# Compare across regions
evaluator = NeuralEmbeddingEvaluator()
cross_results = evaluator.evaluate_cross_region(
    {region: results[region][0] for region in regions},  # embeddings
    {region: results[region][1] for region in regions},  # categories  
    {region: results[region][2] for region in regions},  # metadata
    save_dir="results/cross_region"
)
```

## 🆘 Getting Help

1. **Check the example notebooks** - Most common usage patterns are covered
2. **Run unit tests** - Verify your installation is working
3. **Check configuration** - Many issues are due to incorrect config parameters
4. **GPU memory** - Try reducing batch size or switching to CPU
5. **Data format** - Ensure your data matches the expected format

Happy researching! 🧠🔬