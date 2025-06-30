#!/usr/bin/env python3
"""
Setup script for Neural Representation Learning Package
======================================================

This script sets up the complete neural representation learning environment
and validates that everything is working correctly.

Usage:
    python setup.py

This will:
1. Check environment and dependencies
2. Validate package functionality
3. Run basic tests
4. Create example directory structure
"""

import os
import sys
import subprocess
import importlib
import numpy as np
import torch
import pandas as pd
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        ('numpy', '1.20.0'),
        ('torch', '1.12.0'),
        ('pandas', '1.3.0'),
        ('scikit-learn', '1.0.0'),
        ('matplotlib', '3.5.0'),
        ('seaborn', '0.11.0'),
        ('scipy', '1.7.0'),
        ('jupyter', None)
    ]
    
    missing_packages = []
    
    for package, min_version in required_packages:
        try:
            module = importlib.import_module(package)
            
            if hasattr(module, '__version__'):
                version = module.__version__
                print(f"✅ {package}: {version}")
                
                if min_version and version < min_version:
                    print(f"⚠️  Warning: {package} version {version} < {min_version}")
            else:
                print(f"✅ {package}: Available")
                
        except ImportError:
            print(f"❌ {package}: Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print("   conda env create -f environment.yml")
        print("   # OR")
        print("   pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_gpu_support():
    """Check GPU availability and CUDA support."""
    print("\n🔧 Checking GPU support...")
    
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"✅ CUDA available with {n_gpus} GPU(s)")
        
        for i in range(n_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"   GPU {i}: {gpu_name}")
            
            # Check GPU memory
            if hasattr(torch.cuda, 'get_device_properties'):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"           Memory: {memory_gb:.1f} GB")
        
        return True
    else:
        print("⚠️  CUDA not available - will use CPU")
        print("   For GPU acceleration, install PyTorch with CUDA support")
        return False

def test_package_functionality():
    """Test basic package functionality."""
    print("\n🧪 Testing package functionality...")
    
    try:
        # Test imports
        print("   Testing imports...")
        sys.path.append('.')  # Add current directory to path
        
        from neural_repr_package import NeuralRepresentationLearner, Config
        from neural_evaluation import NeuralEmbeddingEvaluator, EvaluationConfig
        print("   ✅ Package imports successful")
        
        # Test basic configuration
        print("   Testing configuration...")
        config = Config(
            hidden_dims=[32, 16],
            embedding_dim=8,
            batch_size=4,
            cuda_id=None  # Use CPU for testing
        )
        print("   ✅ Configuration creation successful")
        
        # Test data creation
        print("   Testing data creation...")
        np.random.seed(42)
        neural_data = np.random.poisson(0.1, (10, 20, 5, 50))  # Small test data
        stim_table = pd.DataFrame({
            'stim_type': ['natural'] * 10 + ['texture'] * 10,
            'category': ['cat1'] * 5 + ['cat2'] * 5 + ['cat3'] * 5 + ['cat4'] * 5,
            'unique_img': [f'img_{i:03d}.jpg' for i in range(20)]
        })
        print("   ✅ Test data creation successful")
        
        # Test model initialization
        print("   Testing model initialization...")
        learner = NeuralRepresentationLearner(config)
        train_ds, test_ds, info = learner.create_datasets_from_stim_table(neural_data, stim_table)
        learner.initialize_model(neural_data.shape[0])
        print("   ✅ Model initialization successful")
        
        # Test evaluation
        print("   Testing evaluation...")
        eval_config = EvaluationConfig(n_clusters_range=(2, 4), cv_folds=2)
        evaluator = NeuralEmbeddingEvaluator(eval_config)
        print("   ✅ Evaluation setup successful")
        
        print("✅ All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_directory_structure():
    """Create recommended directory structure."""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "results",
        "results/models",
        "results/embeddings", 
        "results/plots",
        "results/cross_region",
        "results/advanced_analysis",
        "notebooks",
        "scripts",
        "configs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created: {directory}/")
    
    # Create example config file
    example_config = """# Example configuration file
# Copy and modify for your experiments

# Model Configuration
hidden_dims: [128, 64]
embedding_dim: 32
activation: relu
dropout: 0.1

# Training Configuration
batch_size: 32
learning_rate: 0.001
weight_decay: 0.0001
temperature: 0.1
num_epochs: 50

# Data Configuration
bin_size: 10
smoothing_sigma: 1.5
balanced_sampling: true
exclude_blanks: true

# Hardware Configuration
cuda_id: 0  # Set to null for CPU

# Evaluation Configuration
n_clusters_range: [2, 20]
cv_folds: 5
save_plots: true
"""
    
    with open("configs/example_config.yaml", 'w') as f:
        f.write(example_config)
    
    print("   ✅ Created example configuration file")
    
    # Create README
    readme_content = """# Neural Representation Learning Project

## Directory Structure

- `data/`: Neural data and stimulus tables
  - `raw/`: Original data files
  - `processed/`: Preprocessed data
- `results/`: All analysis outputs
  - `models/`: Trained model files
  - `embeddings/`: Extracted embeddings
  - `plots/`: Generated visualizations
  - `cross_region/`: Cross-region comparison results
  - `advanced_analysis/`: Advanced evaluation results
- `notebooks/`: Jupyter notebooks for analysis
- `scripts/`: Python scripts for batch processing
- `configs/`: Configuration files

## Getting Started

1. Place your neural data in `data/raw/`
2. Open `notebooks/01_basic_usage.ipynb`
3. Modify the data loading section for your data format
4. Run the notebooks in order:
   - 01_basic_usage.ipynb
   - 02_cross_region_comparison.ipynb  
   - 03_advanced_evaluation.ipynb

## Configuration

See `configs/example_config.yaml` for parameter options.

## Results

All results are automatically saved to the `results/` directory with timestamps.
"""
    
    with open("README.md", 'w') as f:
        f.write(readme_content)
    
    print("   ✅ Created README.md")
    
    return True

def create_example_script():
    """Create an example script for batch processing."""
    print("\n📜 Creating example scripts...")
    
    batch_script = '''#!/usr/bin/env python3
"""
Example batch processing script for multiple brain regions.
Modify this script to process your actual data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from neural_repr_package import NeuralRepresentationLearner, Config
from neural_evaluation import NeuralEmbeddingEvaluator, EvaluationConfig

def main():
    """Main batch processing function."""
    
    # Configuration
    config = Config(
        hidden_dims=[128, 64],
        embedding_dim=32,
        batch_size=32,
        learning_rate=1e-3,
        cuda_id=0  # Adjust for your setup
    )
    
    eval_config = EvaluationConfig(
        n_clusters_range=(2, 15),
        cv_folds=5,
        save_plots=True
    )
    
    # Process multiple regions
    regions = ['V1', 'V2', 'V4']  # Modify for your regions
    
    results = {}
    
    for region in regions:
        print(f"\\n🧠 Processing {region}...")
        
        # Load data (modify for your data format)
        neural_data = load_neural_data(region)  # Implement this function
        stim_table = load_stimulus_table()      # Implement this function
        
        # Train model
        learner = NeuralRepresentationLearner(config)
        train_ds, test_ds, info = learner.create_datasets_from_stim_table(
            neural_data, stim_table
        )
        
        train_loader = learner.create_dataloader(train_ds)
        test_loader = learner.create_dataloader(test_ds, shuffle=False)
        
        learner.initialize_model(neural_data.shape[0])
        
        # Training loop
        for epoch in range(25):  # Adjust number of epochs
            loss = learner.train_epoch(train_loader)
            if epoch % 5 == 0:
                print(f"   Epoch {epoch+1}: Loss = {loss:.4f}")
        
        # Extract embeddings
        embeddings, categories, metadata = learner.extract_embeddings(test_loader)
        
        # Save model and embeddings
        learner.save_model(f"results/models/{region}_model.pth")
        np.savez(f"results/embeddings/{region}_embeddings.npz",
                embeddings=embeddings, categories=categories)
        
        # Individual evaluation
        evaluator = NeuralEmbeddingEvaluator(eval_config)
        region_results = evaluator.evaluate_all(
            embeddings, categories, metadata,
            save_dir=f"results/{region}_analysis"
        )
        
        results[region] = {
            'embeddings': embeddings,
            'categories': categories,
            'metadata': metadata,
            'evaluation': region_results
        }
        
        print(f"   ✅ {region} processing completed")
    
    # Cross-region comparison
    print(f"\\n🔄 Running cross-region comparison...")
    
    embeddings_dict = {region: results[region]['embeddings'] for region in regions}
    categories_dict = {region: results[region]['categories'] for region in regions}
    metadata_dict = {region: results[region]['metadata'] for region in regions}
    
    cross_results = evaluator.evaluate_cross_region(
        embeddings_dict, categories_dict, metadata_dict,
        normalization_method='unit_sphere',
        save_dir="results/cross_region_analysis"
    )
    
    # Print final summary
    evaluator.print_cross_region_summary(cross_results)
    
    print(f"\\n🎉 Batch processing completed!")
    print(f"Results saved to: results/")

def load_neural_data(region):
    """
    Load neural data for a specific region.
    
    MODIFY THIS FUNCTION for your data format.
    
    Should return array of shape:
    - (neurons, images, trials, timebins) for temporal data
    - (neurons, images, trials) for trial-averaged data
    """
    # Example placeholder - replace with your data loading
    data_path = f"data/raw/{region}_neural_data.npy"
    
    if Path(data_path).exists():
        return np.load(data_path)
    else:
        print(f"Warning: {data_path} not found, using synthetic data")
        # Return synthetic data as fallback
        return np.random.poisson(0.1, (100, 200, 10, 100))

def load_stimulus_table():
    """
    Load stimulus table.
    
    MODIFY THIS FUNCTION for your data format.
    
    Should return DataFrame with columns:
    - stim_type, category, unique_img, (optional: block)
    """
    stim_path = "data/raw/stimulus_table.csv"
    
    if Path(stim_path).exists():
        return pd.read_csv(stim_path)
    else:
        print(f"Warning: {stim_path} not found, using synthetic data")
        # Return synthetic data as fallback
        return pd.DataFrame({
            'stim_type': ['natural'] * 100 + ['texture'] * 100,
            'category': ['cat1', 'cat2'] * 100,
            'unique_img': [f'img_{i:03d}.jpg' for i in range(200)]
        })

if __name__ == "__main__":
    main()
'''
    
    with open("scripts/batch_process.py", 'w') as f:
        f.write(batch_script)
    
    print("   ✅ Created batch_process.py")
    
    # Make it executable
    try:
        os.chmod("scripts/batch_process.py", 0o755)
    except:
        pass  # Ignore on Windows
    
    return True

def print_final_instructions():
    """Print final setup instructions."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. 📂 Place your neural data in data/raw/")
    print("2. 📓 Open Jupyter: jupyter notebook")
    print("3. 🚀 Start with: notebooks/01_basic_usage.ipynb")
    print("4. 🔄 Continue with: notebooks/02_cross_region_comparison.ipynb")
    print("5. 🔬 Advanced analysis: notebooks/03_advanced_evaluation.ipynb")
    
    print("\n📁 Directory Structure:")
    print("   data/         - Your neural data and stimulus tables")
    print("   notebooks/    - Interactive Jupyter notebooks")
    print("   results/      - All analysis outputs")
    print("   scripts/      - Batch processing scripts")
    print("   configs/      - Configuration files")
    
    print("\n🔧 Configuration:")
    print("   - Edit configs/example_config.yaml for your setup")
    print("   - Adjust CUDA settings based on your hardware")
    print("   - Modify data loading functions in scripts/")
    
    print("\n💡 Tips:")
    print("   - Start with small datasets to test the pipeline")
    print("   - Use GPU acceleration for faster training")
    print("   - Save intermediate results for reproducibility")
    print("   - Check README.md for detailed instructions")
    
    print("\n🆘 If you encounter issues:")
    print("   - Check that all dependencies are installed")
    print("   - Verify your data format matches expected structure")
    print("   - Run the unit tests: python test_neural_package.py")
    print("   - Reduce batch size if you get memory errors")
    
    print("\n🧠 For your neural representation research:")
    print("   - Train separate models for each brain region")
    print("   - Use cross-region comparison to test your hypotheses")
    print("   - Apply advanced evaluation for detailed analysis")
    print("   - Compare clustering quality across the visual hierarchy")
    
    print("\n" + "="*60)
    print("Ready to explore neural representations! 🚀🧠📊")
    print("="*60)

def main():
    """Main setup function."""
    print("🚀 Neural Representation Learning Package Setup")
    print("="*60)
    
    success = True
    
    # Run all checks
    success &= check_python_version()
    success &= check_dependencies()
    gpu_available = check_gpu_support()
    success &= test_package_functionality()
    success &= create_directory_structure()
    success &= create_example_script()
    
    if success:
        print_final_instructions()
        return 0
    else:
        print("\n❌ Setup failed. Please fix the issues above and try again.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
