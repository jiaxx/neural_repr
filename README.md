# Neural Representation Learning Project

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
