"""
Neural Representation Learning Package
=====================================
A clean, modular package for learning neural embeddings using contrastive learning.
Supports both temporal and averaged neural data with GPU acceleration.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import random
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
import os
import warnings

# ========================
# DEVICE MANAGEMENT
# ========================

def get_device(cuda_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        cuda_id: Specific CUDA device ID. If None, uses first available GPU or CPU.
        
    Returns:
        torch.device object
    """
    if cuda_id is not None:
        if torch.cuda.is_available() and cuda_id < torch.cuda.device_count():
            device = torch.device(f'cuda:{cuda_id}')
            print(f"🔧 Using GPU: {device} ({torch.cuda.get_device_name(cuda_id)})")
        else:
            print(f"⚠️  CUDA device {cuda_id} not available, falling back to CPU")
            device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"🔧 Using GPU: {device} ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device('cpu')
            print("🔧 Using CPU")
    
    return device

def set_device_context(cuda_id: Optional[int] = None):
    """Set the CUDA context for consistent device usage."""
    device = get_device(cuda_id)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    return device

# ========================
# METADATA CONVERSION
# ========================

def auto_detect_block_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect block column in dataframe."""
    cols = df.columns.tolist()
    
    for possible in ['block', 'session', 'run', 'block_id', 'session_id']:
        if possible in cols:
            unique_vals = df[possible].nunique()
            if 1 < unique_vals < len(df) * 0.5:
                return possible
    
    for col in cols:
        if 'block' in col.lower():
            unique_vals = df[col].nunique()
            if 1 < unique_vals < len(df) * 0.5:
                return col
    return None

def smart_convert(df: pd.DataFrame, block_col: Optional[str] = None) -> Tuple[List[Dict], Dict[int, str], Dict[str, Dict[int, str]]]:
    """Smart conversion with automatic column detection."""
    cols = df.columns.tolist()
    
    # Auto-detect columns
    category_col = None
    for possible in ['category', 'cat', 'condition', 'stimulus']:
        if possible in cols:
            category_col = possible
            break
    
    exemplar_col = None  
    for possible in ['unique_img', 'image', 'exemplar', 'img_id', 'filename']:
        if possible in cols:
            exemplar_col = possible
            break
    
    stim_type_col = 'stim_type'
    for possible in ['stim_type', 'type', 'stimulus_type']:
        if possible in cols:
            stim_type_col = possible
            break
    
    if category_col is None:
        raise ValueError(f"Could not find category column. Available: {cols}")
    if exemplar_col is None:
        raise ValueError(f"Could not find exemplar column. Available: {cols}")
    
    print(f"📋 Auto-detected columns: category='{category_col}', exemplar='{exemplar_col}', stim_type='{stim_type_col}'")
    
    if block_col is None:
        block_col = auto_detect_block_column(df)
    if block_col:
        print(f"   Block column: '{block_col}'")
    
    # Convert to metadata
    metadata_list = []
    image_to_category = {}
    category_mappings = defaultdict(dict)
    
    for idx, row in df.reset_index(drop=True).iterrows():
        category = str(row[category_col]) if pd.notna(row[category_col]) else 'unknown'
        exemplar = str(row[exemplar_col]) if pd.notna(row[exemplar_col]) else f'exemplar_{idx}'
        stim_type = str(row[stim_type_col]) if stim_type_col in df.columns and pd.notna(row[stim_type_col]) else 'unknown'
        
        full_category = f"{stim_type}_{category}" if stim_type != 'unknown' else category
        
        metadata = {
            'image_id': idx,
            'category': category,
            'exemplar': exemplar,
            'image_type': stim_type,
            'full_category': full_category,
            'block': row[block_col] if block_col and block_col in df.columns else None
        }
        
        metadata_list.append(metadata)
        image_to_category[idx] = full_category
        
        if stim_type not in category_mappings:
            category_mappings[stim_type] = {}
        if category not in category_mappings[stim_type]:
            category_mappings[stim_type][len(category_mappings[stim_type])] = category
    
    return metadata_list, image_to_category, dict(category_mappings)

# ========================
# NEURAL DATA PREPROCESSING
# ========================

def preprocess_neural_data(neural_data: np.ndarray, 
                          bin_size: int = 10, 
                          sigma: float = 1.5) -> torch.Tensor:
    """
    Preprocess neural data with temporal binning and smoothing.
    
    Args:
        neural_data: Neural data array (3D or 4D)
        bin_size: Time bin size in milliseconds (for 4D data)
        sigma: Gaussian smoothing sigma
        
    Returns:
        Preprocessed tensor
    """
    if neural_data.ndim == 4:
        # Temporal data (n_neurons, n_images, n_trials, n_timebins)
        n_neurons, n_images, n_trials, n_time = neural_data.shape
        new_time_bins = n_time // bin_size
        
        # Temporal binning
        trimmed = neural_data[:, :, :, :new_time_bins * bin_size]
        binned = trimmed.reshape(n_neurons, n_images, n_trials, new_time_bins, bin_size).sum(axis=-1)
        
        # Gaussian smoothing
        if sigma > 0:
            smoothed = np.zeros_like(binned, dtype=np.float32)
            for n in range(n_neurons):
                smoothed[n] = gaussian_filter1d(binned[n].astype(np.float32), sigma=sigma, axis=-1)
        else:
            smoothed = binned.astype(np.float32)
        
        # Convert to firing rate and apply sqrt transform
        firing_rate = smoothed / (bin_size / 1000.0)
        processed = np.sqrt(np.maximum(0.0, firing_rate + 0.01))
        
    elif neural_data.ndim == 3:
        # Averaged data (n_neurons, n_images, n_trials)
        processed = neural_data.astype(np.float32)
        if sigma > 0:
            for i in range(neural_data.shape[1]):
                processed[:, i, :] = gaussian_filter1d(processed[:, i, :], sigma=sigma, axis=0)
        processed = np.sqrt(np.maximum(0.0, processed + 0.01))
    else:
        raise ValueError(f"Invalid neural_data shape: {neural_data.shape}")
    
    return torch.FloatTensor(processed)

# ========================
# CONFIGURATION
# ========================

@dataclass
class Config:
    """Configuration for neural representation learning."""
    # Model parameters
    hidden_dims: List[int] = None
    embedding_dim: int = 128
    activation: str = 'relu'
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.1
    
    # Preprocessing parameters
    bin_size: int = 10
    smoothing_sigma: float = 1.5
    normalize_method: str = 'zscore'
    
    # Sampling parameters
    balanced_sampling: bool = True
    exclude_blanks: bool = True
    
    # Device
    cuda_id: Optional[int] = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]

# ========================
# DATASET
# ========================

class NeuralDataset(Dataset):
    """Dataset for neural responses with contrastive learning support."""
    
    def __init__(self, 
                 neural_data: torch.Tensor,
                 metadata: List[Dict],
                 config: Config,
                 mode: str = 'train'):
        """
        Initialize dataset.
        
        Args:
            neural_data: Preprocessed neural data tensor
            metadata: List of metadata dictionaries
            config: Configuration object
            mode: 'train' or 'test'
        """
        self.neural_data = neural_data
        self.metadata = metadata
        self.config = config
        self.mode = mode
        self.is_temporal = neural_data.ndim == 4
        
        # Build sample structure
        self._build_samples()
        self._create_category_mappings()
        
        print(f"📊 {mode} dataset: {len(self)} {'trials' if self.is_temporal else 'samples'}")
        print(f"   Categories: {len(self.category_to_samples)} unique")
    
    def _build_samples(self):
        """Build sample list based on data format."""
        self.samples = []
        
        if self.is_temporal:
            # Individual trials for temporal data
            n_images, n_trials = self.neural_data.shape[1], self.neural_data.shape[2]
            for img_idx in range(n_images):
                for trial_idx in range(n_trials):
                    sample = {
                        'image_idx': img_idx,
                        'trial_idx': trial_idx,
                        'category': self.metadata[img_idx]['full_category'],
                        'metadata': self.metadata[img_idx]
                    }
                    self.samples.append(sample)
        else:
            # Average across trials for non-temporal data
            n_images = self.neural_data.shape[1]
            for img_idx in range(n_images):
                sample = {
                    'image_idx': img_idx,
                    'category': self.metadata[img_idx]['full_category'],
                    'metadata': self.metadata[img_idx]
                }
                self.samples.append(sample)
    
    def _create_category_mappings(self):
        """Create category mappings for balanced sampling."""
        self.category_to_samples = defaultdict(list)
        
        for idx, sample in enumerate(self.samples):
            category = sample['category']
            self.category_to_samples[category].append(idx)
        
        # Calculate weights for balanced sampling
        total_samples = len(self.samples)
        self.sample_weights = []
        
        for sample in self.samples:
            category = sample['category']
            category_size = len(self.category_to_samples[category])
            weight = total_samples / (len(self.category_to_samples) * category_size)
            self.sample_weights.append(weight)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_idx = sample['image_idx']
        
        if self.is_temporal:
            # Get specific trial
            trial_idx = sample['trial_idx']
            response = self.neural_data[:, image_idx, trial_idx, :].clone()
        else:
            # Average across trials or get single response
            if self.neural_data.ndim == 3:
                response = torch.FloatTensor(self.neural_data[:, image_idx, :]).mean(dim=1)
            else:
                response = self.neural_data[:, image_idx].clone()
        
        # Simple augmentation for training
        if self.mode == 'train' and random.random() < 0.2:
            response += torch.randn_like(response) * 0.02
        
        return {
            'response': response,
            'category': sample['category'],
            'metadata': sample['metadata']
        }

# ========================
# MODEL
# ========================

class ContrastiveEncoder(nn.Module):
    """Neural encoder for contrastive learning."""
    
    def __init__(self, input_dim: int, config: Config):
        super().__init__()
        
        # Build backbone
        layers = []
        dims = [input_dim] + config.hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
        
        self.backbone = nn.Sequential(*layers)
        self.projection = nn.Linear(config.hidden_dims[-1], config.embedding_dim)
    
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        return embeddings

# ========================
# LOSS FUNCTION
# ========================

class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, categories: List[str]) -> torch.Tensor:
        batch_size = embeddings.size(0)
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create masks
        positive_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=embeddings.device)
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and categories[i] == categories[j]:
                    positive_mask[i, j] = True
        
        # Compute loss
        losses = []
        for i in range(batch_size):
            if positive_mask[i].sum() == 0:
                continue
            
            pos_similarities = similarity[i][positive_mask[i]]
            neg_similarities = similarity[i][~positive_mask[i]]
            
            if len(neg_similarities) == 0:
                continue
            
            all_similarities = torch.cat([pos_similarities, neg_similarities])
            numerator = torch.logsumexp(pos_similarities, dim=0)
            denominator = torch.logsumexp(all_similarities, dim=0)
            loss = denominator - numerator
            losses.append(loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0., device=embeddings.device, requires_grad=True)

# ========================
# MAIN LEARNER CLASS
# ========================

class NeuralRepresentationLearner:
    """Main class for neural representation learning."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = set_device_context(config.cuda_id)
        self.model = None
        self.optimizer = None
        self.criterion = None
    
    def create_datasets_from_stim_table(self, 
                                      neural_data: np.ndarray,
                                      stim_table: pd.DataFrame,
                                      test_size: float = 0.2) -> Tuple[NeuralDataset, NeuralDataset, Dict]:
        """
        Create train/test datasets from neural data and stimulus table.
        
        Args:
            neural_data: Neural response array
            stim_table: Stimulus table DataFrame
            test_size: Test set proportion
            
        Returns:
            train_dataset, test_dataset, conversion_info
        """
        print("🔄 Converting stimulus table and creating datasets...")
        
        # Convert stimulus table
        metadata_list, image_to_category, category_mappings = smart_convert(stim_table)
        
        # Preprocess neural data
        processed_data = preprocess_neural_data(neural_data, self.config.bin_size, self.config.smoothing_sigma)
        
        # Validate alignment
        if len(metadata_list) != neural_data.shape[1]:
            raise ValueError(f"Metadata length {len(metadata_list)} != neural data images {neural_data.shape[1]}")
        
        # Stratified split at image level
        categories = [meta['full_category'] for meta in metadata_list]
        train_idx, test_idx = train_test_split(
            range(len(metadata_list)), 
            test_size=test_size, 
            stratify=categories, 
            random_state=42
        )
        
        print(f"📊 Split: {len(train_idx)} train, {len(test_idx)} test images")
        
        # Create datasets
        train_data = processed_data[:, train_idx, ...]
        test_data = processed_data[:, test_idx, ...]
        train_metadata = [metadata_list[i] for i in train_idx]
        test_metadata = [metadata_list[i] for i in test_idx]
        
        train_dataset = NeuralDataset(train_data, train_metadata, self.config, 'train')
        test_dataset = NeuralDataset(test_data, test_metadata, self.config, 'test')
        
        conversion_info = {
            'image_to_category': image_to_category,
            'category_mappings': category_mappings,
            'train_indices': train_idx,
            'test_indices': test_idx
        }
        
        return train_dataset, test_dataset, conversion_info
    
    def create_dataloader(self, dataset: NeuralDataset, shuffle: bool = True) -> DataLoader:
        """Create DataLoader with optional balanced sampling."""
        if self.config.balanced_sampling and shuffle:
            sampler = WeightedRandomSampler(
                weights=dataset.sample_weights,
                num_samples=len(dataset),
                replacement=True
            )
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                drop_last=True
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                drop_last=shuffle
            )
    
    def initialize_model(self, input_dim: int):
        """Initialize model, optimizer, and loss function."""
        self.model = ContrastiveEncoder(input_dim, self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.criterion = InfoNCELoss(self.config.temperature)
        
        print(f"🔧 Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            responses = batch['response'].to(self.device)
            categories = batch['category']
            
            # Forward pass
            embeddings = self.model(responses)
            loss = self.criterion(embeddings, categories)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def extract_embeddings(self, dataloader: DataLoader) -> Tuple[np.ndarray, List[str], List[Dict]]:
        """Extract embeddings for analysis."""
        self.model.eval()
        all_embeddings = []
        all_categories = []
        all_metadata = []
        
        with torch.no_grad():
            for batch in dataloader:
                responses = batch['response'].to(self.device)
                embeddings = self.model(responses)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_categories.extend(batch['category'])
                all_metadata.extend(batch['metadata'])
        
        embeddings_array = np.vstack(all_embeddings)
        return embeddings_array, all_categories, all_metadata
    
    def save_model(self, filepath: str):
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'device': str(self.device)
        }, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str, input_dim: int):
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.model is None:
            self.initialize_model(input_dim)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"📂 Model loaded from {filepath}")

# ========================
# UNIT TESTS
# ========================

def create_test_data():
    """Create test data for unit tests."""
    np.random.seed(42)
    
    # Test neural data (temporal)
    n_neurons, n_images, n_trials, n_timebins = 20, 50, 5, 100
    neural_data = np.random.poisson(0.1, (n_neurons, n_images, n_trials, n_timebins))
    
    # Test stimulus table
    stim_types = ['natural', 'texture', 'noise']
    categories = ['cat1', 'cat2', 'cat3']
    
    stim_table = pd.DataFrame({
        'stim_type': np.random.choice(stim_types, n_images),
        'category': np.random.choice(categories, n_images),
        'unique_img': [f'img_{i:03d}.jpg' for i in range(n_images)],
        'block': np.random.randint(1, 4, n_images)
    })
    
    return neural_data, stim_table

def test_device_management():
    """Test device management functions."""
    print("🧪 Testing device management...")
    
    device = get_device()
    assert isinstance(device, torch.device)
    
    device_ctx = set_device_context()
    assert isinstance(device_ctx, torch.device)
    
    print("   ✅ Device management tests passed")

def test_metadata_conversion():
    """Test metadata conversion functions."""
    print("🧪 Testing metadata conversion...")
    
    neural_data, stim_table = create_test_data()
    
    # Test smart_convert
    metadata_list, image_to_category, category_mappings = smart_convert(stim_table)
    
    assert len(metadata_list) == len(stim_table)
    assert len(image_to_category) == len(stim_table)
    assert isinstance(category_mappings, dict)
    
    print("   ✅ Metadata conversion tests passed")

def test_preprocessing():
    """Test neural data preprocessing."""
    print("🧪 Testing preprocessing...")
    
    neural_data, _ = create_test_data()
    
    # Test 4D data
    processed_4d = preprocess_neural_data(neural_data, bin_size=10, sigma=1.0)
    assert isinstance(processed_4d, torch.Tensor)
    assert processed_4d.ndim == 4
    
    # Test 3D data
    neural_3d = neural_data.mean(axis=3)  # Average across time
    processed_3d = preprocess_neural_data(neural_3d, sigma=1.0)
    assert isinstance(processed_3d, torch.Tensor)
    assert processed_3d.ndim == 3
    
    print("   ✅ Preprocessing tests passed")

def test_dataset():
    """Test NeuralDataset class."""
    print("🧪 Testing dataset...")
    
    neural_data, stim_table = create_test_data()
    
    # Create config
    config = Config(batch_size=8, embedding_dim=32)
    
    # Test preprocessing and dataset creation
    processed_data = preprocess_neural_data(neural_data)
    metadata_list, _, _ = smart_convert(stim_table)
    
    dataset = NeuralDataset(processed_data, metadata_list, config, 'train')
    
    assert len(dataset) > 0
    assert len(dataset.category_to_samples) > 0
    
    # Test data loading
    sample = dataset[0]
    assert 'response' in sample
    assert 'category' in sample
    assert isinstance(sample['response'], torch.Tensor)
    
    print("   ✅ Dataset tests passed")

def test_model():
    """Test model and training."""
    print("🧪 Testing model and training...")
    
    neural_data, stim_table = create_test_data()
    
    config = Config(
        hidden_dims=[32, 16],
        embedding_dim=8,
        batch_size=4,
        learning_rate=1e-3
    )
    
    learner = NeuralRepresentationLearner(config)
    
    # Create datasets
    train_ds, test_ds, _ = learner.create_datasets_from_stim_table(neural_data, stim_table, test_size=0.3)
    
    # Create dataloaders
    train_loader = learner.create_dataloader(train_ds, shuffle=True)
    test_loader = learner.create_dataloader(test_ds, shuffle=False)
    
    # Initialize model
    input_dim = neural_data.shape[0]
    learner.initialize_model(input_dim)
    
    # Test training
    initial_loss = learner.train_epoch(train_loader)
    assert isinstance(initial_loss, float)
    
    # Test embedding extraction
    embeddings, categories, metadata = learner.extract_embeddings(test_loader)
    assert isinstance(embeddings, np.ndarray)
    assert len(categories) == len(metadata)
    assert embeddings.shape[0] == len(categories)
    
    print("   ✅ Model and training tests passed")

def test_full_pipeline():
    """Test complete pipeline."""
    print("🧪 Testing full pipeline...")
    
    neural_data, stim_table = create_test_data()
    
    config = Config(
        hidden_dims=[32, 16],
        embedding_dim=8,
        batch_size=4,
        cuda_id=None  # Use CPU for testing
    )
    
    learner = NeuralRepresentationLearner(config)
    train_ds, test_ds, info = learner.create_datasets_from_stim_table(neural_data, stim_table)
    
    train_loader = learner.create_dataloader(train_ds)
    test_loader = learner.create_dataloader(test_ds, shuffle=False)
    
    learner.initialize_model(neural_data.shape[0])
    
    # Train for a few epochs
    for epoch in range(2):
        loss = learner.train_epoch(train_loader)
        print(f"      Epoch {epoch+1}: Loss = {loss:.4f}")
    
    # Extract embeddings
    embeddings, categories, metadata = learner.extract_embeddings(test_loader)
    
    assert embeddings.shape[1] == config.embedding_dim
    print(f"      Final embeddings shape: {embeddings.shape}")
    
    print("   ✅ Full pipeline test passed")

def run_all_tests():
    """Run all unit tests."""
    print("🚀 Running Neural Representation Learning Package Tests")
    print("=" * 60)
    
    try:
        test_device_management()
        test_metadata_conversion()
        test_preprocessing()
        test_dataset()
        test_model()
        test_full_pipeline()
        
        print("=" * 60)
        print("🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ========================
# EXAMPLE USAGE
# ========================

if __name__ == "__main__":
    # Run unit tests
    if run_all_tests():
        print("\n" + "=" * 60)
        print("📖 EXAMPLE USAGE")
        print("=" * 60)
        
        # Example usage
        neural_data, stim_table = create_test_data()
        
        config = Config(
            hidden_dims=[64, 32],
            embedding_dim=16,
            batch_size=8,
            learning_rate=1e-3,
            cuda_id=0 if torch.cuda.is_available() else None
        )
        
        learner = NeuralRepresentationLearner(config)
        train_ds, test_ds, info = learner.create_datasets_from_stim_table(neural_data, stim_table)
        
        train_loader = learner.create_dataloader(train_ds)
        test_loader = learner.create_dataloader(test_ds, shuffle=False)
        
        learner.initialize_model(neural_data.shape[0])
        
        print("\n🏋️ Training model...")
        for epoch in range(3):
            loss = learner.train_epoch(train_loader)
            print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
        
        print("\n📊 Extracting embeddings...")
        embeddings, categories, metadata = learner.extract_embeddings(test_loader)
        
        print(f"Final embeddings shape: {embeddings.shape}")
        print(f"Categories found: {set(categories)}")
        
        print("\n🎯 Package ready for use!")
