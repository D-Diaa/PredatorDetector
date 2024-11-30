# Visualization Package

A comprehensive Python package for analyzing and visualizing dataset features, sequential patterns, and model interpretability through various feature importance methods.

## Overview

The Visualization package provides tools for:
- Dataset feature analysis and visualization
- Sequential pattern analysis
- Feature importance computation using multiple methods
- Model interpretability
- Temporal correlation analysis

The package supports both conversation-level and author-level analyses, making it suitable for text analysis and user behavior studies.


## Key Components

### 1. Dataset Analysis (`dataset_analysis.py`)

Provides tools for analyzing dataset features and their distributions:

- Feature correlation analysis
- Distribution statistics
- Temporal pattern analysis
- Class-wise feature analysis

```python
from visualization import analyze_dataset_features

# Analyze dataset features
results = analyze_dataset_features(
    dataloader=train_loader,
    feature_names=dataset.feature_keys,
    output_dir='dataset_analysis'
)
```

The dataset analysis module examines:

#### Distribution Statistics
- Mean and standard deviation: Captures central tendency and spread
- Skewness: Measures asymmetry in feature distributions
- Kurtosis: Indicates the presence of outliers and tail behavior
- KL Divergence: Quantifies differences between class distributions

#### Correlation Analysis
- Pearson correlation between features
- Class-conditional correlations
- Hierarchical clustering of correlated features

These analyses help identify:
- Feature redundancy
- Class-discriminative features
- Data quality issues
- Potential preprocessing needs

#### Sequential Analysis

Analyze temporal patterns and correlations in sequential data:

```python
from visualization import analyze_sequential_features

# Analyze sequential patterns
results = analyze_sequential_features(
    dataloader=test_loader,
    feature_names=dataset.feature_keys,
    output_dir='sequential_analysis'
)
```

Sequential analysis examines:

- Temporal Autocorrelation: Measures how a feature correlates with itself over time
- Cross-feature Temporal Correlations: Identifies relationships between features across time steps
- Trajectory Analysis: Studies how features evolve over sequences
- Pattern Divergence: Quantifies differences in temporal patterns between classes

### 2. Feature Importance Analysis (`feature_importance.py`)

The feature importance module implements multiple interpretability methods, each offering unique insights into model behavior:

#### SHAP (SHapley Additive exPlanations)
SHAP values provide a game-theoretic approach to feature importance. They measure each feature's contribution to model predictions by considering all possible feature combinations. SHAP values:
- Are theoretically grounded in cooperative game theory
- Provide both global and local explanations
- Account for feature interactions
- Are consistent and locally accurate

```python
shap_values = compute_shap_values(
    inputs=batch_data,
    model=trained_model,
    num_samples=100
)
```

#### Integrated Gradients
Integrated Gradients (IG) attributes importance by accumulating gradients along a path from a baseline to the input. This method:
- Satisfies implementation invariance
- Provides attribution symmetry
- Is computationally efficient
- Captures feature interactions implicitly

```python
ig_attributions = compute_integrated_gradients(
    inputs=batch_data,
    model=trained_model,
    device='cuda',
    steps=50
)
```

#### Attention Attribution
Analyzes the attention weights in transformer-based models to understand which input elements the model focuses on. This method:
- Provides insight into model decision-making
- Identifies important sequence positions
- Helps understand temporal dependencies
- Visualizes feature relationships

```python
attn_weights, attn_attribution = compute_attention_attribution(
    inputs=batch_data,
    model=trained_model
)
```

#### Feature Ablation
Measures feature importance by observing changes in model performance when features are removed. This approach:
- Is model-agnostic
- Provides direct performance impact measurement
- Can identify redundant features
- Helps understand feature dependencies

```python
ablation_scores = compute_feature_ablation(
    dataloader=test_loader,
    model=trained_model,
    device='cuda',
    feature_names=feature_names,
    metric_fn=f1_score
)
```

#### Permutation Importance
Assesses feature importance by randomly shuffling feature values and measuring the impact on model performance. This method:
- Is model-agnostic
- Maintains feature distributions
- Accounts for feature interactions
- Provides robust importance estimates

#### Gradient Saliency
Computes feature importance using input gradients with respect to the model output. The method:
- Is computationally efficient
- Provides local explanations
- Captures feature sensitivity
- Can be smoothed for stability

#### Convolutional Filter Analysis
Examines learned filters in convolutional layers to understand:
- Pattern detection capabilities
- Feature hierarchies
- Learned representations
- Filter specialization

## Visualization Functions

The package includes comprehensive visualization tools:

### Feature Importance Visualization
```python
visualize_feature_importance(
    importance_scores=results['ablation'],
    top_k=10,
    title="Feature Ablation Importance"
)
```

### All Results Visualization
```python
visualize_all_results(
    results=all_analysis_results,
    output_dir='feature_importance_plots'
)
```

## Output Directory Structure

```
output/
├── dataset_analysis/
│   ├── feature_correlation_all.png
│   ├── feature_correlation_positive.png
│   ├── feature_correlation_negative.png
│   └── distribution_*.png
├── sequential_analysis/
│   ├── temporal_autocorr_all.png
│   ├── cross_feature_temporal_corr_all.png
│   └── trajectory_*.png
└── feature_importance_plots/
    ├── ablation_importance.png
    ├── integrated_gradients_importance.png
    ├── attention_importance.png
    ├── shap_importance.png
    ├── saliency_importance.png
    ├── permutation_importance.png
    ├── conv_filter_importance.png
    └── method_correlation.png
```

## Example Usage

### Basic Analysis Pipeline

```python
from visualization import (
    analyze_dataset_features,
    analyze_sequential_features,
    analyze_feature_importance
)

# 1. Dataset Analysis
dataset_results = analyze_dataset_features(
    dataloader=train_loader,
    feature_names=dataset.feature_keys
)

# 2. Sequential Analysis
sequential_results = analyze_sequential_features(
    dataloader=test_loader,
    feature_names=dataset.feature_keys
)

# 3. Feature Importance Analysis
importance_results = analyze_feature_importance(
    model_type='conversation',
    model_path='models/model.pt',
    dataset_path='data/dataset'
)

# 4. Visualize Results
visualize_all_results(importance_results)
```

### Custom Analysis

```python
# Custom temporal correlation analysis
temporal_results = temporal_correlation_analysis(
    features_array=features,
    feature_names=feature_names,
    output_dir='output',
    subset='custom'
)

# Custom gradient saliency analysis
saliency_scores = compute_gradient_saliency(
    inputs=batch_data,
    model=model,
    smooth_samples=50,
    noise_scale=0.1
)
```

## Configuration

### Model Types
- 'conversation': For conversation-level analysis
- 'author': For author-level analysis

### Dataset Parameters
- max_seq_length: Maximum sequence length (default: 256)
- min_seq_length: Minimum sequence length (default: 8)
- batch_size: Batch size for data loading (default: 64)

### Analysis Parameters
- num_samples: Number of samples for SHAP analysis (default: 100)
- steps: Number of steps for Integrated Gradients (default: 50)
- smooth_samples: Number of samples for gradient smoothing (default: 50)
- noise_scale: Scale of noise for gradient saliency (default: 0.1)