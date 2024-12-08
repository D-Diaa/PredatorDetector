# Visualization

## Description

This package provides a comprehensive set of tools for analyzing and visualizing dataset features, sequential patterns, and model interpretability, specifically designed for **identifying relevant features in detecting sexual predators in online conversations**. It leverages various feature importance methods and statistical analyses to provide insights into the characteristics of predatory behavior in textual data.

The package supports both conversation-level and author-level analyses, making it suitable for text analysis and user behavior studies in the context of online safety.

## Key Components

### 1. Dataset Analysis (`dataset_analysis.py`) [dataset_analysis.py]

Provides tools for analyzing dataset features and their distributions, crucial for understanding the underlying patterns in online conversations:

*   **Feature correlation analysis:** Identifies relationships between different features, highlighting potential indicators of predatory behavior [dataset_analysis.py].
*   **Distribution statistics:** Calculates mean, standard deviation, skewness, kurtosis, and KL divergence to understand feature distributions and differences between safe and predatory conversations [dataset_analysis.py].
*   **Temporal pattern analysis:** Examines how features change over time within a conversation, revealing evolving patterns [dataset_analysis.py].
*   **Class-wise feature analysis:** Compares feature distributions between conversations flagged as predatory and those that are not, pinpointing key differentiators [dataset_analysis.py].

### 2. Feature Importance Analysis (`feature_importance.py`) [feature_importance.py]

Implements multiple interpretability methods to determine the most relevant features for detecting sexual predators:

*   **SHAP (SHapley Additive exPlanations):** Uses game theory to measure each feature's contribution to model predictions, considering all possible feature combinations. Provides both global and local explanations, highlighting feature interactions crucial for identifying complex predatory patterns [feature_importance.py].
*   **Integrated Gradients (IG):** Attributes importance by accumulating gradients along a path from a baseline to the input, satisfying implementation invariance and providing attribution symmetry. Captures feature interactions implicitly, offering insights into the model's decision-making process [feature_importance.py].
*   **Attention Attribution:** Analyzes attention weights in transformer-based models, revealing which parts of the conversation the model focuses on. Identifies important sequence positions and helps understand temporal dependencies in predatory conversations [feature_importance.py].
*   **Feature Ablation:** Measures feature importance by observing changes in model performance when features are removed. Provides a direct measure of performance impact, identifying redundant features and dependencies, crucial for model optimization in predator detection [feature_importance.py].
*   **Permutation Importance:** Assesses feature importance by randomly shuffling feature values and measuring the impact on model performance. This model-agnostic method maintains feature distributions, accounts for feature interactions, and provides robust importance estimates [feature_importance.py].
*   **Gradient Saliency:** Computes feature importance using input gradients with respect to the model output. Offers computationally efficient local explanations, captures feature sensitivity, and can be smoothed for stability [feature_importance.py].
*   **Convolutional Filter Analysis:** Examines learned filters in convolutional layers to understand pattern detection capabilities, feature hierarchies, learned representations, and filter specialization, relevant for identifying specific linguistic patterns in predatory conversations [feature_importance.py].

## Installation

```bash
pip install visualization-package
```

## Usage

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
    feature_names=dataset.feature_keys,
    output_dir='dataset_analysis/{feature_set}/{model_type}'
)

# 2. Sequential Analysis
sequential_results = analyze_sequential_features(
    dataloader=test_loader,
    feature_names=dataset.feature_keys,
    output_dir='sequence_analysis/{feature_set}/{model_type}'
)

# 3. Feature Importance Analysis
importance_results = analyze_feature_importance(
    model_type='conversation',  # or 'author'
    model_path='models/predator_detection_model.pt',
    dataset_path='data/conversations',
    feature_keys=dataset.feature_keys,
    device='cuda',
    batch_size=64,
    output_dir='feature_importance/{feature_set}/{model_type}'
)

# 4. Visualize Results
visualize_all_results(importance_results, output_dir=f'feature_importance/{{feature_set}}/{{model_type}}')
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
- 'conversation': For conversation-level analysis [feature_importance.py]
- 'author': For author-level analysis [feature_importance.py]

### Dataset Parameters
- max_seq_length: Maximum sequence length (default: 256) [dataset_analysis.py]
- min_seq_length: Minimum sequence length (default: 8) [dataset_analysis.py]
- batch_size: Batch size for data loading (default: 64) [feature_importance.py]

### Analysis Parameters
- num_samples: Number of samples for SHAP analysis (default: 100) [feature_importance.py]
- steps: Number of steps for Integrated Gradients (default: 50) [feature_importance.py]
- smooth_samples: Number of samples for gradient smoothing (default: 50) [feature_importance.py]
- noise_scale: Scale of noise for gradient saliency (default: 0.1) [feature_importance.py]

## Visualization Functions

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
    output_dir=f'feature_importance/{{feature_set}}/{{model_type}}'
)
```

## Output Directory Structure

```
.
├── dataset_analysis/
│   └── {feature_set}/
│       └── {model_type}/
│           ├── distribution_*.png
│           ├── correlation_heatmap_all.png
│           ├── correlation_heatmap_positive.png
│           └── correlation_heatmap_negative.png
├── sequence_analysis/
│   └── {feature_set}/
│       └── {model_type}/
│           ├── temporal_autocorr_all.png
│           ├── temporal_autocorr_positive.png
│           ├── temporal_autocorr_negative.png
│           ├── time_lagged_cross_corr_all.png
│           ├── time_lagged_cross_corr_positive.png
│           ├── time_lagged_cross_corr_negative.png
│           └── trajectory_*.png
└── feature_importance/
    └── {feature_set}/
        └── {model_type}/
            ├── ablation_importance.png
            ├── integrated_gradients_importance.png
            ├── attention_importance.png
            ├── shap_importance.png
            ├── saliency_importance.png
            ├── permutation_importance.png
            ├── conv_filter_importance.png
            ├── method_correlation.png
            └── results.json
```