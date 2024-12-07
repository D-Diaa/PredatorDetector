# Training Package

A PyTorch-based package for training and evaluating sequence classification models with a focus on conversation and author profiling tasks.

## Overview

The Training package provides a comprehensive framework for:
- Training and evaluating sequence classifiers for conversation-level analysis
- Building author profile classifiers using aggregated conversation data
- Supporting both single-sequence and multi-sequence classification tasks
- Implementing advanced model architectures with transformers and multi-scale feature extraction

## Key Features

- **Flexible Model Architecture**
  - Transformer-based sequence encoding
  - Multi-scale feature aggregation using conv1d layers
  - Residual connections and layer normalization
  - Customizable model hyperparameters

- **Advanced Training Features**
  - Class imbalance handling with multiple weighting strategies
  - Dynamic threshold optimization
  - Automated model checkpointing
  - Learning rate scheduling
  - Comprehensive metrics tracking

- **Multi-level Classification**
  - Conversation-level classification
  - Author-level sequence classification
  - Profile-level aggregation with multiple strategies


## Model Architecture

### SequenceClassifier

The core model architecture consists of:

1. **Transformer Encoder**
   - Input projection layer
   - Positional encoding
   - Multi-head attention layers
   - Feed-forward networks

2. **Multi-scale Feature Extraction**
   - Multiple conv1d layers with different kernel sizes
   - Adaptive max pooling
   - Feature concatenation

3. **Classification Head**
   - Residual blocks
   - Layer normalization
   - Dropout regularization
   - Final linear projection

```python
# Initialize a sequence classifier
model = SequenceClassifier(
    input_size=256,          # Input feature dimension
    hidden_size=256,         # Hidden layer dimension
    num_layers=2,            # Number of transformer layers
    num_heads=8,             # Number of attention heads
    dropout=0.3,             # Dropout rate
    kernel_sizes=[3, 5, 7]   # Conv1d kernel sizes
)
```

### ProfileClassifier

The profile classifier extends the sequence classifier for multi-sequence analysis:

- Uses a pretrained sequence classifier
- Supports multiple aggregation strategies
- Configurable decision threshold

```python
# Create a profile classifier
profile_model = ProfileClassifier(
    sequence_classifier=sequence_model,
    threshold=0.5,
    aggregation='mean'  # Options: 'mean', 'median', 'mean_vote', 'total_vote'
)
```

## Training

### Basic Training

```python
# Train a conversation-level classifier
model_path = train_model(
    model_type='conversation',
    dataset_path='data/analyzed_conversations',
    num_epochs=50,
    batch_size=256,
    learning_rate=1e-4
)

# Train an author-level classifier
author_model_path = train_model(
    model_type='author',
    dataset_path='data/analyzed_conversations',
    num_epochs=75,
    batch_size=256
)
```

### Class Imbalance Handling

The package provides multiple strategies for handling class imbalance:

- **Balanced**: Simple inverse class frequency
- **Effective**: Based on effective number of samples
- **Focal**: Inspired by focal loss weighting
- **Sqrt**: Square root scaling of class weights

```python
# Calculate class weights
pos_weight = estimate_pos_weight(
    labels=dataset.labels.numpy(),
    method='focal'  # Options: 'balanced', 'effective', 'focal', 'sqrt'
)
```

### Model Evaluation

```python
# Evaluate a trained model
metrics = test_model(
    model_type='conversation',
    model_path='models/conversation_classifier.pt',
    dataset_path='data/analyzed_conversations'
)

# Evaluate profile classification
profile_metrics = evaluate_profile_classifier(
    author_model_path='models/author_classifier.pt',
    dataset_path='data/analyzed_conversations',
    aggregation='mean'
)
```

### Aggregation Methods Comparison

Compare different profile-level aggregation strategies:

```python
results = compare_aggregation_methods(
    author_model_path='models/author_classifier.pt',
    dataset_path='data/analyzed_conversations'
)
```

## Metrics

The package tracks multiple performance metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Optimal classification threshold

## Advanced Features

### Custom Positional Encoding

The package includes an advanced positional encoding implementation:

- Sinusoidal position embeddings
- Dynamic sequence length handling
- Dropout regularization

### Residual Blocks

Custom residual blocks with:

- Dual linear transformations
- Layer normalization
- ReLU activation
- Configurable dropout

### Weight Initialization

Specialized weight initialization for different layer types:

- Xavier/Glorot initialization for linear layers
- Kaiming/He initialization for convolutional layers
- Custom initialization for attention layers

## Usage Examples

### Complete Training Pipeline

```python
# 1. Train conversation classifier
conv_model_path = train_model(
    model_type='conversation',
    dataset_path='data/analyzed_conversations',
    num_epochs=50
)

# 2. Train author classifier
author_model_path = train_model(
    model_type='author',
    dataset_path='data/analyzed_conversations',
    num_epochs=75
)

# 3. Evaluate models
test_model('conversation', conv_model_path, 'data/analyzed_conversations')
test_model('author', author_model_path, 'data/analyzed_conversations')

# 4. Compare profile aggregation methods
compare_aggregation_methods(
    author_model_path=author_model_path,
    dataset_path='data/analyzed_conversations'
)
```

### Custom Training Loop

```python
# Initialize model and optimizer
model = SequenceClassifier(input_size=256)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')

# Training loop
for epoch in range(num_epochs):
    # Train
    train_loss, train_preds, train_labels = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    
    # Validate
    val_loss, val_preds, val_labels, val_metrics, threshold = evaluate(
        model, val_loader, criterion, device
    )
    
    # Update learning rate
    scheduler.step(val_metrics['f1'])
```

## API Reference

### Models

#### SequenceClassifier
- Main sequence classification model
- Combines transformer encoding with multi-scale feature extraction

#### ProfileClassifier
- Profile-level classifier using pretrained sequence model
- Supports multiple aggregation strategies

### Training Functions

#### train_model
- Main training function
- Supports both conversation and author classification

#### evaluate_profile_classifier
- Evaluates profile-level classification
- Tests different aggregation methods

#### test_model
- Evaluation function for trained models
- Provides comprehensive metrics

### Utility Functions

#### estimate_pos_weight
- Calculates class weights for imbalanced datasets
- Supports multiple weighting strategies

#### calculate_metrics_threshold
- Computes classification metrics
- Determines optimal classification threshold
## Best Practices

1. **Data Preparation**
   - Clean and preprocess input features
   - Handle missing values
   - Normalize/standardize features

2. **Model Selection**
   - Start with default hyperparameters
   - Adjust based on validation performance
   - Monitor for overfitting

3. **Training**
   - Use appropriate batch size for your GPU
   - Monitor learning rate changes
   - Save checkpoints regularly

4. **Evaluation**
   - Use multiple metrics for assessment
   - Compare different aggregation methods
   - Consider domain-specific requirements