# Trainer

This package provides a comprehensive framework for training and evaluating machine learning models, specifically designed for sequence and profile classification tasks. It includes modules for data handling, model definition, training, and evaluation.

## Features

-   **Sequence Classification**: Train models to classify sequences of data, such as conversations or time series.
-   **Profile Classification**: Aggregate sequence-level predictions to classify profiles, such as authors or users.
-   **Customizable Models**: Define and train custom models using a flexible architecture based on Transformer encoders and convolutional layers.
-   **Data Handling**: Efficiently load and preprocess data using custom dataset classes.
-   **Training Utilities**: Utilize functions for training, validation, and testing, including support for different loss functions and optimization strategies.
-   **Evaluation Metrics**: Calculate various performance metrics, including accuracy, precision, recall, F1-score, and AUC.
-   **Weight Estimation**: Estimate positive class weights for imbalanced datasets using different methods.


## Usage

### Training a Model

The `train_model` function [training.py] allows you to train either a conversation-level or author-level classifier.

```python
from trainer.training import train_model

# Train a conversation classifier
conv_model_path = train_model(
    model_type='conversation',
    dataset_path='data/analyzed_conversations',
    feature_set='models',
    feature_keys=['feat_1', 'feat_2', 'feat_3'],  # Replace with your feature keys
    num_epochs=100,
    batch_size=256,
    learning_rate=1e-4,
    device='cuda',
    seed=42
)

# Train an author classifier
author_model_path = train_model(
    model_type='author',
    dataset_path='data/analyzed_conversations',
    feature_set='models',
    feature_keys=['feat_1', 'feat_2', 'feat_3'],  # Replace with your feature keys
    num_epochs=100,
    batch_size=256,
    learning_rate=1e-4,
    device='cuda',
    seed=42
)
```

### Testing a Model

The `test_model` function [training.py] allows you to evaluate a trained model on a test dataset.

```python
from trainer.training import test_model

# Test a conversation classifier
test_model(
    model_type='conversation',
    model_path='path/to/conversation_model.pt',
    dataset_path='data/analyzed_conversations',
    device='cuda',
    seed=42
)

# Test an author classifier
test_model(
    model_type='author',
    model_path='path/to/author_model.pt',
    dataset_path='data/analyzed_conversations',
    device='cuda',
    seed=42
)
```

### Evaluating Profile Classifier

You can evaluate the profile classifier with different aggregation methods using the `compare_aggregation_methods` function.

```python
from trainer.training import evaluate_profile_classifier

evaluate_profile_classifier(
    author_model_path='path/to/author_model.pt',
    dataset_path='data/analyzed_conversations'
)
```

### Model Architecture

The package provides two main model classes:

-   **SequenceClassifier** [models.py]: A model for classifying sequences of data. It uses a Transformer encoder followed by multi-scale feature aggregation using Conv1d layers and a classifier head.
-   **ProfileClassifier** [models.py]: A model for classifying profiles by aggregating sequence-level predictions from a pretrained `SequenceClassifier`.

### Utilities

The `utils` module [utils.py] provides several helper functions for training and evaluation:

-   `train_epoch`: Trains the model for one epoch.
-   `evaluate`: Evaluates the model on a given dataset.
-   `calculate_metrics_threshold`: Calculates various performance metrics with a given or optimal threshold.
-   `estimate_pos_weight`: Estimates the positive class weight for imbalanced datasets using different methods.

## Modules

-   `__init__.py` [__init__.py]: Initializes the package and defines the main entry points.
-   `models.py` [models.py]: Contains the model architectures for sequence and profile classification.
-   `utils.py` [utils.py]: Provides utility functions for training and evaluation.
-   `training.py` [training.py]: Includes functions for training, testing, and evaluating models.
