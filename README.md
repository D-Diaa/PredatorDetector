# Conversation Analysis Framework

A comprehensive Python framework for analyzing conversations, detecting problematic interactions, and building interpretable ML models for conversation and author-level classification.

## Overview

This project provides an end-to-end solution for conversation analysis through four integrated packages:

1. **Extractors**: Feature extraction using state-of-the-art transformer models
2. **Datasets**: Efficient dataset creation and processing for conversation analysis
3. **Training**: PyTorch-based model training and evaluation
4. **Visualization**: Comprehensive tools for result analysis and model interpretability

## Core Components

### 1. Extractors Package
The `extractors` package provides a comprehensive suite of feature extractors using state-of-the-art transformer models:

- **SentimentExtractor**: Uses RoBERTa for advanced sentiment analysis (positive, negative, neutral) with confidence scores
- **EmotionExtractor**: Detects multiple emotions (happiness, sadness, anger, fear, surprise, disgust) with intensity scores
- **ToxicityExtractor**: Identifies various forms of toxic content (severe toxicity, obscenity, threats, insults, identity attacks)
- **IntentExtractor**: Recognizes conversation intents including trust building, manipulation, isolation attempts
- **Word2AffectExtractor**: Extracts psycholinguistic features like valence, arousal, dominance

All extractors support:
- Batch processing with GPU acceleration
- Automatic text chunking for long sequences
- Memory-efficient processing
- Built-in error handling

### 2. Datasets Package
The `datasets` package handles dataset creation and processing with:
- **ConversationParser**: Efficient XML parsing and dataset creation
- **ConversationAnalyzer**: Multi-feature extraction and analysis pipeline
- **ConversationSequenceDataset**: Conversation-level sequence handling
- **AuthorConversationSequenceDataset**: Author-level sequence analysis
- Built-in caching and memory-mapped file support
- Integration with HuggingFace's datasets library

### 3. Training Package
The `training` package provides PyTorch-based model training with:
- **SequenceClassifier**: Transformer-based model with multi-scale feature extraction
- **ProfileClassifier**: Author profile classification with multiple aggregation strategies
- Advanced training features like class imbalance handling and dynamic thresholding
- Comprehensive metrics tracking and model checkpointing
- Flexible model architecture with customizable hyperparameters

### 4. Visualization Package
The `visualization` package offers comprehensive analysis tools:
- **Dataset Analysis**: Feature correlation, distribution statistics, temporal patterns
- **Sequential Analysis**: Temporal autocorrelation, cross-feature correlations, trajectory analysis
- **Feature Importance**: Multiple interpretability methods (SHAP, Integrated Gradients, etc.)
- **Visualization Functions**: Comprehensive plotting utilities for all analyses
- Automated report generation and result visualization

## Features

### Feature Extraction
- Sentiment analysis using RoBERTa
- Emotion detection with GoEmotions
- Toxicity measurement across multiple dimensions
- Intent classification using zero-shot learning
- Word affect analysis for psycholinguistic features
- GPU-accelerated batch processing

### Dataset Processing
- Memory-efficient handling of large conversation datasets
- Automatic feature extraction and caching
- Support for both conversation and author-level analysis
- Flexible sequence length handling
- Weighted sampling for imbalanced datasets
- Integration with HuggingFace's datasets library

### Model Training
- Transformer-based sequence encoding
- Multi-scale feature extraction
- Advanced class imbalance handling
- Profile-level aggregation strategies
- Comprehensive metrics tracking
- Automated model checkpointing

### Visualization and Interpretability
- Dataset feature analysis and visualization
- Sequential pattern analysis
- Multiple feature importance methods:
  - SHAP (SHapley Additive exPlanations)
  - Integrated Gradients
  - Attention Attribution
  - Feature Ablation
  - Permutation Importance
- Temporal correlation analysis

## Installation

```bash
pip install torch  # Install appropriate version for your system
pip install transformers detoxify
git clone https://huggingface.co/hplisiecki/word2affect_english  # Required for Word2Affect features
```

## Quick Start

### 1. Feature Extraction

```python
from extractors import SentimentExtractor, EmotionExtractor, ToxicityExtractor

# Initialize extractors
sentiment = SentimentExtractor()
emotion = EmotionExtractor()
toxicity = ToxicityExtractor()

# Extract features
text = "Your message here"
sentiment_scores = sentiment.extract(text)
emotion_scores = emotion.extract(text)
toxicity_scores = toxicity.extract(text)
```

### 2. Dataset Creation

```python
from datasets import ConversationParser, ConversationAnalyzer

# Parse and analyze conversations
parser = ConversationParser('conversations.xml')
analyzer = ConversationAnalyzer(batch_size=16)

# Create dataset
raw_dataset = parser.create_dataset()
analyzed_dataset = analyzer.process_dataset(raw_dataset)
```

### 3. Model Training

```python
from training import train_model, test_model

# Train models
conv_model_path = train_model(
    model_type='conversation',
    dataset_path='analyzed_conversations',
    num_epochs=50
)

# Evaluate
metrics = test_model(
    model_type='conversation',
    model_path=conv_model_path,
    dataset_path='analyzed_conversations'
)
```

### 4. Visualization

```python
from visualization import (
    analyze_dataset_features,
    analyze_sequential_features,
    analyze_feature_importance
)

# Analyze and visualize results
dataset_results = analyze_dataset_features(
    dataloader=train_loader,
    feature_names=dataset.feature_keys
)

visualize_all_results(dataset_results)
```

## System Architecture

### Main Pipeline
```
Raw Conversations → Feature Extraction → Dataset Creation → Model Training → Visualization
     ↓                     ↓                    ↓                 ↓              ↓
   XML/Text        Sentiment/Emotion     Sequence Dataset    Training Loop    Analysis Plots
   Format          Toxicity/Intent       Author Dataset      Evaluation      Feature Importance
                   Word Affect           Cache System        Checkpoints     Interpretability
```

### Feature Processing Pipeline
```
Input Text → Extractors Pipeline → Feature Vector
    ↓              ↓                    ↓
Cleaning    Sentiment Analysis    Feature Fusion
Batching    Emotion Detection    Normalization
Chunking    Toxicity Analysis    Validation
            Intent Classification
            Word Affect Analysis
```

### Training Pipeline
```
Dataset → Data Processing → Model Architecture → Training Loop → Evaluation
   ↓            ↓                    ↓                ↓            ↓
XML Data    Normalization    Transformer Encoder    Forward     Metrics
Parsing     Sequencing      Feature Extraction     Backward    Threshold
Features    Batching        Classification Head    Updates     Validation
```

### Analysis Pipeline
```
Model Output → Feature Analysis → Visualization → Interpretation
     ↓               ↓                  ↓              ↓
Predictions    SHAP Values         Distribution    Feature
Sequences      Gradient Analysis    Correlation    Importance
Profiles       Ablation Studies     Temporal       Explanation
               Permutation Tests    Patterns       Insights
```

## Best Practices

### Data Processing
- Clean and preprocess input features
- Handle missing values appropriately
- Use batch processing for large datasets
- Enable GPU acceleration when available
- Implement proper caching strategies

### Model Training
- Start with default hyperparameters
- Monitor for overfitting
- Use appropriate batch sizes
- Save checkpoints regularly
- Validate with multiple metrics

### Analysis
- Examine feature distributions
- Consider temporal patterns
- Use multiple interpretability methods
- Validate results across different approaches

## Error Handling

All packages include comprehensive error handling:
- Input validation
- NaN/Inf checking
- Sequence length validation
- Feature compatibility verification
- Graceful failure handling

## Performance Optimization

- GPU acceleration for feature extraction
- Memory-mapped file loading
- Efficient caching system
- Vectorized operations
- Batch processing