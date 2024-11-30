# Conversation Analysis Dataset Package

## Overview

This Python package provides a robust framework for processing, analyzing, and creating datasets from conversation data, with a focus on detecting problematic interactions. It includes tools for feature extraction, sequence analysis, and both conversation-level and author-level classification tasks.

## Key Features

- Memory-efficient processing of large conversation datasets
- Automatic feature extraction including sentiment, emotion, toxicity, and intent
- Support for both conversation-level and author-level analysis
- Built-in caching system for improved performance
- Comprehensive data normalization and preprocessing
- Flexible sequence length handling with padding/truncation
- Integrated with HuggingFace's datasets library
- CUDA-accelerated feature extraction
- Weighted sampling support for imbalanced datasets

## Package Structure

```
datasets/
├── __init__.py
├── analyzer.py    # Conversation analysis and feature extraction
├── dataset.py     # Dataset creation and processing
└── parser.py      # XML conversation parsing
```

## Core Components

### ConversationParser

The `ConversationParser` class handles loading and parsing conversation data from XML files into a structured format compatible with HuggingFace datasets.

```python
from datasets import ConversationParser

# Initialize parser
parser = ConversationParser(
    xml_path='conversations.xml',
    max_conversations=-1  # -1 for all conversations
)

# Create and save dataset
dataset = parser.create_dataset()
parser.save_to_disk('output_directory')
```

### ConversationAnalyzer

The `ConversationAnalyzer` processes conversations using multiple feature extractors:

- Sentiment analysis
- Emotion detection
- Toxicity measurement
- Intent classification
- Word affect analysis

```python
from datasets import ConversationAnalyzer

analyzer = ConversationAnalyzer(batch_size=16)

# Load known problematic authors list
analyzer.load_attackers('attackers.txt')

# Process dataset
analyzed_dataset = analyzer.process_dataset(dataset)

# Get statistics
user_stats = analyzer.collect_user_statistics(analyzed_dataset)
```

### Dataset Classes

#### BaseDataset

Abstract base class implementing core dataset functionality:

- Caching mechanism
- Memory-mapped loading
- Sequence normalization
- Vectorized operations

#### ConversationSequenceDataset

Handles conversation-level sequence classification:

```python
from datasets import ConversationSequenceDataset

dataset = ConversationSequenceDataset(
    dataset_path='analyzed_conversations',
    max_seq_length=50,
    min_seq_length=5,
    normalize=True
)
```

#### AuthorConversationSequenceDataset

Focuses on author-level sequence classification:

```python
from datasets import AuthorConversationSequenceDataset

dataset = AuthorConversationSequenceDataset(
    dataset_path='analyzed_conversations',
    max_seq_length=30,
    min_seq_length=5,
    normalize=True
)
```

## Data Processing Pipeline

1. **XML Parsing**: Load raw conversation data
   ```python
   parser = ConversationParser('input.xml')
   raw_dataset = parser.create_dataset()
   ```

2. **Feature Extraction**: Process conversations with multiple analyzers
   ```python
   analyzer = ConversationAnalyzer()
   analyzed_dataset = analyzer.process_dataset(raw_dataset)
   ```

3. **Dataset Creation**: Create specialized datasets for training
   ```python
   conv_dataset = ConversationSequenceDataset(
       dataset_path='analyzed_data',
       max_seq_length=50
   )
   ```

4. **DataLoader Creation**: Prepare data for model training
   ```python
   train_loader, val_loader, test_loader = create_dataloaders(
       conv_dataset,
       batch_size=32,
       train_split=0.8,
       val_split=0.1
   )
   ```

## Feature Extraction Details

The package extracts several types of features:

- **Sentiment**: Overall message sentiment (positive/negative/neutral)
- **Emotions**: Detection of specific emotions in messages
- **Toxicity**: Measurement of harmful content
- **Intent**: Classification of message purpose
- **Word Affect**: Emotional impact of word choices

## Memory Efficiency Features

- Batch processing for large datasets
- Memory-mapped file loading
- Efficient caching system
- Vectorized operations for feature extraction
- GPU acceleration for feature extractors

## Performance Optimization

### Caching System
The package implements a sophisticated caching system:

```python
# Cache handling is automatic
dataset = ConversationSequenceDataset(
    dataset_path='data',
    cache_prefix="conv"  # Defines cache file naming
)
```

### Batch Processing
Optimized batch processing for memory efficiency:

```python
analyzer = ConversationAnalyzer(batch_size=16)
analyzed_dataset = analyzer.process_dataset(dataset)
```

## Working with Imbalanced Data

The package provides built-in support for handling imbalanced datasets through weighted sampling:

```python
train_loader, val_loader, test_loader = create_dataloaders(
    dataset,
    batch_size=32,
    use_weighted_sampler=True
)
```

## Error Handling and Validation

The package includes comprehensive error checking:

- Validation of input data formats
- Checking for NaN/Inf values
- Sequence length validation
- Feature compatibility verification

## Usage Examples

### Complete Pipeline Example

```python
from datasets import (
    ConversationParser,
    ConversationAnalyzer,
    ConversationSequenceDataset,
    create_dataloaders
)

# Parse raw data
parser = ConversationParser('conversations.xml')
raw_dataset = parser.create_dataset()
parser.save_to_disk('raw_data')

# Analyze conversations
analyzer = ConversationAnalyzer(batch_size=16)
analyzer.load_attackers('attackers.txt')
analyzed_dataset = analyzer.process_dataset(raw_dataset)
analyzed_dataset.save_to_disk('analyzed_data')

# Create sequence dataset
conv_dataset = ConversationSequenceDataset(
    dataset_path='analyzed_data',
    max_seq_length=50,
    normalize=True
)

# Create data loaders
train_loader, val_loader, test_loader = create_dataloaders(
    conv_dataset,
    batch_size=32,
    train_split=0.8,
    val_split=0.1,
    use_weighted_sampler=True
)
```

### Analyzing User Statistics

```python
# Get user-level statistics
analyzer = ConversationAnalyzer()
user_stats = analyzer.collect_user_statistics(analyzed_dataset)
user_stats.to_csv('user_statistics.csv')

# Get conversation-level statistics
conv_stats = get_conversation_statistics(analyzed_dataset)
conv_stats.to_csv('conversation_statistics.csv')
```

## Debug and Monitoring Tools

The package includes utilities for monitoring dataset processing:

```python
from datasets import print_dataset_stats

# Print detailed statistics
print_dataset_stats(dataset, "Dataset Name")

# Check data loader integrity
check_data(train_loader)
```