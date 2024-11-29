# Cyber-Predator Detection System

## Overview
This project implements a machine learning system for detecting potential cyber-predators in online conversations using a multi-dimensional approach combining EPA (Evaluation, Potency, Activity) sequences with transformer-based models for emotional and behavioral pattern analysis. The system processes chat logs through multiple feature extractors to identify suspicious behavioral patterns and temporal dynamics that may indicate predatory behavior.

## Architecture

### Core Components and Data Flow

1. **Data Processing Pipeline**
   - `parser.py`: 
     - XML chat log parsing using ElementTree
     - Conversation structuring with author, timestamp, and message tracking
     - Dataset creation using HuggingFace datasets format
     - Memory-efficient data loading with batch processing

2. **Analysis Pipeline**
   - `analyzer.py`:
     - ConversationAnalyzer class for streaming analysis
     - Feature extraction orchestration
     - Memory-efficient processing using map function
     - Attacker identification and tracking

3. **Dataset Management**
   - `dataset.py`:
     - BaseDataset with caching and normalization
     - ConversationSequenceDataset for conversation-level analysis
     - AuthorConversationSequenceDataset for author-level analysis
     - Custom data loaders with stratified sampling

4. **Model Architectures**
   - `models.py`:
     - Transformer Encoder:
       - Multi-head self-attention mechanism
       - Positional encoding for sequence order
       - Layer normalization and residual connections
     
     - Sequence Classifier:
       - Input projection layer
       - Multi-scale feature aggregation using Conv1d
       - Multiple kernel sizes (3, 5, 7) for pattern detection
       - Residual blocks with dropout
       
     - Profile Classifier:
       - Pretrained sequence classifier integration
       - Multiple aggregation methods (mean, median, vote)
       - Confidence thresholding
     
     - Initialization:
       - Xavier/Glorot for linear layers
       - Kaiming/He for convolutional layers
       - Scaled initialization for attention layers

5. **Training and Evaluation**
   - `training.py`:
     - Training loops with gradient clipping
     - Multi-stage evaluation pipeline
     - Metrics calculation with optimal thresholding
     - Model selection based on F1 score
     - Learning rate scheduling

6. **Utilities**
   - `utils.py`:
     - Class weight estimation methods
     - Performance metric calculations
     - Model selection utilities

## Features

### Feature Extractors

1. **Word2Affect Features**
   - Valence: Emotional positivity/negativity
   - Arousal: Level of energy/intensity
   - Dominance: Degree of control/power
   - Age of Acquisition: Developmental timing of word learning
   - Concreteness: Degree of tangibility/abstractness

2. **Toxicity Analysis**
   - General toxicity
   - Severe toxicity
   - Obscenity detection
   - Threat detection
   - Insult detection
   - Identity-based attacks

3. **Sentiment Analysis (RoBERTa-based)**
   - Positive sentiment
   - Neutral sentiment
   - Negative sentiment
   - Sentiment confidence scores

4. **Emotion Detection (GoEmotions)**
   - Basic emotions (happy, sad, angry, etc.)
   - Emotion intensity
   - Temporal emotional patterns
   - Emotional transitions

5. **Intent Classification (BART Zero-Shot)**
   - Trust building detection
   - Isolation attempt identification
   - Boundary testing patterns
   - Personal probing detection
   - Manipulation tactics
   - Secrecy pressure
   - Authority undermining
   - Reward offering
   - Normalization attempts
   - Meeting planning indicators

### System Features

1. **Data Processing**
   - Automatic XML parsing and structuring
   - Memory-mapped dataset handling
   - Efficient batch processing
   - Automatic caching with smart invalidation
   - Text cleaning and normalization

2. **Performance Optimization**
   - GPU acceleration for all extractors
   - Vectorized operations for feature processing
   - Memory-efficient streaming analysis
   - Smart batching for transformer models
   - Automatic sequence length optimization

3. **Analysis Capabilities**
   - Conversation-level pattern detection
   - Author-level behavioral profiling
   - Temporal sequence analysis
   - Multi-scale feature aggregation
   - Cross-conversation user tracking

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
tqdm>=4.65.0
detoxify>=0.5.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/cyber-predator-detection.git
cd cyber-predator-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The system follows a pipeline architecture with four main stages:

### 1. Data Preparation (parser.py)
```python
# Initialize the conversation loader
loader = ConversationLoader(
    xml_path='data/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml',
    max_conversations=-1  # Load all conversations
)

# Create and save the dataset
dataset = loader.create_dataset()
loader.save_to_disk('data/conversations')
```

### 2. Feature Extraction (analyzer.py)
```python
# Initialize the analyzer with GPU support
analyzer = ConversationAnalyzer(batch_size=16)

# Load known attackers list
analyzer.load_attackers('data/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt')

# Process conversations and extract features
dataset = datasets.load_from_disk('data/conversations')
analyzed_dataset = analyzer.process_dataset(dataset)
analyzed_dataset.save_to_disk('data/analyzed_conversations')
```

### 3. Dataset Creation (dataset.py)
```python
# Create conversation-level dataset
conv_dataset = ConversationSequenceDataset(
    dataset_path='data/analyzed_conversations',
    max_seq_length=256,
    min_seq_length=10
)

# Create author-level dataset
author_dataset = AuthorConversationSequenceDataset(
    dataset_path='data/analyzed_conversations',
    max_seq_length=128,
    min_seq_length=5
)

# Create data loaders
train_loader, val_loader, test_loader = create_dataloaders(
    dataset=conv_dataset,
    batch_size=256,
    train_split=0.8,
    val_split=0.1
)
```

### 4. Training and Evaluation (training.py)

#### Training Models
```python
# Train conversation classifier
conv_model_path = train_model(
    model_type='conversation',
    dataset_path='data/analyzed_conversations',
    num_epochs=50,
    batch_size=256,
    learning_rate=1e-4
)

# Train author classifier
author_model_path = train_model(
    model_type='author',
    dataset_path='data/analyzed_conversations',
    num_epochs=75,
    batch_size=256,
    learning_rate=1e-4
)
```

#### Evaluation
```python
# Evaluate models
test_metrics = test_model(
    model_type='conversation',
    model_path='models/conversation_classifier.pt',
    dataset_path='data/analyzed_conversations'
)

# Profile-level evaluation
profile_metrics = evaluate_profile_classifier(
    author_model_path='models/author_classifier.pt',
    dataset_path='data/analyzed_conversations',
    aggregation='mean'
)
```
