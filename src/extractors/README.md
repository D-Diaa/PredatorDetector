# Extractors

A comprehensive Python package for extracting linguistic features, emotions, sentiment, toxicity, and intent from text using state-of-the-art transformer models.

## Features

- **Sentiment Analysis**: Detect positive, negative, and neutral sentiment with confidence scores
- **Emotion Detection**: Analyze multiple emotions (happiness, sadness, anger, fear, surprise, disgust) with intensity scores
- **Toxicity Detection**: Identify various forms of toxic content including severe toxicity, obscenity, threats, insults, and identity attacks
- **Intent Classification**: Recognize multiple conversation intents including trust building, manipulation, isolation attempts, and more
- **Linguistic Features**: Extract psycholinguistic features like valence, arousal, dominance, age of acquisition, and concreteness
- **Efficient Processing**: Batch processing, automatic text chunking, and GPU acceleration support

## Installation

The package requires Python 3.7+ and PyTorch. Install the package and its dependencies:

```bash
pip install torch  # Install appropriate version for your system
pip install transformers detoxify
git clone https://huggingface.co/hplisiecki/word2affect_english  # Required for Word2Affect features
```

## Quick Start

```python
from extractors import SentimentExtractor, EmotionExtractor, ToxicityExtractor

# Initialize extractors
sentiment = SentimentExtractor()
emotion = EmotionExtractor()
toxicity = ToxicityExtractor()

# Analyze text
text = "I'm really excited about this new project!"

# Get sentiment scores
sentiment_scores = sentiment.extract(text)
print("Sentiment:", sentiment_scores)
# Output: {
#     'sentiment_positive': 0.92,
#     'sentiment_neutral': 0.06,
#     'sentiment_negative': 0.02,
#     'sentiment_confidence': 0.92
# }

# Get emotion scores
emotion_scores = emotion.extract(text)
print("Emotions:", emotion_scores)
# Output: {
#     'emotion_happy': 0.85,
#     'emotion_surprise': 0.12,
#     'emotion_sad': 0.01,
#     'emotion_angry': 0.01,
#     'emotion_fear': 0.01,
#     'emotion_disgust': 0.00,
#     'emotion_intensity': 0.85
# }
```

## Detailed Usage

### Base Feature Extractor

All extractors inherit from the `FeatureExtractor` base class, which provides common functionality:

```python
from extractors import FeatureExtractor

class CustomExtractor(FeatureExtractor):
    def __init__(self, config=None):
        super().__init__(config)
        self._feature_names = {'feature1', 'feature2'}
    
    def extract(self, text):
        # Implementation
        pass

    def get_feature_ranges(self):
        return {
            'feature1': (0.0, 1.0),
            'feature2': (0.0, 1.0)
        }
```

### Sentiment Analysis

The `SentimentExtractor` uses RoBERTa for advanced sentiment analysis:

```python
from extractors import SentimentExtractor

# Initialize with custom config
config = {
    'device': 'cuda',  # Use GPU if available
    'batch_size': 32   # Adjust based on your needs
}
sentiment = SentimentExtractor(config)

# Analyze multiple texts efficiently
texts = [
    "This is amazing!",
    "I'm not sure about this.",
    "This is terrible."
]
results = sentiment.batch_extract(texts)
```

### Emotion Detection

The `EmotionExtractor` uses the GoEmotions dataset for fine-grained emotion detection:

```python
from extractors import EmotionExtractor

emotion = EmotionExtractor()

# Single text analysis
text = "I can't believe how wonderful this is!"
emotions = emotion.extract(text)

# Access specific emotions
happiness = emotions['emotion_happy']
intensity = emotions['emotion_intensity']
```

### Toxicity Detection

The `ToxicityExtractor` identifies various forms of toxic content:

```python
from extractors import ToxicityExtractor

toxicity = ToxicityExtractor()

text = "Your opinion is valuable and I respect it."
toxicity_scores = toxicity.extract(text)
print(toxicity_scores)
# Output: {
#     'toxicity': 0.01,
#     'severe_toxicity': 0.00,
#     'obscene': 0.00,
#     'threat': 0.00,
#     'insult': 0.01,
#     'identity_attack': 0.00
# }
```

### Intent Classification

The `IntentExtractor` uses zero-shot classification to identify conversation intents:

```python
from extractors import IntentExtractor

intent = IntentExtractor()

text = "Let's keep this conversation between us."
intent_scores = intent.extract(text)
print(intent_scores)
# Output: {
#     'intent_secrecy_pressure': 0.85,
#     'intent_isolation_attempt': 0.45,
#     'intent_trust_building': 0.30,
#     ...
#     'intent_confidence': 0.85
# }
```

### Word2Affect Features

Extract psycholinguistic features using the `Word2AffectExtractor`:

```python
from extractors import Word2AffectExtractor

w2a = Word2AffectExtractor()

text = "I feel energized and confident about this decision!"
features = w2a.extract(text)
print(features)
# Output: {
#     'Valence': 0.82,
#     'Arousal': 0.65,
#     'Dominance': 0.75,
#     'Age of Acquisition': 0.45,
#     'Concreteness': 0.30
# }
```

## Advanced Features

### Text Chunking

All extractors automatically handle long texts by chunking them into smaller pieces with overlap:

```python
from extractors import SentimentExtractor

sentiment = SentimentExtractor()
long_text = "..." # Very long text

# Automatically chunks text and averages results
results = sentiment.extract(long_text)
```

### Batch Processing

Process multiple texts efficiently:

```python
from extractors import ToxicityExtractor

toxicity = ToxicityExtractor({'batch_size': 64})

# Process many texts at once
texts = ["text1", "text2", "text3", ...]
results = toxicity.batch_extract(texts)
```

### Custom Configuration

All extractors accept a configuration dictionary for customization:

```python
config = {
    'device': 'cuda',  # Use GPU
    'batch_size': 32,  # Batch size for processing
}

# Apply config to any extractor
from extractors import EmotionExtractor
emotion = EmotionExtractor(config)
```

## Performance Considerations

- All extractors support GPU acceleration when available
- Batch processing is more efficient than processing texts individually
- Text chunking helps handle long texts while maintaining accuracy
- Models are loaded lazily upon first use to conserve memory

## Error Handling

The extractors include robust error handling:

```python
from extractors import SentimentExtractor

sentiment = SentimentExtractor()

# Empty or invalid texts return zero scores
result = sentiment.extract("")
# Returns: {'sentiment_positive': 0.0, 'sentiment_neutral': 0.0, ...}

# Batch processing handles errors gracefully
texts = ["valid text", "", None, "another text"]
results = sentiment.batch_extract(texts)
# Returns valid results for valid texts, zero scores for invalid ones
```

## Best Practices

1. **Batch Processing**: Use `batch_extract()` for multiple texts
2. **GPU Usage**: Set device='cuda' in config when GPU is available
3. **Memory Management**: Initialize extractors only when needed
4. **Text Cleaning**: Input text is automatically cleaned, but pre-cleaning sensitive data is recommended
5. **Error Handling**: Always validate extractor outputs using `validate_features()`