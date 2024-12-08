# Extractors

## Introduction
The Extractors package is a sophisticated text analysis toolkit designed to extract various linguistic and behavioral features from text data. It leverages transformer-based models to analyze multiple dimensions of text, including sentiment, emotions, toxicity, intent, and psycholinguistic features.

The package employs a modular approach with a base abstract class and specialized extractors, each focused on specific aspects of text analysis. All extractors support both single-text and batch processing, with efficient handling of long texts through automatic chunking and aggregation.

## Components

### Base Classes
- **FeatureExtractor**: Abstract base class defining the interface for all feature extractors. Handles basic functionality like feature validation and batch processing.
- **TransformerExtractor**: Base implementation for transformer-based extractors. Manages text chunking, batching, and device management.

### Specialized Extractors
- **Word2AffectExtractor**: Implementation of the VAD+ approach from Plisiecki & Sobieszek (2024) [DOI: 10.3758/s13428-023-02212-3]. Uses transformer-based architecture to analyze five key psycholinguistic dimensions: Valence (emotional positivity/negativity, r=0.95), Arousal (intensity of emotion), Dominance (sense of control), Age of Acquisition (when a word is typically learned), and Concreteness (how tangible/abstract a concept is, r=0.95). Model available on [Hugging Face](https://huggingface.co/hplisiecki/word2affect_english).
   - Features: Valence, Arousal, Dominance, Age of Acquisition, Concreteness

- **ToxicityExtractor**: Implements toxic comment classification using the Detoxify library (Hanu & Unitary team, 2020). Utilizes transformer-based models trained on three Jigsaw challenges: Toxic Comment Classification, Unintended Bias, and Multilingual Classification. The extractor supports three model variants:
   - Original model (BERT-based): Trained on Wikipedia comments, achieves 98.64% AUC score
   - Unbiased model (RoBERTa-based): Trained to minimize unintended bias, achieves 93.74% AUC score
   - Multilingual model (XLM-RoBERTa-based): Supports 7 languages with 92.11% AUC score
   - Features: toxicity, severe_toxicity, obscene, threat, insult, identity_attack
   - Additional bias metrics available for identity groups including gender, religion, and race

- **SentimentExtractor**: Implements sentiment analysis using cardiffnlp/twitter-roberta-base-sentiment-latest (Camacho-Collados et al., 2022), trained on ~124M tweets (2018-2021) and fine-tuned on the TweetEval benchmark. Specialized in social media text analysis.
   - Base Model: RoBERTa trained on social media content
   - Features: sentiment_positive, sentiment_negative, sentiment_neutral
   - Preprocessing: Automatically handles @mentions and URL links
   - Output: Confidence scores for each sentiment class
   - Optimal for: Social media content, informal text, and contemporary language

- **EmotionExtractor**: Uses RoBERTa-base model fine-tuned on the GoEmotions dataset (SamLowe/roberta-base-go_emotions) for multi-label emotion classification. The model is trained on Reddit data and can detect multiple emotions simultaneously.
   - Base Model: RoBERTa trained on social media content
   - Dataset: GoEmotions (Reddit-based, 28 emotion labels)
   - Features: Comprehensive emotion detection including admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, and neutral
   - Performance: Strong accuracy for well-represented emotions (e.g., gratitude: 96% precision, love: 80% F1-score)
   - Output: Probability scores for each emotion (threshold of 0.5 typically applied)

- **IntentExtractor**: Implements zero-shot intent classification using BART-large-MNLI (facebook/bart-large-mnli), trained on the MultiNLI dataset. Uses a novel approach of posing intent classification as a natural language inference task.
   - Base Model: BART-large fine-tuned on MNLI
   - Methodology: Zero-shot classification by converting intents to natural language hypotheses
   - Features: Detects 10 conversation intents: trust_building, isolation_attempt, boundary_testing, personal_probing, manipulation, secrecy_pressure, authority_undermining, reward_offering, normalization, meeting_planning
   - Output: Independent probability scores for each intent category
   - Advantage: Can be easily extended to detect new intent types without retraining

## Structure

```
extractors/
├── __init__.py            # Package initialization and feature set definitions
├── extractor.py           # Base abstract class definition
├── TransformerExtractors.py   # Implementation of transformer-based extractors
└── word2affect.py         # Custom model for psycholinguistic feature extraction
```

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

### Batch Processing
```python
texts = ["Great day!", "This is terrible.", "Just okay."]
results = sentiment.batch_extract(texts)
```

## Summary
The Extractors package is a comprehensive toolkit for multi-dimensional text analysis, combining various transformer-based models to extract linguistic, emotional, and behavioral features. Its modular design allows for easy extension and customization, while built-in optimizations ensure efficient processing of both single texts and large batches. The package is particularly suited for applications in content moderation, conversation analysis, user behavior understanding, and general text mining tasks.