import html
import os
from typing import List

import torch
from transformers import pipeline

from typing import Dict, Any

from detoxify import Detoxify

from src.extractor import FeatureExtractor
from src.intents import INTENT_LABELS

os.environ["TOKENIZERS_PARALLELISM"] = "false"
def clean_text(text: str) -> str:
    """Clean and normalize text for sentiment analysis."""
    text = html.unescape(text)
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        t = 'email@email.com' if '@' in t and '.' in t and len(t) > 3 and not t.startswith('@') else t
        new_text.append(t)
    return " ".join(new_text)


class ToxicityExtractor(FeatureExtractor):
    """
    Extracts toxicity-related features using a model fine-tuned on toxic comment classification.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = Detoxify('original', device='mps')

        self._feature_names = {
            'toxicity', 'severe_toxicity', 'obscene', 'threat',
            'insult', 'identity_attack'
        }

    def extract(self, text: str) -> Dict[str, float]:
        if not text:
            return {name: 0.0 for name in self._feature_names}
        features = self.model.predict(text)
        return {k: v.item() for k, v in features.items()}

    def get_feature_ranges(self) -> Dict[str, tuple]:
        return {name: (0.0, 1.0) for name in self._feature_names}


class SentimentExtractor(FeatureExtractor):
    """
    Feature extractor for sentiment analysis using the CardiffNLP Twitter RoBERTa model.
    Auto-detects sentiment labels from the model pipeline.
    """

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initialize the sentiment extractor with the CardiffNLP model.

        Args:
            config: Optional configuration dictionary that can include:
                   - batch_size: size of batches for processing
                   - device: computing device (cuda, cpu, or mps)
        """
        super().__init__(config)
        self.config = config or {}
        self.batch_size = self.config.get('batch_size', 32)

        # Set up device
        self.device = (self.config.get('device') or
                       "mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else
                       "cpu")

        # Initialize the sentiment analyzer
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.analyzer = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=self.device,
            top_k=None
        )

        # Auto-detect sentiment labels by running a test prediction
        test_result = self.analyzer("test")[0]
        self.sentiment_labels = sorted({r['label'] for r in test_result})

        # Define feature names based on detected labels
        self._feature_names = {f'sentiment_{label.lower()}' for label in self.sentiment_labels}
        self._feature_names.add('sentiment_confidence')

    def extract(self, text: str) -> Dict[str, float]:
        """
        Extract sentiment features from the given text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary of sentiment scores
        """
        if not text:
            return {name: 0.0 for name in self._feature_names}

        # Clean the text
        cleaned_text = clean_text(text)

        # Get sentiment predictions
        results = self.analyzer(cleaned_text)[0]

        # Initialize scores dictionary
        scores = {f'sentiment_{label.lower()}': 0.0 for label in self.sentiment_labels}

        # Process results
        for result in results:
            label = result['label']
            score = result['score']
            scores[f'sentiment_{label.lower()}'] = score

        # Calculate confidence as max score
        scores['sentiment_confidence'] = max(result['score'] for result in results)

        return scores

    def batch_extract(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Extract features from multiple texts efficiently using batching.

        Args:
            texts: List of input texts to analyze

        Returns:
            List of sentiment feature dictionaries
        """
        cleaned_texts = [clean_text(text) for text in texts]
        all_results = []

        # Process in batches
        for i in range(0, len(cleaned_texts), self.batch_size):
            batch = cleaned_texts[i:i + self.batch_size]
            batch_results = self.analyzer(batch)
            all_results.extend(batch_results)

        # Convert results to feature dictionaries
        features = []
        for results in all_results:
            scores = {f'sentiment_{label.lower()}': 0.0 for label in self.sentiment_labels}

            for result in results:
                label = result['label']
                score = result['score']
                scores[f'sentiment_{label.lower()}'] = score

            scores['sentiment_confidence'] = max(result['score'] for result in results)
            features.append(scores)

        return features

    def get_feature_ranges(self) -> Dict[str, tuple]:
        """Get the expected value ranges for each feature."""
        ranges = {f'sentiment_{label.lower()}': (0.0, 1.0)
                  for label in self.sentiment_labels}
        ranges['sentiment_confidence'] = (0.0, 1.0)
        return ranges


class EmotionExtractor(FeatureExtractor):
    """
    Feature extractor for emotion analysis using the RoBERTa GoEmotions model.
    Auto-detects emotion labels from the model pipeline.
    """

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initialize the emotion extractor with the GoEmotions model.

        Args:
            config: Optional configuration dictionary that can include:
                   - batch_size: size of batches for processing
                   - device: computing device (cuda, cpu, or mps)
        """
        super().__init__(config)
        self.config = config or {}
        self.batch_size = self.config.get('batch_size', 32)

        # Set up device
        self.device = (self.config.get('device') or
                       "mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else
                       "cpu")

        # Initialize the emotion classifier
        model_name = "SamLowe/roberta-base-go_emotions"
        self.classifier = pipeline(
            task="text-classification",
            model=model_name,
            device=self.device,
            top_k=None
        )

        # Auto-detect emotion labels by running a test prediction
        test_result = self.classifier("test")[0]
        self.emotion_labels = sorted({r['label'] for r in test_result})

        # Set feature names based on detected emotions
        self._feature_names = {f'emotion_{e.lower()}' for e in self.emotion_labels}
        self._feature_names.add('emotion_intensity')

    def extract(self, text: str) -> Dict[str, float]:
        """
        Extract emotion features from the given text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary of emotion scores
        """
        if not text:
            return {name: 0.0 for name in self._feature_names}

        # Clean the text
        cleaned_text = clean_text(text)

        # Get emotion predictions
        results = self.classifier(cleaned_text)[0]

        # Initialize scores dictionary with zeros
        scores = {f'emotion_{e.lower()}': 0.0 for e in self.emotion_labels}

        # Process results
        max_score = 0.0
        for result in results:
            emotion = result['label'].lower()
            score = result['score']
            scores[f'emotion_{emotion}'] = score
            max_score = max(max_score, score)

        # Add overall emotion intensity
        scores['emotion_intensity'] = max_score

        return scores

    def batch_extract(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Extract features from multiple texts efficiently using batching.

        Args:
            texts: List of input texts to analyze

        Returns:
            List of emotion feature dictionaries
        """
        cleaned_texts = [clean_text(text) for text in texts]
        all_results = []

        # Process in batches
        for i in range(0, len(cleaned_texts), self.batch_size):
            batch = cleaned_texts[i:i + self.batch_size]
            batch_results = self.classifier(batch)
            all_results.extend(batch_results)

        # Convert results to feature dictionaries
        features = []
        for results in all_results:
            scores = {f'emotion_{e.lower()}': 0.0 for e in self.emotion_labels}
            max_score = 0.0

            for result in results:
                emotion = result['label'].lower()
                score = result['score']
                scores[f'emotion_{emotion}'] = score
                max_score = max(max_score, score)

            scores['emotion_intensity'] = max_score
            features.append(scores)

        return features

    def get_feature_ranges(self) -> Dict[str, tuple]:
        """Get the expected value ranges for each feature."""
        ranges = {f'emotion_{e.lower()}': (0.0, 1.0)
                  for e in self.emotion_labels}
        ranges['emotion_intensity'] = (0.0, 1.0)
        return ranges


class IntentExtractor(FeatureExtractor):
    """
    Feature extractor for Intent analysis using the BERT model.
    """

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """

        Args:
            config: Optional configuration dictionary that can include:
                   - batch_size: size of batches for processing
                   - device: computing device (cuda, cpu, or mps)
        """
        super().__init__(config)
        self.config = config or {}
        self.batch_size = self.config.get('batch_size', 32)

        # Set up device
        self.device = (self.config.get('device') or
                       "mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else
                       "cpu")

        # Initialize the sentiment analyzer
        model_name = "facebook/bart-large-mnli"
        self.analyzer = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=self.device,
            top_k=None
        )

        self.intent_labels = INTENT_LABELS
        # Define feature names based on detected labels
        self._feature_names = {f'intent_{label.lower()}' for label in self.intent_labels}
        self._feature_names.add('intent_confidence')

    def extract(self, text: str) -> Dict[str, float]:
        """
        Extract sentiment features from the given text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary of sentiment scores
        """
        if not text:
            return {name: 0.0 for name in self._feature_names}

        # Clean the text
        cleaned_text = clean_text(text)

        # Get sentiment predictions
        results = self.analyzer(cleaned_text, self.intent_labels, multi_label=True)

        # Initialize scores dictionary
        scores = {f'intent_{label.lower()}': 0.0 for label in self.intent_labels}

        # Process results
        for label, score in zip(results["labels"], results["scores"]):
            scores[f'intent_{label.lower()}'] = score

        # Calculate confidence as max score
        scores['intent_confidence'] = max(results["scores"])

        return scores

    def batch_extract(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Extract features from multiple texts efficiently using batching.

        Args:
            texts: List of input texts to analyze

        Returns:
            List of sentiment feature dictionaries
        """
        cleaned_texts = [clean_text(text) for text in texts]
        all_results = []

        # Process in batches
        for i in range(0, len(cleaned_texts), self.batch_size):
            batch = cleaned_texts[i:i + self.batch_size]
            batch_results = self.analyzer(batch, self.intent_labels, multi_label=True)
            all_results.extend(batch_results)

        # Convert results to feature dictionaries
        features = []
        for results in all_results:
            scores = {f'intent_{label.lower()}': 0.0 for label in self.intent_labels}

            for label, score in zip(results["labels"], results["scores"]):
                scores[f'intent_{label.lower()}'] = score

            scores['intent_confidence'] = max(results["scores"])
            features.append(scores)

        return features

    def get_feature_ranges(self) -> Dict[str, tuple]:
        """Get the expected value ranges for each feature."""
        ranges = {f'sentiment_{label.lower()}': (0.0, 1.0)
                  for label in self.intent_labels}
        ranges['intent_confidence'] = (0.0, 1.0)
        return ranges
