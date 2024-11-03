import math
import re
import string
from collections import Counter
from typing import Dict, Any

from src.extractor import FeatureExtractor


class LinguisticExtractor(FeatureExtractor):
    """
    Advanced linguistic feature extractor that analyzes various text characteristics
    potentially useful for identifying aggressive or unusual communication patterns.
    """

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initialize the feature extractor with custom configuration.

        Args:
            config: Optional configuration dictionary that can include:
                   - min_word_length: minimum length for word length calculations
                   - max_ngram_size: maximum size of character n-grams to analyze
                   - custom_patterns: additional regex patterns to match
        """
        super().__init__(config)
        self.config = config or {}
        self.min_word_length = self.config.get('min_word_length', 2)
        self.max_ngram_size = self.config.get('max_ngram_size', 3)

        # Define custom patterns for aggressive language markers
        self.patterns = {
            'uppercase_words': r'\b[A-Z]{2,}\b',
            'repeated_punctuation': r'[!?.]{2,}',
            'repeated_letters': r'(.)\1{2,}',
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'email_tags': r'<email/>',
        }

        # Update patterns with any custom ones from config
        if 'custom_patterns' in self.config:
            self.patterns.update(self.config['custom_patterns'])

        # Define feature names
        self._feature_names = {
            'avg_word_length',
            'sentence_length',
            'unique_words_ratio',
            'capital_ratio',
            'punctuation_ratio',
            'special_char_ratio',
            'numeric_ratio',
            'uppercase_word_ratio',
            'repeated_punct_ratio',
            'repeated_letter_ratio',
            'word_count',
            'url_count',
            'email_tag_count',
            'question_count',
            'exclamation_count',
            'word_length_variance',
            'lexical_density',
            'character_diversity',
        }

    def extract(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic features from the given text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary of extracted features and their values
        """
        if not text:
            return {name: 0.0 for name in self._feature_names}

        # Basic text statistics
        words = [w for w in text.lower().split() if len(w) >= self.min_word_length]
        word_count = len(words)
        unique_words = set(words)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        # Character class counts
        capitals = sum(1 for c in text if c.isupper())
        punctuation = sum(1 for c in text if c in string.punctuation)
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        numbers = sum(1 for c in text if c.isnumeric())
        question_count = text.count('?')
        exclamation_count = text.count('!')

        # Pattern matches
        uppercase_words = len(re.findall(self.patterns['uppercase_words'], text))
        repeated_punct = len(re.findall(self.patterns['repeated_punctuation'], text))
        repeated_letters = len(re.findall(self.patterns['repeated_letters'], text))
        urls = len(re.findall(self.patterns['urls'], text))
        email_tags = len(re.findall(self.patterns['email_tags'], text))

        # Advanced calculations
        word_lengths = [len(w) for w in words]
        avg_word_length = sum(word_lengths) / word_count if word_count else 0
        word_length_var = (sum((l - avg_word_length) ** 2 for l in word_lengths) / word_count) if word_count > 1 else 0

        # Character diversity (entropy-based)
        char_freq = Counter(text.lower())
        total_chars = sum(char_freq.values())
        char_diversity = -sum((count / total_chars) * math.log2(count / total_chars)
                              for count in char_freq.values()) if total_chars else 0

        features = {
            'avg_word_length': avg_word_length,
            'sentence_length': sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0,
            'unique_words_ratio': len(unique_words) / word_count if word_count else 0,
            'capital_ratio': capitals / total_chars if total_chars else 0,
            'punctuation_ratio': punctuation / total_chars if total_chars else 0,
            'special_char_ratio': special_chars / total_chars if total_chars else 0,
            'numeric_ratio': numbers / total_chars if total_chars else 0,
            'uppercase_word_ratio': uppercase_words / word_count if word_count else 0,
            'repeated_punct_ratio': repeated_punct / len(sentences) if sentences else 0,
            'repeated_letter_ratio': repeated_letters / word_count if word_count else 0,
            'url_count': urls,
            'word_count': word_count,
            'email_tag_count': email_tags,
            'question_count': question_count,
            'exclamation_count': exclamation_count,
            'word_length_variance': word_length_var,
            'lexical_density': len(unique_words) / word_count if word_count else 0,
            'character_diversity': char_diversity
        }

        return features

    def get_feature_ranges(self) -> Dict[str, tuple]:
        """
        Get the expected value ranges for each feature.

        Returns:
            Dictionary mapping feature names to (min, max) tuples
        """
        return {
            'avg_word_length': (0.0, 30.0),
            'sentence_length': (0.0, 200.0),
            'unique_words_ratio': (0.0, 1.0),
            'capital_ratio': (0.0, 1.0),
            'punctuation_ratio': (0.0, 1.0),
            'special_char_ratio': (0.0, 1.0),
            'numeric_ratio': (0.0, 1.0),
            'uppercase_word_ratio': (0.0, 1.0),
            'repeated_punct_ratio': (0.0, 10.0),
            'repeated_letter_ratio': (0.0, 1.0),
            'url_count': (0.0, float('inf')),
            'word_count': (0.0, float('inf')),
            'email_tag_count': (0.0, float('inf')),
            'question_count': (0.0, float('inf')),
            'exclamation_count': (0.0, float('inf')),
            'word_length_variance': (0.0, float('inf')),
            'lexical_density': (0.0, 1.0),
            'character_diversity': (0.0, 8.0)  # log2(256) for ASCII
        }
