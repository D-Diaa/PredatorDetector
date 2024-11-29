from abc import ABC, abstractmethod
from typing import Dict, List, Any, Set

import torch


class FeatureExtractor(ABC):
    """
    Abstract base class defining the interface for feature extractors.
    Feature extractors are responsible for analyzing text and extracting relevant features
    for conversation analysis and attacker identification.
    """

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initialize the feature extractor with optional configuration.

        Args:
            config: Dictionary containing configuration parameters for the extractor
        """
        self._config = config or {}
        self.device = (self._config.get('device') or
                       "cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else
                       "cpu")
        self._feature_names: Set[str] = set()

    @abstractmethod
    def extract(self, text: str) -> Dict[str, float]:
        """
        Extract features from the given text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary mapping feature names to their values
        """
        pass

    @property
    def keys(self) -> List[str]:
        """
        Get the names of all features this extractor can extract.

        Returns:
            List of feature names
        """
        return list(self._feature_names)

    def batch_extract(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Extract features from multiple texts efficiently.

        Args:
            texts: List of input texts to analyze

        Returns:
            List of dictionaries containing extracted features for each text
        """
        # Default implementation - override for more efficient batch processing
        return [self.extract(text) for text in texts]

    @property
    def feature_names(self) -> Set[str]:
        """
        Get the set of feature names this extractor produces.

        Returns:
            Set of feature names
        """
        return self._feature_names

    def validate_features(self, features: Dict[str, float]) -> bool:
        """
        Validate that extracted features match the expected feature names.

        Args:
            features: Dictionary of extracted features

        Returns:
            True if features are valid, False otherwise
        """
        return set(features.keys()) == self._feature_names

    @abstractmethod
    def get_feature_ranges(self) -> Dict[str, tuple]:
        """
        Get the expected value ranges for each feature.

        Returns:
            Dictionary mapping feature names to tuples of (min_value, max_value)
        """
        pass