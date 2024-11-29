import html
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from detoxify import Detoxify
from transformers import AutoTokenizer, pipeline, Pipeline

from extractors import FeatureExtractor
from extractors.word2affect import CustomModel

# Constants
INTENT_LABELS = [
    "trust_building",
    "isolation_attempt",
    "boundary_testing",
    "personal_probing",
    "manipulation",
    "secrecy_pressure",
    "authority_undermining",
    "reward_offering",
    "normalization",
    "meeting_planning"
]

# Environment Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def clean_text(text: str) -> str:
    """
    Cleans the input text by unescaping HTML entities and anonymizing sensitive information.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = html.unescape(text)
    tokens = []
    for token in text.split():
        if token.startswith('@') and len(token) > 1:
            tokens.append('@user')
        elif token.startswith('http'):
            tokens.append('http')
        elif '@' in token and '.' in token and len(token) > 3 and not token.startswith('@'):
            tokens.append('email@email.com')
        else:
            tokens.append(token)
    return " ".join(tokens)


class TransformerExtractor(FeatureExtractor):
    """
    Base class for transformer-based feature extractors.

    Handles multiprocessing, batching, and performance tracking.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes the TransformerExtractor.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary.
        """
        super().__init__(config)
        self.batch_size = self._config.get('batch_size', 32)
        self.tokenizer = None
        self.is_initialized = False

    def _load(self):
        """
        Loads the necessary resources for the extractor.

        Must be implemented by child classes.
        """
        raise NotImplementedError("Child classes must implement the _load method.")

    def chunk_text(self, text: str, max_length: int, overlap: int = 50) -> List[str]:
        """
        Splits text into chunks based on tokenizer's max length with specified overlap.

        Args:
            text (str): The text to be chunked.
            max_length (int): Maximum number of tokens per chunk.
            overlap (int, optional): Number of overlapping tokens between chunks.

        Returns:
            List[str]: List of text chunks.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        chunks_tokens = []
        start = 0

        while start < len(tokens):
            # Calculate end position accounting for EOS token
            end = min(start + max_length - 1, len(tokens))  # -1 to leave room for EOS

            # Get chunk tokens and append EOS token
            chunk_tokens = tokens[start:end]
            chunks_tokens.append(chunk_tokens)

            if end >= len(tokens):
                break

            # Move start position accounting for overlap
            start = end - overlap

        # Decode all chunks at once
        chunks = self.tokenizer.batch_decode(chunks_tokens, skip_special_tokens=True)

        return chunks

    def preprocess(self, texts: List[str], max_length: int) -> List[Tuple[int, str]]:
        """
        Cleans and chunks texts, associating each chunk with its original text index.

        Args:
            texts (List[str]): List of input texts.
            max_length (int): Maximum number of tokens per chunk.

        Returns:
            List[Tuple[int, str]]: List of tuples containing text index and chunk.
        """
        chunked_texts_with_indices = []
        for idx, text in enumerate(texts):
            cleaned = clean_text(text)
            chunks = self.chunk_text(cleaned, max_length)
            if not chunks:
                chunked_texts_with_indices.append((idx, ""))
                continue
            for chunk in chunks:
                chunked_texts_with_indices.append((idx, chunk))
        return chunked_texts_with_indices

    def chunked_predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Processes texts in chunks and aggregates the predictions.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            List[Dict[str, float]]: List of aggregated prediction scores per text.
        """

        chunked_texts_with_indices = self.preprocess(texts, self.get_max_length())
        if not chunked_texts_with_indices:
            return [{k: 0.0 for k in self._feature_names} for _ in texts]

        # Mapping from text index to its chunks' scores
        scores_dict = defaultdict(list)

        # Prepare batches
        batches = []
        current_batch = []
        current_indices = []
        discarded_indices = []
        for idx, chunk in chunked_texts_with_indices:
            if len(chunk) == 0:
                discarded_indices.append(idx)
            else:
                current_batch.append(chunk)
                current_indices.append(idx)
            if len(current_batch) == self.batch_size:
                batches.append((current_indices.copy(), current_batch.copy()))
                current_batch.clear()
                current_indices.clear()
        if current_batch:
            batches.append((current_indices.copy(), current_batch.copy()))

        # Process each batch
        for indices, batch in batches:
            batch_scores = self.predict(batch)
            for idx, scores in zip(indices, batch_scores):
                scores_dict[idx].append(scores)

        # Handle discarded indices by ensuring they have empty score lists
        for idx in discarded_indices:
            if idx not in scores_dict:
                scores_dict[idx] = []

        # Average the scores per text
        averaged_scores = self.average_scores(scores_dict)

        return averaged_scores

    def average_scores(self, scores_dict: Dict[int, List[Dict[str, float]]]) -> List[Dict[str, float]]:
        """
        Averages the scores for each feature across all chunks of a text.

        Args:
            scores_dict (Dict[int, List[Dict[str, float]]]): Mapping from text index to list of score dictionaries.

        Returns:
            List[Dict[str, float]]: List of averaged scores per text.
        """
        averaged = []
        for idx in range(len(scores_dict)):
            scores_list = scores_dict[idx]
            if not scores_list:
                averaged.append({k: 0.0 for k in self._feature_names})
                continue
            temp_avg = {k: 0.0 for k in self._feature_names}
            for scores in scores_list:
                for k, v in scores.items():
                    temp_avg[k] += v
            num = len(scores_list)
            for k in temp_avg:
                temp_avg[k] /= num
            averaged.append(temp_avg)
        return averaged

    def batch_extract(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Extracts features from a batch of texts.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            List[Dict[str, float]]: List of feature dictionaries per text.
        """
        if not self.is_initialized:
            self._load()
            self.is_initialized = True
        return self.chunked_predict(texts)

    def extract(self, text: str) -> Dict[str, float]:
        """
        Extracts features from a single text.

        Args:
            text (str): The input text.

        Returns:
            Dict[str, float]: Dictionary of extracted features.
        """
        results = self.batch_extract([text])
        return results[0] if results else {k: 0.0 for k in self._feature_names}

    def get_max_length(self) -> int:
        """
        Retrieves the maximum token length for the transformer model.

        Must be implemented by child classes.

        Returns:
            int: Maximum number of tokens.
        """
        raise NotImplementedError("Child classes must implement get_max_length method.")

    def get_feature_ranges(self) -> Dict[str, tuple]:
        """
        Retrieves the feature ranges for normalization or analysis.

        Must be implemented by child classes.

        Returns:
            Dict[str, tuple]: Mapping of feature names to their value ranges.
        """
        raise NotImplementedError("Child classes must implement get_feature_ranges method.")

    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Performs prediction on a batch of texts.

        Must be implemented by child classes.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            List[Dict[str, float]]: List of prediction results.
        """
        raise NotImplementedError("Child classes must implement the predict method.")


class Word2AffectExtractor(TransformerExtractor):
    """
    Extractor for linguistic features such as Valence, Arousal, Dominance, etc., using Word2Affect model.
    """

    def __init__(self, config: Dict[str, Any] = None, directory: str = "word2affect_english"):
        """
        Initializes the Word2AffectExtractor.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary.
            directory (str, optional): Directory where the model is stored.
        """
        super().__init__(config)
        self.directory = directory
        self._feature_names = {
            'Valence', 'Arousal', 'Dominance', 'Age of Acquisition', 'Concreteness'
        }

    def _load(self):
        """
        Loads the Word2Affect model and tokenizer.
        """
        self.model = CustomModel.from_pretrained(self.directory).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.directory)

    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predicts linguistic features for a batch of texts.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            List[Dict[str, float]]: List of feature dictionaries.
        """

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(inputs['input_ids'], inputs['attention_mask'])

        results = [{} for _ in range(len(texts))]
        for emotion, ratings in zip(
                ['Valence', 'Arousal', 'Dominance', 'Age of Acquisition', 'Concreteness'],
                outputs
        ):
            for i in range(len(texts)):
                results[i][emotion] = ratings[i].item()

        return results

    def get_max_length(self) -> int:
        """
        Retrieves the maximum token length for the Word2Affect model.

        Returns:
            int: Maximum number of tokens.
        """
        return 504


class ToxicityExtractor(TransformerExtractor):
    """
    Extractor for detecting various types of toxicity in text using Detoxify model.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes the ToxicityExtractor.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary.
        """
        super().__init__(config)
        self._feature_names = {
            'toxicity', 'severe_toxicity', 'obscene', 'threat',
            'insult', 'identity_attack'
        }

    def _load(self):
        """
        Loads the Detoxify pipeline and tokenizer.
        """
        self.pipe = Detoxify('original', device=self.device)
        self.tokenizer = self.pipe.tokenizer

    def get_max_length(self) -> int:
        """
        Retrieves the maximum token length for the Detoxify model.

        Returns:
            int: Maximum number of tokens.
        """
        return self.pipe.model.config.max_position_embeddings - 8  # Typically 512

    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predicts toxicity scores for a batch of texts.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            List[Dict[str, float]]: List of toxicity score dictionaries.
        """

        self.pipe.model.eval()
        inputs = self.pipe.tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True
        ).to(self.pipe.device)
        outputs = self.pipe.model(**inputs)

        scores_list = torch.sigmoid(outputs.logits).cpu().detach().numpy()
        results = []
        for scores in scores_list:
            results.append({
                cla: float(scores[i]) for i, cla in enumerate(self.pipe.class_names)
            })
        return results


class SentimentExtractor(TransformerExtractor):
    """
    Extractor for sentiment analysis using a RoBERTa-based model.
    """

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initializes the SentimentExtractor.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary.
        """
        super().__init__(config)

    def _load(self):
        """
        Loads the sentiment analysis pipeline and tokenizer.
        """
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.analyzer: Pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=self.device,
            batch_size=self.batch_size,
            top_k=None
        )
        self.tokenizer = self.analyzer.tokenizer

        # Initialize sentiment labels
        test_result = self.analyzer("test")
        if isinstance(test_result, list) and isinstance(test_result[0], dict):
            self.sentiment_labels = sorted({r['label'] for r in test_result})
        else:
            # Fallback in case of unexpected format
            self.sentiment_labels = ["positive", "neutral", "negative"]

        self._feature_names = {f'sentiment_{label.lower()}' for label in self.sentiment_labels}
        self._feature_names.add('sentiment_confidence')

    def get_max_length(self) -> int:
        """
        Retrieves the maximum token length for the sentiment model.

        Returns:
            int: Maximum number of tokens.
        """
        return self.analyzer.model.config.max_position_embeddings - 8  # Typically 512

    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predicts sentiment scores for a batch of texts.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            List[Dict[str, float]]: List of sentiment score dictionaries.
        """

        results = self.analyzer(texts)
        processed_results = []
        for result in results:
            scores = {f'sentiment_{label.lower()}': 0.0 for label in self.sentiment_labels}
            if isinstance(result, list):
                for res in result:
                    label = res['label']
                    score = res['score']
                    scores[f'sentiment_{label.lower()}'] = score
                scores['sentiment_confidence'] = max(res['score'] for res in result)
            else:
                raise ValueError("Unexpected sentiment result format.")
            processed_results.append(scores)
        return processed_results


class EmotionExtractor(TransformerExtractor):
    """
    Extractor for emotion detection using a RoBERTa-based model trained on GoEmotions dataset.
    """

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initializes the EmotionExtractor.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary.
        """
        super().__init__(config)

    def _load(self):
        """
        Loads the emotion classification pipeline and tokenizer.
        """
        model_name = "SamLowe/roberta-base-go_emotions"
        self.classifier: Pipeline = pipeline(
            task="text-classification",
            model=model_name,
            device=self.device,
            batch_size=self.batch_size,
            top_k=None
        )
        self.tokenizer = self.classifier.tokenizer

        # Initialize emotion labels
        test_result = self.classifier("test")
        if isinstance(test_result, list) and isinstance(test_result[0], list):
            self.emotion_labels = sorted({r['label'] for r in test_result[0]})
        else:
            # Fallback in case of unexpected format
            self.emotion_labels = ["happy", "sad", "angry", "fear", "surprise", "disgust"]

        self._feature_names = {f'emotion_{e.lower()}' for e in self.emotion_labels}
        self._feature_names.add('emotion_intensity')

    def get_max_length(self) -> int:
        """
        Retrieves the maximum token length for the emotion model.

        Returns:
            int: Maximum number of tokens.
        """
        return self.classifier.model.config.max_position_embeddings - 8  # Typically 512

    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predicts emotion scores for a batch of texts.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            List[Dict[str, float]]: List of emotion score dictionaries.
        """

        results = self.classifier(texts, truncation=True, padding=True)
        processed_results = []
        for result in results:
            scores = {f'emotion_{e.lower()}': 0.0 for e in self.emotion_labels}
            max_score = 0.0
            if isinstance(result, list):
                for res in result:
                    emotion = res['label'].lower()
                    score = res['score']
                    scores[f'emotion_{emotion}'] = score
                    if score > max_score:
                        max_score = score
            processed_results.append({'emotion_intensity': max_score, **scores})
        return processed_results


class IntentExtractor(TransformerExtractor):
    """
    Extractor for intent classification using a BART-based model trained on MNLI.
    """

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initializes the IntentExtractor.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary.
        """
        super().__init__(config)

    def _load(self):
        """
        Loads the intent classification pipeline and tokenizer.
        """
        model_name = "facebook/bart-large-mnli"
        self.analyzer: Pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=self.device,
            batch_size=self.batch_size,
            top_k=None
        )
        self.tokenizer = self.analyzer.tokenizer
        self.intent_labels = INTENT_LABELS
        self._feature_names = {f'intent_{label.lower()}' for label in self.intent_labels}
        self._feature_names.add('intent_confidence')

    def get_max_length(self) -> int:
        """
        Retrieves the maximum token length for the intent model.

        Returns:
            int: Maximum number of tokens.
        """
        return self.analyzer.model.config.max_position_embeddings - 8  # Typically 512

    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predicts intent scores for a batch of texts.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            List[Dict[str, float]]: List of intent score dictionaries.
        """

        results = self.analyzer(texts, self.intent_labels, multi_label=True)
        processed_results = []
        for result in results:
            scores = {f'intent_{label.lower()}': 0.0 for label in self.intent_labels}
            if isinstance(result, dict):
                for label, score in zip(result['labels'], result['scores']):
                    scores[f'intent_{label.lower()}'] = score
                scores['intent_confidence'] = max(result['scores'], default=0.0)
            else:
                # Unexpected format
                scores['intent_confidence'] = 0.0
            processed_results.append(scores)
        return processed_results
