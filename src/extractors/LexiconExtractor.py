import html
import re
from functools import lru_cache
from typing import List, Set, Tuple, Dict
import contractions
import emoji
import nltk
import numpy as np
import pandas as pd
from nltk import PorterStemmer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.corpus import stopwords, wordnet, brown, gutenberg
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

from src.extractor import FeatureExtractor

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('words', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('gutenberg', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


class ConversationTextCleaner:
    """Focused text cleanup for conversation artifacts using NLTK"""

    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))

        # Initialize compound words detector
        self._initialize_compound_detector()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        self.spell_checker = SpellChecker(distance=1)
        # Pre-fetch known words for spell checker to optimize
        self.known_words = self.spell_checker.word_frequency
        self.patterns = {
            'urls': re.compile(r'http[s]?://\S+|www\.\S+'),
            'emails': re.compile(r'\S+@\S+'),
            'mentions': re.compile(r'@\S+'),
            'html_tags': re.compile(r'<.*?>'),
            'multiple_spaces': re.compile(r'\s+'),
            'special_chars': re.compile(r'[^\w\s@#]'),
        }

    def get_stems(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(token) for token in tokens]

    def _initialize_compound_detector(self):
        """Initialize the compound word detector using multiple methods"""
        print("Initializing compound word detector...")

        # 1. Build corpus for collocation analysis
        words = []
        for sent in brown.sents() + gutenberg.sents():
            words.extend([w.lower() for w in sent])

        # 2. Find collocations using statistical measures
        bigram_measures = BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(words)

        # Apply frequency filter
        finder.apply_freq_filter(3)

        # Get top collocations using multiple metrics
        self.common_compounds = set()

        # Using PMI (Pointwise Mutual Information)
        pmi_scored = finder.score_ngrams(bigram_measures.pmi)
        pmi_scored = [(pair, score) for pair, score in pmi_scored if all(w.isalpha() for w in pair)]
        pmi_scored.sort(key=lambda x: -x[1])
        self.common_compounds.update(tuple(pair) for pair, score in pmi_scored[:5000])

        # Using chi square
        chi_scored = finder.score_ngrams(bigram_measures.chi_sq)
        chi_scored = [(pair, score) for pair, score in chi_scored if all(w.isalpha() for w in pair)]
        self.common_compounds.update(tuple(pair) for pair, score in chi_scored[:5000])

        # 3. Add compounds from WordNet
        for synset in list(wordnet.all_synsets()):
            name = synset.name().split('.')[0]
            if '_' in name:
                parts = name.split('_')
                if len(parts) == 2 and all(p.isalpha() for p in parts):  # only consider bigrams
                    self.common_compounds.add(tuple(parts))

        # 4. Build compound validation rules
        self.compound_rules = {
            # Particle verbs (verb + preposition)
            'particle_verbs': {
                'pos_patterns': [('VB', 'IN'), ('VB', 'RP')],
                'common_particles': {'up', 'down', 'in', 'out', 'on', 'off', 'away', 'back'}
            },
            # Noun compounds
            'noun_compounds': {
                'pos_patterns': [('NN', 'NN'), ('JJ', 'NN')],
            }
        }

        print(f"Initialized with {len(self.common_compounds)} potential compound patterns")

    def _validate_compound(self, pair: Tuple[str, str]) -> bool:
        """Validate if a word pair should be treated as a compound"""
        word1, word2 = pair

        # Skip if either word is too short
        if len(word1) < 2 or len(word2) < 2:
            return False

        # Check if it's in our common compounds
        if pair in self.common_compounds:
            return True

        # Get POS tags
        pos_tags = nltk.pos_tag([word1, word2])
        pos_pair = (pos_tags[0][1], pos_tags[1][1])

        # Check particle verbs
        if pos_tags[0][1].startswith('VB') and word2 in self.compound_rules['particle_verbs']['common_particles']:
            return True

        # Check pos patterns
        for rule_type, rule_info in self.compound_rules.items():
            if 'pos_patterns' in rule_info and pos_pair in rule_info['pos_patterns']:
                # Additional validation for noun compounds
                if rule_type == 'noun_compounds':
                    # Check if combined form exists in WordNet
                    combined = word1 + word2
                    if wordnet.synsets(combined):
                        return True
                else:
                    return True

        return False

    def _detect_compounds(self, tokens: List[str]) -> Set[str]:
        """Automatically detect and join compound words"""
        compounds = set()

        for i in range(len(tokens) - 1):
            current_pair = (tokens[i].lower(), tokens[i + 1].lower())

            if self._validate_compound(current_pair):
                # Try different joining strategies
                variants = {
                    current_pair[0] + current_pair[1],  # joined
                    current_pair[0] + '_' + current_pair[1],  # underscored
                }
                compounds.update(variants)

        return compounds

    def get_lemmas(self, tokens: List[str]) -> List[str]:
        """Get lemmatized tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def get_enriched_tokens(self, text: str) -> List[str]:
        """Get enriched tokens including automatic compounds"""
        # Get initial tokens
        base_tokens = self.get_clean_tokens(text)
        if not base_tokens:
            return []
        # Initialize result set
        enriched_tokens = set(base_tokens)
        # Add automatic compounds
        enriched_tokens.update(self._detect_compounds(base_tokens))
        # Add synonyms for each base token and compound
        for token in list(enriched_tokens):
            synonyms = self._get_synonyms(token)
            enriched_tokens.update(synonyms)
        # Add lemmatized tokens
        enriched_tokens.update(self.get_lemmas(base_tokens))
        enriched_tokens.update(self.get_stems(base_tokens))
        # Add spell-checked tokens only if they are not known
        corrected_tokens = [self.spell_checker.correction(token) for token in base_tokens if
                            token not in self.known_words]
        corrected_tokens = [token for token in corrected_tokens if token is not None]
        enriched_tokens.update(corrected_tokens)
        enriched_tokens.update(self._detect_compounds(corrected_tokens))
        return sorted(enriched_tokens)

    @lru_cache(maxsize=10000)
    def _get_synonyms(self, word: str) -> Set[str]:
        """Get the most relevant synonyms using WordNet."""
        synonyms = set()

        # Retrieve all synsets for the given word
        synsets = wordnet.synsets(word)

        if synsets:
            # Consider only the first synset (most common meaning)
            first_synset = synsets[0]
            for lemma in first_synset.lemmas():
                synonym = lemma.name().lower()
                # Exclude the original word and multi-word synonyms
                if synonym != word and '_' not in synonym:
                    synonyms.add(synonym)

        return synonyms

    def expand_contractions(self, text: str) -> str:
        return contractions.fix(text)

    def _replace_emojis(self, text: str) -> str:
        return emoji.demojize(text)

    def get_clean_tokens(self, text: str) -> List[str]:
        """Basic token cleaning"""
        if not text:
            return []

        # Clean text
        cleaned = self._replace_emojis(text)  # Replace emojis with text
        cleaned = html.unescape(cleaned)
        cleaned = self.expand_contractions(cleaned)  # Expand contractions
        for pattern in self.patterns.values():
            cleaned = pattern.sub(' ', cleaned)
        cleaned = cleaned.strip()

        # Tokenize
        tokens = word_tokenize(cleaned.lower())

        # Filter tokens
        clean_tokens = []
        for token in tokens:
            if token.isspace():
                continue
            if self.remove_stopwords and token in self.stop_words:
                continue

            token = re.sub(r'^[^\w]+|[^\w]+$', '', token)

            if token:
                clean_tokens.append(token)

        return clean_tokens


class MultiLexiconEmotionExtractor(FeatureExtractor):
    def __init__(self, lexicon_paths: Dict[str, str] = None):
        super().__init__()
        if lexicon_paths is None:
            lexicon_paths = {
                'affect_intensity': 'lexicons/NRC-Affect-Intensity-Lexicon.csv',
                'emolex': 'lexicons/EmoLex.csv',
                'vad': 'lexicons/NRC-VAD-Lexicon.csv',
            }
        self.cleaner = ConversationTextCleaner()
        # Load lexicons
        self.lexicons = {}
        self.dimension_maps = {}
        for name, path in lexicon_paths.items():
            self._load_lexicon(name, path)

    @property
    def keys(self):
        return self.dimension_maps.keys()

    def _load_lexicon(self, name: str, path: str):
        """Load and prepare a lexicon"""
        # Determine separator based on file extension
        sep = '\t' if path.endswith('.txt') else ','
        lexicon = pd.read_csv(path, sep=sep)

        # Ensure 'word' column exists
        if 'word' not in lexicon.columns:
            raise ValueError(f"Lexicon {name} must have a 'word' column.")

        # Lowercase words
        lexicon['word'] = lexicon['word'].astype(str).str.lower()

        # Lemmatize words using vectorized operation
        lexicon['lemmatized'] = lexicon['word'].apply(self.cleaner.lemmatizer.lemmatize)

        # Prepare the original DataFrame (excluding 'lemmatized')
        original_df = lexicon.drop(columns=['lemmatized'])

        # Prepare the lemmatized DataFrame by renaming 'lemmatized' to 'word' and excluding original 'word'
        lemmatized_df = lexicon[
            ['lemmatized'] + [col for col in lexicon.columns if col not in ['word', 'lemmatized']]].rename(
            columns={'lemmatized': 'word'})

        # Concatenate the original and lemmatized DataFrames
        combined_lexicon = pd.concat([original_df, lemmatized_df], ignore_index=True)

        # Drop duplicate words
        combined_lexicon.drop_duplicates(subset=['word'], inplace=True)

        # Melt the dataframe to handle multiple emotion dimensions
        emotion_columns = [col for col in combined_lexicon.columns if col != 'word']
        lookup = {}
        for col in emotion_columns:
            lookup[f"{col}_{name}"] = dict(zip(combined_lexicon['word'], combined_lexicon[col].astype(float)))

        self.lexicons[name] = lookup
        self.dimension_maps.update({dim: name for dim in lookup.keys()})

    def extract(self, text: str) -> Dict[str, float]:
        """Extract all emotion scores for a single text"""
        tokens = self.cleaner.get_enriched_tokens(text)
        scores = {}

        for dimension, lexicon_name in self.dimension_maps.items():
            lexicon = self.lexicons[lexicon_name]
            dimension_scores = [
                lexicon[dimension][token]
                for token in tokens
                if token in lexicon[dimension]
            ]
            scores[dimension] = np.mean(dimension_scores) if dimension_scores else np.nan

        return scores

    def process_conversation(self, conv_features: Dict) -> Dict:
        """Process entire conversation and add emotion sequences"""
        # Process each message and collect scores
        all_scores = [self.extract(msg) for msg in conv_features['messages']]

        # Convert to sequences for each dimension
        for dimension in self.dimension_maps:
            conv_features[dimension] = [
                scores.get(dimension, np.nan) for scores in all_scores
            ]

        return conv_features

    def get_feature_ranges(self) -> Dict[str, tuple]:
        """Get the expected ranges for each feature"""
        return {dim: (0.0, 1.0) for dim in self.dimension_maps.keys()}


def add_emotion_sequences(conv_features: Dict, lexicon_paths: Dict[str, str]) -> Dict:
    """Add emotion sequences to conversation features"""
    extractor = MultiLexiconEmotionExtractor(lexicon_paths)
    return extractor.process_conversation(conv_features)


# Example usage:
if __name__ == "__main__":
    # Example conversation
    conv_features = {
        'messages': [
            "I'm really excited about this project! 🎉",
            "@user The results are concerning... http://example.com",
            "Let's carefully analyze the implications."
        ],
        'authors': ['user1', 'user2', 'user1'],
        'timestamps': ['2024-01-01 10:00', '2024-01-01 10:01', '2024-01-01 10:02']
    }

    lexicon_paths = {
        'affect_intensity': 'lexicons/NRC-Affect-Intensity-Lexicon.csv',
        'emolex': 'lexicons/EmoLex.csv',
        'vad': 'lexicons/NRC-VAD-Lexicon.csv',
    }

    # Add emotion sequences
    conv_features = add_emotion_sequences(conv_features, lexicon_paths)

    # Print first few scores
    for dim in conv_features:
        if dim.endswith('_scores'):
            print(f"\n{dim}:")
            print(conv_features[dim][:3])
