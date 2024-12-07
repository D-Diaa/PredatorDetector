__all__ = ['FeatureExtractor', 'SentimentExtractor', 'EmotionExtractor',
           'ToxicityExtractor', 'IntentExtractor', 'Word2AffectExtractor', 'INTENT_LABELS', 'feature_sets']

from .extractor import FeatureExtractor
from .TransformerExtractors import SentimentExtractor, EmotionExtractor, ToxicityExtractor, IntentExtractor, \
    Word2AffectExtractor, INTENT_LABELS

feature_sets = {
    "vad": ['Valence', 'Arousal', 'Dominance'],
    "vad+": ['Valence', 'Arousal', 'Dominance', 'Age of Acquisition', 'Concreteness'],
    "sentiment": ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral'],
    "emotions": ['emotion_admiration', 'emotion_amusement', 'emotion_anger', 'emotion_annoyance',
                 'emotion_approval', 'emotion_caring', 'emotion_confusion', 'emotion_curiosity',
                 'emotion_desire', 'emotion_disappointment', 'emotion_disapproval', 'emotion_disgust',
                 'emotion_embarrassment', 'emotion_excitement', 'emotion_fear', 'emotion_gratitude',
                 'emotion_grief', 'emotion_intensity', 'emotion_joy', 'emotion_love',
                 'emotion_nervousness', 'emotion_neutral', 'emotion_optimism', 'emotion_pride',
                 'emotion_realization', 'emotion_relief', 'emotion_remorse', 'emotion_sadness',
                 'emotion_surprise'],
    "toxicity": ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack'],
    "intent": [f"intent_{label}" for label in INTENT_LABELS],
    "base": ['message_lengths', 'time_deltas'],
    "best": ['Age of Acquisition', 'intent_meeting_planning', 'emotion_confusion']
}

feature_sets["all"] = sorted(list(set(feat for key in feature_sets for feat in feature_sets[key])))
