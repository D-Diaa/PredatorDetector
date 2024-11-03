import re
from typing import Dict, Any

from src.extractor import FeatureExtractor


class KeywordExtractor(FeatureExtractor):
    """Analyzes the style of interaction for potential warning signs."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.patterns = {
            'questions': r'\?|(?:^|\s)(who|what|when|where|why|how)(?:\s|$)',
            'commands': r'(?:^|\s)(do|don\'t|must|should|tell|give|send|show)(?:\s|$)',
            'personal_info': r'(?:^|\s)(age|name|location|address|phone|school|live|family)(?:\s|$)',
            'agreement_seeking': r'(?:^|\s)(right|okay|agree|correct|understand)(?:\s|\?|$)',
            'isolation': r'(?:^|\s)(just|only|between|secret|private|alone)(?:\s|$)',
            'praise': r'(?:^|\s)(smart|mature|special|beautiful|pretty|nice|cool)(?:\s|$)',
            'urgency': r'(?:^|\s)(now|quick|fast|hurry|soon|tonight|today)(?:\s|$)',
            'aggression': r'\b(hate|stupid|idiot|fool|dumb|moron|shut|kill)\b',
            'politeness': r'(?:^|\s)(please|thank|sorry|excuse|pardon|kind|nice|gentle)(?:\s|$)',
            'secrecy': r'(?:^|\s)(secret|private|don\'t tell|ssshh|quiet|hide)(?:\s|$)',
            'specific_time': r'\b([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]\b',
            'time_reference': r'\b(today|tonight|tomorrow|later|soon|now)\b',
            'day_reference': r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            'urgency_markers': r'\b(asap|urgent|quickly|hurry|rush)\b',
            'duration': r'\b(\d+\s*(min(ute)?s?|hr?s?|hours|days|weeks))\b',
            'scheduling': r'\b(meet|meeting|schedule|appointment|arrange)\b',
            "topic_shift":  r'\b(anyway|moving on|speaking of|by the way|btw|back to|change of topic)\b',
            "inappropriate": r'\b(sex|sexy|nude|naked|horny|hot|kiss|fuck|suck|dick|pussy|boobs|ass|penis|vagina'
                             r'|orgasm|cum|anal|bdsm|bondage|fetish|nipple|clit|cock|butt|booty|tits|vulva|vagina'
                             r'|genital|erotic|porn|xxx|nsfw|nude|naked)\b',
            'control': r'\b(let\'s|should we|why don\'t we|how about)\b',
            'personal': r'\b(feel|hurt|alone|scared|worried|trust)\b',
            'probing': r'\b(why do you|what makes you|tell me about)\b'
        }
        self._feature_names = set(self.patterns.keys())

    def extract(self, text: str) -> Dict[str, float]:
        text = text.lower()
        total_words = len(text.split())
        if total_words == 0:
            return {name: 0.0 for name in self._feature_names}

        features = {}
        for pattern_name, pattern in self.patterns.items():
            features[pattern_name] = len(re.findall(pattern, text)) / total_words
        return features

    def get_feature_ranges(self) -> Dict[str, tuple]:
        return {name: (0.0, 1.0) for name in self._feature_names}